import base64
import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class FakeLineClient:
    def __init__(self) -> None:
        self.replies: list[tuple[str, str]] = []
        self.loading_calls: list[tuple[str, int]] = []

    async def start_loading_animation(self, chat_id: str, loading_seconds: int = 20) -> None:
        self.loading_calls.append((chat_id, loading_seconds))

    async def reply_text(self, reply_token: str, text: str) -> None:
        self.replies.append((reply_token, text))

    async def aclose(self) -> None:
        pass


class FakeQwenService:
    def __init__(self) -> None:
        self.loaded = False
        self.prompts: list[tuple[str, str]] = []
        self.resets: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    @property
    def active_model_name(self) -> str:
        return "Qwen/Qwen2.5-0.5B-Instruct"

    async def load(self) -> None:
        self.loaded = True

    def reset_history(self, chat_id: str) -> None:
        self.resets.append(chat_id)

    async def generate_reply(self, chat_id: str, user_message: str) -> str:
        self.prompts.append((chat_id, user_message))
        return f"Qwen echo: {user_message}"

    async def aclose(self) -> None:
        pass


def make_signature(body: bytes, secret: str) -> str:
    return base64.b64encode(
        hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode("utf-8")


def build_test_app():
    secret = "test_secret_key"
    settings = Settings(
        line_channel_id="12345",
        line_channel_secret=secret,
        qwen_lazy_load=False,
    )
    fake_line = FakeLineClient()
    fake_qwen = FakeQwenService()
    app = create_app(settings=settings, line_client=fake_line, qwen_service=fake_qwen)
    return app, secret, fake_line, fake_qwen


def test_health_endpoint():
    app, _, _, fake_qwen = build_test_app()
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert fake_qwen.loaded is True


def test_webhook_rejects_missing_or_invalid_signature():
    app, _, _, _ = build_test_app()
    with TestClient(app) as client:
        res_missing = client.post("/webhook", content=b'{"events":[]}')
        assert res_missing.status_code == 400

        res_invalid = client.post(
            "/webhook",
            content=b'{"events":[]}',
            headers={"X-Line-Signature": "bad-sig"},
        )
        assert res_invalid.status_code == 403


def test_webhook_verify_empty_events_and_dummy_token():
    app, secret, fake_line, fake_qwen = build_test_app()
    payload = {
        "destination": "U123",
        "events": [
            {
                "type": "message",
                "replyToken": "00000000000000000000000000000000",
                "source": {"type": "user", "userId": "U999"},
                "message": {"type": "text", "text": "hello"},
            }
        ],
    }
    raw = json.dumps(payload).encode("utf-8")

    with TestClient(app) as client:
        res = client.post(
            "/webhook",
            content=raw,
            headers={"X-Line-Signature": make_signature(raw, secret)},
        )
        assert res.status_code == 200
        assert fake_qwen.prompts == []
        assert fake_line.replies == []


def test_webhook_text_message_and_reset_command():
    app, secret, fake_line, fake_qwen = build_test_app()
    payload = {
        "events": [
            {
                "type": "message",
                "replyToken": "reply-tok-1",
                "source": {"type": "user", "userId": "U1001"},
                "message": {"type": "text", "text": "Hello Qwen!"},
            },
            {
                "type": "message",
                "replyToken": "reply-tok-2",
                "source": {"type": "user", "userId": "U1001"},
                "message": {"type": "text", "text": "/reset"},
            },
        ]
    }
    raw = json.dumps(payload).encode("utf-8")

    with TestClient(app) as client:
        res = client.post(
            "/webhook",
            content=raw,
            headers={"X-Line-Signature": make_signature(raw, secret)},
        )
        assert res.status_code == 200

    assert fake_qwen.prompts == [("user:U1001", "Hello Qwen!")]
    assert fake_qwen.resets == ["user:U1001"]
    assert fake_line.replies == [
        ("reply-tok-1", "Qwen echo: Hello Qwen!"),
        ("reply-tok-2", "Conversation history cleared."),
    ]
