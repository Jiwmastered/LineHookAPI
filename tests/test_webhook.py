import base64
import hashlib
import hmac
import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models_catalog import DEFAULT_MODEL_KEY, MODEL_CATALOG, ModelSpec


class FakeLineClient:
    def __init__(self) -> None:
        self.replies: list[tuple[str, str]] = []
        self.raw_replies: list[tuple[str, list[dict]]] = []
        self.loading_calls: list[tuple[str, int]] = []

    async def start_loading_animation(self, chat_id: str, loading_seconds: int = 20) -> None:
        self.loading_calls.append((chat_id, loading_seconds))

    async def reply_messages(self, reply_token: str, messages: list[dict]) -> None:
        self.raw_replies.append((reply_token, messages))

    async def reply_text(
        self,
        reply_token: str,
        text: str,
        quick_reply: dict | None = None,
    ) -> None:
        self.replies.append((reply_token, text))
        msg: dict = {"type": "text", "text": text}
        if quick_reply:
            msg["quickReply"] = quick_reply
        self.raw_replies.append((reply_token, [msg]))

    async def aclose(self) -> None:
        pass


class FakeQwenService:
    def __init__(self) -> None:
        self.loaded = False
        self.loaded_key: str | None = None
        self.chat_models: dict[str, str] = {}
        self.prompts: list[tuple[str, str, str]] = []
        self.resets: list[str] = []

    @property
    def gemini_enabled(self) -> bool:
        return True

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    @property
    def loaded_model_key(self) -> str | None:
        return self.loaded_key

    @property
    def active_model_name(self) -> str:
        key = self.loaded_key or DEFAULT_MODEL_KEY
        return MODEL_CATALOG[key].gguf_filename

    def get_chat_model_key(self, chat_id: str) -> str:
        return self.chat_models.get(chat_id, DEFAULT_MODEL_KEY)

    async def load(self, model_key: str | None = None) -> None:
        self.loaded = True
        self.loaded_key = model_key or DEFAULT_MODEL_KEY

    async def select_model(self, chat_id: str, model_key: str) -> ModelSpec:
        spec = MODEL_CATALOG[model_key]
        self.chat_models[chat_id] = model_key
        self.loaded = True
        self.loaded_key = model_key
        return spec

    def reset_history(self, chat_id: str) -> None:
        self.resets.append(chat_id)

    async def generate_reply(self, chat_id: str, user_message: str) -> str:
        key = self.get_chat_model_key(chat_id)
        self.prompts.append((chat_id, key, user_message))
        return f"[{key}] Qwen echo: {user_message}"

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
        assert data["loaded_model_key"] == DEFAULT_MODEL_KEY
        assert set(data["available_models"]) == set(MODEL_CATALOG.keys())
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

    assert fake_qwen.prompts == [("user:U1001", "qwen2.5-0.5b", "Hello Qwen!")]
    assert fake_qwen.resets == ["user:U1001"]
    assert fake_line.replies == [
        ("reply-tok-1", "[qwen2.5-0.5b] Qwen echo: Hello Qwen!"),
        ("reply-tok-2", "Conversation history cleared."),
    ]
    # Normal reply attaches Quick Reply chips for changing model / resetting chat
    quick_items = fake_line.raw_replies[0][1][0]["quickReply"]["items"]
    assert [item["action"]["text"] for item in quick_items] == ["/model", "/reset"]


def test_webhook_model_card_and_postback_switch():
    app, secret, fake_line, fake_qwen = build_test_app()
    payload = {
        "events": [
            {
                "type": "message",
                "replyToken": "tok-menu",
                "source": {"type": "user", "userId": "U2002"},
                "message": {"type": "text", "text": "/model"},
            },
            {
                "type": "postback",
                "replyToken": "tok-switch",
                "source": {"type": "user", "userId": "U2002"},
                "postback": {"data": "action=select_model&model=qwen2.5-1.5b"},
            },
            {
                "type": "message",
                "replyToken": "tok-chat",
                "source": {"type": "user", "userId": "U2002"},
                "message": {"type": "text", "text": "Solve 12 * 12"},
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

    # 1. /model sends Flex carousel with all 3 models
    token0, msgs0 = fake_line.raw_replies[0]
    assert token0 == "tok-menu"
    assert msgs0[0]["type"] == "flex"
    bubbles = msgs0[0]["contents"]["contents"]
    assert len(bubbles) == len(MODEL_CATALOG)

    # 2. postback switches model and sends confirmation
    assert fake_qwen.get_chat_model_key("user:U2002") == "qwen2.5-1.5b"
    assert fake_line.replies[0][0] == "tok-switch"
    assert "Qwen 2.5 1.5B" in fake_line.replies[0][1]

    # 3. subsequent message uses the newly selected smarter model
    assert fake_qwen.prompts == [("user:U2002", "qwen2.5-1.5b", "Solve 12 * 12")]


def test_webhook_follow_sends_welcome_and_model_card():
    app, secret, fake_line, _ = build_test_app()
    payload = {
        "events": [
            {
                "type": "follow",
                "replyToken": "tok-follow",
                "source": {"type": "user", "userId": "U3003"},
            }
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

    assert len(fake_line.raw_replies) == 1
    token, messages = fake_line.raw_replies[0]
    assert token == "tok-follow"
    assert messages[0]["type"] == "text"
    assert messages[1]["type"] == "flex"


def test_background_preload_allows_immediate_health_check():
    secret = "test_secret_key"
    settings = Settings(
        line_channel_id="12345",
        line_channel_secret=secret,
        qwen_lazy_load=False,
        qwen_preload_in_background=True,
    )
    fake_line = FakeLineClient()
    fake_qwen = FakeQwenService()
    app = create_app(settings=settings, line_client=fake_line, qwen_service=fake_qwen)

    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    assert fake_qwen.loaded is True
