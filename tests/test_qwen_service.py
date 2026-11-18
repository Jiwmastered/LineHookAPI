import json

import httpx
import pytest

from app.config import Settings
from app.models_catalog import MODEL_CATALOG, build_model_selector_flex, get_available_models
from app.qwen_service import QwenService


class FakeLlama:
    def __init__(self, label: str = "fake") -> None:
        self.label = label
        self.closed = False
        self.calls: list[list[dict[str, str]]] = []

    def close(self) -> None:
        self.closed = True

    def create_chat_completion(self, *, messages, max_tokens, temperature, top_p, **kwargs):
        self.calls.append(messages)
        last_user = messages[-1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"<think>brief</think>{self.label} reply: {last_user}",
                    }
                }
            ]
        }


@pytest.mark.asyncio
async def test_llamacpp_backend_reply_history_and_model_switch():
    settings = Settings(
        qwen_backend="llamacpp",
        qwen_default_model_key="qwen2.5-0.5b",
        max_history_turns=2,
    )
    svc = QwenService(settings)
    assert svc.active_model_name == "qwen2.5-0.5b-instruct-q4_0.gguf"
    assert svc.is_loaded is False

    created_instances: list[FakeLlama] = []

    def fake_switch(spec):
        svc._unload_current_model_sync()
        inst = FakeLlama(label=spec.key)
        created_instances.append(inst)
        svc._llama = inst

    svc._switch_llamacpp_model_sync = fake_switch  # type: ignore[method-assign]

    reply1 = await svc.generate_reply("user:1", "Hi")
    assert reply1 == "qwen2.5-0.5b reply: Hi"
    assert svc.loaded_model_key == "qwen2.5-0.5b"

    # Switch to Qwen3 0.6B -> previous model is closed before new one loads, and /no_think is added
    spec = await svc.select_model("user:1", "qwen3-0.6b")
    assert spec.key == "qwen3-0.6b"
    assert created_instances[0].closed is True
    assert svc.loaded_model_key == "qwen3-0.6b"

    reply2 = await svc.generate_reply("user:1", "How are you?")
    assert reply2 == "qwen3-0.6b reply: How are you?"
    # System prompt includes /no_think for Qwen3 and carries conversation history
    assert "/no_think" in created_instances[1].calls[0][0]["content"]
    assert len(created_instances[1].calls[0]) == 4

    # Switch to smarter Qwen2.5 1.5B
    spec_smart = await svc.select_model("user:1", "qwen2.5-1.5b")
    assert "Smart" in spec_smart.tier
    assert created_instances[1].closed is True
    assert svc.active_model_name == "qwen2.5-1.5b-instruct-q2_k.gguf"

    svc.reset_history("user:1")
    await svc.generate_reply("user:1", "Fresh start")
    assert len(created_instances[2].calls[0]) == 2


@pytest.mark.asyncio
async def test_gemini_generation_and_503_fallback():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-goog-api-key") == "test-gemini-key"
        url = str(request.url)
        calls.append(url)
        body = json.loads(request.content.decode("utf-8"))
        assert body["contents"][-1]["role"] == "user"

        if "gemini-3.8-flash" in url:
            return httpx.Response(503, json={"error": {"code": 503, "status": "UNAVAILABLE"}})
        if "gemini-3.5-flash-lite" in url:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "hidden thought", "thought": True},
                                    {"text": "Hello from Google Gemini!"},
                                ]
                            }
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL: {url}")

    settings = Settings(
        qwen_backend="llamacpp",
        gemini_api_key="test-gemini-key",
    )
    svc = QwenService(settings)
    svc._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    # Selecting Gemini does not touch local GGUF loading
    spec = await svc.select_model("user:g1", "gemini-3.8-flash")
    assert spec.provider == "gemini"
    assert svc.is_loaded is False

    # 503 on 3.8 Flash falls back smoothly to 3.5 Flash-Lite
    reply = await svc.generate_reply("user:g1", "Hi Gemini!")
    assert "Hello from Google Gemini!" in reply
    assert "Gemini 3.5 Lite" in reply
    assert any("gemini-3.8-flash" in u for u in calls)
    assert any("gemini-3.5-flash-lite" in u for u in calls)

    await svc.aclose()


def test_model_selector_flex_respects_gemini_flag():
    flex_all = build_model_selector_flex("gemini-3.5-flash-lite", gemini_enabled=True)
    bubbles_all = flex_all["contents"]["contents"]
    assert len(bubbles_all) == len(MODEL_CATALOG)
    active_headers = [b["header"]["contents"][0]["text"] for b in bubbles_all]
    assert active_headers.count("ACTIVE MODEL") == 1
    assert active_headers[3] == "ACTIVE MODEL"

    local_only = get_available_models(gemini_enabled=False)
    assert all(m.provider == "local" for m in local_only.values())
    flex_local = build_model_selector_flex("qwen2.5-1.5b", gemini_enabled=False)
    assert len(flex_local["contents"]["contents"]) == len(local_only)
