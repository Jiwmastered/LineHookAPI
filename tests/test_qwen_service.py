import pytest

from app.config import Settings
from app.models_catalog import MODEL_CATALOG, build_model_selector_flex
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
    assert spec_smart.tier == "Smart"
    assert created_instances[1].closed is True
    assert svc.active_model_name == "qwen2.5-1.5b-instruct-q2_k.gguf"

    svc.reset_history("user:1")
    await svc.generate_reply("user:1", "Fresh start")
    assert len(created_instances[2].calls[0]) == 2


def test_model_selector_flex_marks_active_model():
    flex = build_model_selector_flex("qwen2.5-1.5b")
    assert flex["type"] == "flex"
    bubbles = flex["contents"]["contents"]
    assert len(bubbles) == len(MODEL_CATALOG)
    active_headers = [b["header"]["contents"][0]["text"] for b in bubbles]
    assert active_headers.count("ACTIVE MODEL") == 1
    assert active_headers[2] == "ACTIVE MODEL"
