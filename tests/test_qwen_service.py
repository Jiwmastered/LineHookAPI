import pytest

from app.config import Settings
from app.qwen_service import QwenService


class FakeLlama:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def create_chat_completion(self, *, messages, max_tokens, temperature, top_p, **kwargs):
        self.calls.append(messages)
        last_user = messages[-1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"<think>brief</think>GGUF reply: {last_user}",
                    }
                }
            ]
        }


@pytest.mark.asyncio
async def test_llamacpp_backend_reply_and_history():
    settings = Settings(
        qwen_backend="llamacpp",
        qwen_gguf_path="/app/models/qwen2.5-0.5b-instruct-q4_0.gguf",
        max_history_turns=2,
    )
    svc = QwenService(settings)
    assert svc.active_model_name == "qwen2.5-0.5b-instruct-q4_0.gguf"
    assert svc.is_loaded is False

    fake_llama = FakeLlama()
    svc._llama = fake_llama
    assert svc.is_loaded is True

    reply1 = await svc.generate_reply("user:1", "Hi")
    assert reply1 == "GGUF reply: Hi"

    reply2 = await svc.generate_reply("user:1", "How are you?")
    assert reply2 == "GGUF reply: How are you?"
    # Second call includes system + turn1 user + turn1 assistant + turn2 user
    assert len(fake_llama.calls[1]) == 4

    svc.reset_history("user:1")
    await svc.generate_reply("user:1", "Fresh start")
    assert len(fake_llama.calls[2]) == 2
