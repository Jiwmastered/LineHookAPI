import asyncio
import logging
import re
from collections import deque
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def clean_model_output(text: str) -> str:
    """Strip reasoning tags (e.g. Qwen3 <think> blocks) and trim whitespace."""
    cleaned = THINK_TAG_PATTERN.sub("", text).strip()
    return cleaned or text.strip()


class QwenService:
    """Runs a Qwen chat model locally in-process (transformers) or via Ollama."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: str = "cpu"
        self._load_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()
        self._histories: dict[str, deque[dict[str, str]]] = {}
        self._http: httpx.AsyncClient | None = None

    @property
    def is_loaded(self) -> bool:
        if self.settings.qwen_backend == "ollama":
            return True
        return self._model is not None and self._tokenizer is not None

    @property
    def active_model_name(self) -> str:
        if self.settings.qwen_backend == "ollama":
            return self.settings.ollama_model_name
        return self.settings.qwen_model_name

    async def load(self) -> None:
        """Load the Qwen model into memory (local backend only)."""
        if self.settings.qwen_backend != "local":
            logger.info(
                "Using Ollama backend at %s with model %s",
                self.settings.ollama_base_url,
                self.settings.ollama_model_name,
            )
            return

        if self.is_loaded:
            return

        async with self._load_lock:
            if self.is_loaded:
                return
            await asyncio.to_thread(self._load_local_model_sync)

    def _load_local_model_sync(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.settings.qwen_model_name
        if torch.cuda.is_available():
            self._device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
            dtype = torch.float16
        else:
            self._device = "cpu"
            dtype = torch.float32

        logger.info("Loading Qwen model '%s' on device=%s (dtype=%s)...", model_name, self._device, dtype)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
        model.to(self._device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        logger.info("Qwen model '%s' loaded successfully on %s.", model_name, self._device)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def reset_history(self, chat_id: str) -> None:
        self._histories.pop(chat_id, None)

    def _build_messages(self, chat_id: str, user_message: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.settings.qwen_system_prompt}
        ]
        history = self._histories.get(chat_id)
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _record_turn(self, chat_id: str, user_message: str, assistant_reply: str) -> None:
        if self.settings.max_history_turns <= 0:
            return
        max_items = self.settings.max_history_turns * 2
        history = self._histories.get(chat_id)
        if history is None:
            history = deque(maxlen=max_items)
            self._histories[chat_id] = history
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_reply})

    async def generate_reply(self, chat_id: str, user_message: str) -> str:
        """Generate an AI reply for a given chat_id and user message."""
        messages = self._build_messages(chat_id, user_message)

        if self.settings.qwen_backend == "ollama":
            reply = await self._generate_ollama(messages)
        else:
            if not self.is_loaded:
                await self.load()
            async with self._infer_lock:
                reply = await asyncio.to_thread(self._generate_local_sync, messages)

        reply = clean_model_output(reply)
        self._record_turn(chat_id, user_message, reply)
        return reply

    def _generate_local_sync(self, messages: list[dict[str, str]]) -> str:
        import torch

        assert self._tokenizer is not None and self._model is not None

        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )

        model_inputs = self._tokenizer([prompt_text], return_tensors="pt").to(self._device)

        do_sample = self.settings.qwen_temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.settings.qwen_max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.settings.qwen_temperature
            gen_kwargs["top_p"] = self.settings.qwen_top_p

        with torch.inference_mode():
            generated_ids = self._model.generate(**model_inputs, **gen_kwargs)

        input_length = model_inputs["input_ids"].shape[1]
        output_ids = generated_ids[0][input_length:]
        return self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    async def _generate_ollama(self, messages: list[dict[str, str]]) -> str:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=120.0)

        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        response = await self._http.post(
            url,
            json={
                "model": self.settings.ollama_model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.settings.qwen_temperature,
                    "top_p": self.settings.qwen_top_p,
                    "num_predict": self.settings.qwen_max_new_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("message", {}).get("content", "")).strip()
