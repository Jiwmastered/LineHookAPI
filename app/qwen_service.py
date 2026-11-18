import asyncio
import gc
import logging
import os
import re
from collections import deque
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings
from app.models_catalog import DEFAULT_MODEL_KEY, MODEL_CATALOG, ModelSpec

logger = logging.getLogger(__name__)

THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
UNCLOSED_THINK_PATTERN = re.compile(r"<think>.*", flags=re.DOTALL)


def clean_model_output(text: str) -> str:
    """Strip reasoning tags (e.g. Qwen3 <think> blocks) and trim whitespace."""
    cleaned = THINK_TAG_PATTERN.sub("", text)
    cleaned = UNCLOSED_THINK_PATTERN.sub("", cleaned).strip()
    return cleaned or text.strip()


def detect_cpu_threads(cap: int = 4) -> int:
    """Detect effective CPU threads respecting Linux cgroup CPU quotas."""
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.exists():
            parts = cpu_max.read_text(encoding="utf-8").strip().split()
            if len(parts) == 2 and parts[0] != "max":
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    return max(1, min(quota // period or 1, cap))
    except Exception:
        pass
    return max(1, min(os.cpu_count() or 2, cap))


class QwenService:
    """Runs Qwen chat models locally in-process (llama.cpp GGUF or transformers) or via Ollama."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._default_model_key: str = (
            settings.qwen_default_model_key
            if settings.qwen_default_model_key in MODEL_CATALOG
            else DEFAULT_MODEL_KEY
        )
        self._chat_models: dict[str, str] = {}
        self._loaded_model_key: str | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._llama: Any = None
        self._device: str = "cpu"
        self._infer_lock = asyncio.Lock()
        self._histories: dict[str, deque[dict[str, str]]] = {}
        self._http: httpx.AsyncClient | None = None

    @property
    def is_loaded(self) -> bool:
        if self.settings.qwen_backend == "ollama":
            return True
        if self.settings.qwen_backend == "llamacpp":
            return self._llama is not None
        return self._model is not None and self._tokenizer is not None

    @property
    def loaded_model_key(self) -> str | None:
        if self.settings.qwen_backend == "ollama":
            return self._default_model_key
        return self._loaded_model_key

    @property
    def active_model_name(self) -> str:
        key = self._loaded_model_key or self._default_model_key
        spec = MODEL_CATALOG.get(key)
        if self.settings.qwen_backend == "ollama":
            return spec.ollama_model if spec else self.settings.ollama_model_name
        if self.settings.qwen_backend == "llamacpp":
            if spec:
                return spec.gguf_filename
            if self.settings.qwen_gguf_path:
                return Path(self.settings.qwen_gguf_path).name
            return f"{self.settings.qwen_gguf_repo}/{self.settings.qwen_gguf_filename}"
        return spec.hf_model if spec else self.settings.qwen_model_name

    def get_chat_model_key(self, chat_id: str) -> str:
        return self._chat_models.get(chat_id, self._default_model_key)

    def get_model_spec(self, model_key: str) -> ModelSpec:
        if model_key not in MODEL_CATALOG:
            raise ValueError(f"Unknown model key: {model_key}")
        return MODEL_CATALOG[model_key]

    async def load(self, model_key: str | None = None) -> None:
        """Load the specified (or default) Qwen model into memory."""
        target_key = model_key or self._default_model_key
        if target_key not in MODEL_CATALOG:
            target_key = self._default_model_key

        if self.settings.qwen_backend == "ollama":
            logger.info(
                "Using Ollama backend at %s with model %s",
                self.settings.ollama_base_url,
                MODEL_CATALOG[target_key].ollama_model,
            )
            return

        async with self._infer_lock:
            await self._ensure_model_loaded_locked(target_key)

    async def select_model(self, chat_id: str, model_key: str) -> ModelSpec:
        """Switch the active model for chat_id and preload it in memory."""
        spec = self.get_model_spec(model_key)
        self._chat_models[chat_id] = model_key

        if self.settings.qwen_backend != "ollama":
            async with self._infer_lock:
                await self._ensure_model_loaded_locked(model_key)

        return spec

    async def _ensure_model_loaded_locked(self, model_key: str) -> None:
        """Ensure model_key is loaded (caller must hold self._infer_lock)."""
        if self._loaded_model_key == model_key and self.is_loaded:
            return

        spec = self.get_model_spec(model_key)
        if self.settings.qwen_backend == "llamacpp":
            await asyncio.to_thread(self._switch_llamacpp_model_sync, spec)
        else:
            await asyncio.to_thread(self._switch_local_model_sync, spec)
        self._loaded_model_key = model_key

    def _unload_current_model_sync(self) -> None:
        """Free the currently loaded model before loading another to avoid OOM."""
        if self._llama is not None:
            try:
                close_fn = getattr(self._llama, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                logger.warning("Failed to close previous Llama instance cleanly.", exc_info=True)
            self._llama = None

        if self._model is not None or self._tokenizer is not None:
            self._model = None
            self._tokenizer = None

        self._loaded_model_key = None
        gc.collect()

    def _resolve_gguf_path(self, spec: ModelSpec) -> str:
        if self.settings.qwen_models_dir:
            candidate = Path(self.settings.qwen_models_dir) / spec.gguf_filename
            if candidate.exists():
                return str(candidate)

        if (
            spec.key == self._default_model_key
            and self.settings.qwen_gguf_path.strip()
            and os.path.exists(self.settings.qwen_gguf_path.strip())
        ):
            return self.settings.qwen_gguf_path.strip()

        from huggingface_hub import hf_hub_download

        logger.info(
            "Downloading GGUF model %s/%s from Hugging Face...",
            spec.gguf_repo,
            spec.gguf_filename,
        )
        download_kwargs: dict[str, Any] = {
            "repo_id": spec.gguf_repo,
            "filename": spec.gguf_filename,
        }
        if self.settings.qwen_models_dir:
            Path(self.settings.qwen_models_dir).mkdir(parents=True, exist_ok=True)
            download_kwargs["local_dir"] = self.settings.qwen_models_dir
        return hf_hub_download(**download_kwargs)

    def _switch_llamacpp_model_sync(self, spec: ModelSpec) -> None:
        from llama_cpp import Llama

        self._unload_current_model_sync()
        model_path = self._resolve_gguf_path(spec)

        n_threads = self.settings.qwen_n_threads or detect_cpu_threads()
        logger.info(
            "Loading GGUF Qwen model '%s' from '%s' (n_ctx=%s, n_threads=%s)...",
            spec.label,
            model_path,
            self.settings.qwen_n_ctx,
            n_threads,
        )

        self._llama = Llama(
            model_path=model_path,
            n_ctx=self.settings.qwen_n_ctx,
            n_threads=n_threads,
            n_threads_batch=n_threads,
            n_batch=128,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        logger.info("GGUF Qwen model '%s' loaded successfully.", spec.label)

    def _switch_local_model_sync(self, spec: ModelSpec) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._unload_current_model_sync()

        model_name = (
            self.settings.qwen_model_name
            if spec.key == self._default_model_key and self.settings.qwen_model_name
            else spec.hf_model
        )
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
        self._unload_current_model_sync()
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def reset_history(self, chat_id: str) -> None:
        self._histories.pop(chat_id, None)

    def _build_messages(self, chat_id: str, user_message: str, spec: ModelSpec) -> list[dict[str, str]]:
        system_prompt = self.settings.qwen_system_prompt
        if spec.disable_thinking and "/no_think" not in system_prompt:
            system_prompt = f"{system_prompt}\n/no_think"

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
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
        """Generate an AI reply for a given chat_id and user message using its selected model."""
        model_key = self.get_chat_model_key(chat_id)
        spec = self.get_model_spec(model_key)
        messages = self._build_messages(chat_id, user_message, spec)

        if self.settings.qwen_backend == "ollama":
            reply = await self._generate_ollama(messages, spec)
        elif self.settings.qwen_backend == "llamacpp":
            async with self._infer_lock:
                await self._ensure_model_loaded_locked(model_key)
                reply = await asyncio.to_thread(self._generate_llamacpp_sync, messages)
        else:
            async with self._infer_lock:
                await self._ensure_model_loaded_locked(model_key)
                reply = await asyncio.to_thread(self._generate_local_sync, messages)

        reply = clean_model_output(reply)
        self._record_turn(chat_id, user_message, reply)
        return reply

    def _generate_llamacpp_sync(self, messages: list[dict[str, str]]) -> str:
        assert self._llama is not None
        response = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=self.settings.qwen_max_new_tokens,
            temperature=self.settings.qwen_temperature,
            top_p=self.settings.qwen_top_p,
            repeat_penalty=1.1,
        )
        choices = response.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()

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

    async def _generate_ollama(self, messages: list[dict[str, str]], spec: ModelSpec) -> str:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=120.0)

        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        response = await self._http.post(
            url,
            json={
                "model": spec.ollama_model,
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
