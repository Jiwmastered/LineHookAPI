from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LINE Messaging API
    line_channel_id: str = Field(default="", description="LINE Messaging API Channel ID")
    line_channel_secret: str = Field(default="", description="LINE Messaging API Channel Secret")
    line_channel_access_token: str = Field(
        default="",
        description="Optional static Channel Access Token. If empty, stateless v3 tokens are issued automatically.",
    )

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # Qwen Model
    qwen_backend: Literal["local", "llamacpp", "ollama"] = Field(default="local")
    qwen_default_model_key: str = Field(default="qwen2.5-0.5b")
    qwen_models_dir: str = Field(
        default="",
        description="Directory containing pre-downloaded .gguf model files (e.g. /app/models).",
    )
    qwen_model_name: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    qwen_gguf_repo: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    qwen_gguf_filename: str = Field(default="qwen2.5-0.5b-instruct-q4_0.gguf")
    qwen_gguf_path: str = Field(
        default="",
        description="Optional local filesystem path to a pre-downloaded .gguf model file.",
    )
    qwen_n_ctx: int = Field(default=2048, ge=256, le=32768)
    qwen_n_threads: int = Field(default=0, ge=0, le=128)
    qwen_system_prompt: str = Field(
        default="You are a helpful, concise AI assistant on LINE. Reply in the same language as the user."
    )
    qwen_max_new_tokens: int = Field(default=512, ge=16, le=4096)
    qwen_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    qwen_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    qwen_lazy_load: bool = Field(default=False)
    qwen_preload_in_background: bool = Field(
        default=False,
        description="Load the Qwen model in a background task on startup so /health responds immediately.",
    )
    max_history_turns: int = Field(default=6, ge=0, le=50)

    # Optional Ollama backend
    ollama_base_url: str = Field(default="http://127.0.0.1:11434")
    ollama_model_name: str = Field(default="qwen2.5:1.5b")

    # Optional Google Gemini API (free tier supported)
    gemini_api_key: str = Field(default="", description="Google AI Studio / Gemini API key")
    gemini_max_output_tokens: int = Field(default=1024, ge=64, le=8192)

    # Optional Giphy API key for live GIF search (curated hilarious GIFs work without a key)
    giphy_api_key: str = Field(default="", description="Optional Giphy API key for dynamic GIF search")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
