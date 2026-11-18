from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    tier: str
    description: str
    provider: Literal["local", "gemini"]
    accent_color: str
    size_label: str
    hf_model: str = ""
    gguf_repo: str = ""
    gguf_filename: str = ""
    ollama_model: str = ""
    gemini_model: str = ""
    gemini_fallback_model: str = ""
    disable_thinking: bool = False


DEFAULT_MODEL_KEY = "qwen2.5-0.5b"

MODEL_CATALOG: dict[str, ModelSpec] = {
    "qwen2.5-0.5b": ModelSpec(
        key="qwen2.5-0.5b",
        label="Qwen 2.5 0.5B",
        tier="Local · Fast",
        description="Fastest local replies on server for everyday chat.",
        provider="local",
        hf_model="Qwen/Qwen2.5-0.5B-Instruct",
        gguf_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        gguf_filename="qwen2.5-0.5b-instruct-q4_0.gguf",
        size_label="409 MB",
        ollama_model="qwen2.5:0.5b",
        accent_color="#0F766E",
        disable_thinking=False,
    ),
    "qwen3-0.6b": ModelSpec(
        key="qwen3-0.6b",
        label="Qwen 3 0.6B",
        tier="Local · Balanced",
        description="Next-gen local Qwen3 with sharper instruction following.",
        provider="local",
        hf_model="Qwen/Qwen3-0.6B",
        gguf_repo="unsloth/Qwen3-0.6B-GGUF",
        gguf_filename="Qwen3-0.6B-Q4_K_M.gguf",
        size_label="378 MB",
        ollama_model="qwen3:0.6b",
        accent_color="#2563EB",
        disable_thinking=True,
    ),
    "qwen2.5-1.5b": ModelSpec(
        key="qwen2.5-1.5b",
        label="Qwen 2.5 1.5B",
        tier="Local · Smart",
        description="3x larger local model for richer Thai & English replies.",
        provider="local",
        hf_model="Qwen/Qwen2.5-1.5B-Instruct",
        gguf_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        gguf_filename="qwen2.5-1.5b-instruct-q2_k.gguf",
        size_label="718 MB",
        ollama_model="qwen2.5:1.5b",
        accent_color="#7C3AED",
        disable_thinking=False,
    ),
    "gemini-3.5-flash-lite": ModelSpec(
        key="gemini-3.5-flash-lite",
        label="Gemini 3.5 Lite",
        tier="Google · Free",
        description="Fast Google Gemini cloud model with generous free-tier limits.",
        provider="gemini",
        gemini_model="gemini-3.5-flash-lite",
        size_label="Cloud API",
        accent_color="#0284C7",
    ),
    "gemini-3.8-flash": ModelSpec(
        key="gemini-3.8-flash",
        label="Gemini 3.8 Flash",
        tier="Google · Smart",
        description="Flagship Google Gemini Flash for top reasoning, coding & Thai/English.",
        provider="gemini",
        gemini_model="gemini-3.8-flash",
        gemini_fallback_model="gemini-3.5-flash-lite",
        size_label="Cloud API",
        accent_color="#1A73E8",
    ),
}


def get_available_models(gemini_enabled: bool = True) -> dict[str, ModelSpec]:
    """Return models available under the current configuration."""
    if gemini_enabled:
        return dict(MODEL_CATALOG)
    return {k: v for k, v in MODEL_CATALOG.items() if v.provider != "gemini"}


def build_model_selector_flex(
    active_model_key: str,
    gemini_enabled: bool = True,
) -> dict[str, Any]:
    """Build a LINE Flex Message carousel allowing users to pick a local Qwen or Google Gemini model."""
    bubbles: list[dict[str, Any]] = []
    models = get_available_models(gemini_enabled=gemini_enabled)

    for spec in models.values():
        is_active = spec.key == active_model_key
        badge_text = "ACTIVE MODEL" if is_active else spec.tier.upper()
        badge_color = "#4ADE80" if is_active else "#E2E8F0"
        button_label = "Selected" if is_active else f"Use {spec.label}"
        button_style = "secondary" if is_active else "primary"

        button_component: dict[str, Any] = {
            "type": "button",
            "style": button_style,
            "height": "sm",
            "action": {
                "type": "postback",
                "label": button_label[:40],
                "data": f"action=select_model&model={spec.key}",
                "displayText": f"Switch to {spec.label}",
            },
        }
        if not is_active:
            button_component["color"] = spec.accent_color

        bubbles.append(
            {
                "type": "bubble",
                "size": "kilo",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "backgroundColor": spec.accent_color,
                    "paddingAll": "14px",
                    "contents": [
                        {
                            "type": "text",
                            "text": badge_text,
                            "size": "xxs",
                            "weight": "bold",
                            "color": badge_color,
                        },
                        {
                            "type": "text",
                            "text": spec.label,
                            "size": "lg",
                            "weight": "bold",
                            "color": "#FFFFFF",
                            "margin": "xs",
                        },
                    ],
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "spacing": "sm",
                    "paddingAll": "14px",
                    "contents": [
                        {
                            "type": "text",
                            "text": spec.description,
                            "size": "sm",
                            "color": "#334155",
                            "wrap": True,
                        },
                        {
                            "type": "text",
                            "text": f"{spec.tier}  •  {spec.size_label}",
                            "size": "xs",
                            "color": "#64748B",
                            "margin": "md",
                        },
                    ],
                },
                "footer": {
                    "type": "box",
                    "layout": "vertical",
                    "paddingAll": "12px",
                    "contents": [button_component],
                },
            }
        )

    return {
        "type": "flex",
        "altText": "Select an AI model",
        "contents": {
            "type": "carousel",
            "contents": bubbles,
        },
    }


def build_default_quick_reply() -> dict[str, Any]:
    """Quick Reply chips attached to replies so users can switch models, get a GIF, or reset."""
    return {
        "items": [
            {
                "type": "action",
                "action": {
                    "type": "message",
                    "label": "Change Model",
                    "text": "/model",
                },
            },
            {
                "type": "action",
                "action": {
                    "type": "message",
                    "label": "Funny GIF",
                    "text": "/gif",
                },
            },
            {
                "type": "action",
                "action": {
                    "type": "message",
                    "label": "Reset Chat",
                    "text": "/reset",
                },
            },
        ]
    }
