import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import parse_qs

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from app.config import Settings, get_settings
from app.line_client import LineClient, split_text_messages, verify_signature
from app.media_reactions import (
    build_gif_video_message,
    build_sticker_message,
    detect_user_media_request,
    extract_media_tags,
    search_giphy_gif,
)
from app.models_catalog import (
    MODEL_CATALOG,
    build_default_quick_reply,
    build_model_selector_flex,
    get_available_models,
)
from app.qwen_service import QwenService

logger = logging.getLogger(__name__)


def configure_logging(level_name: str) -> None:
    """Route INFO/WARNING logs to stdout and ERROR+ to stderr (prevents Railway false error severity)."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [stdout_handler, stderr_handler]


DUMMY_REPLY_TOKENS = {
    "00000000000000000000000000000000",
    "ffffffffffffffffffffffffffffffff",
}
RESET_COMMANDS = {"/reset", "/clear"}
MODEL_MENU_COMMANDS = {"/model", "/models", "model", "menu", "เปลี่ยนโมเดล", "เลือกโมเดล"}
GIF_COMMAND_PREFIXES = ("/gif", "/meme", "ส่ง gif", "ส่งกิ๊ฟ", "ส่งมีม")
STICKER_COMMAND_PREFIXES = ("/sticker", "ส่งสติกเกอร์", "ส่งสติ๊กเกอร์")


async def _build_ai_reply_messages(
    ai_reply: str,
    user_text: str,
    giphy_api_key: str = "",
    force_sticker_mood: str | None = None,
) -> list[dict[str, Any]]:
    """Convert AI output (with optional [STICKER:...] or [GIF:...] tags) into LINE messages."""
    cleaned_text, tags = extract_media_tags(ai_reply)
    media_msg: dict[str, Any] | None = None

    if tags:
        kind, arg = tags[0]
        if kind == "STICKER":
            media_msg = build_sticker_message(arg)
        elif kind == "GIF":
            media_msg = await search_giphy_gif(arg, giphy_api_key)
            if media_msg is None:
                _, media_msg = build_gif_video_message(arg)
    else:
        requested = detect_user_media_request(user_text)
        if requested == "GIF":
            media_msg = await search_giphy_gif(user_text, giphy_api_key)
            if media_msg is None:
                _, media_msg = build_gif_video_message(user_text)
        elif requested == "STICKER":
            media_msg = build_sticker_message(user_text)
        elif force_sticker_mood is not None:
            media_msg = build_sticker_message(force_sticker_mood)

    messages: list[dict[str, Any]] = []
    if cleaned_text:
        max_text_msgs = 4 if media_msg else 5
        messages.extend(split_text_messages(cleaned_text, max_messages=max_text_msgs))
    if media_msg:
        messages.append(media_msg)
    if not messages:
        messages = split_text_messages(ai_reply)

    messages[-1]["quickReply"] = build_default_quick_reply()
    return messages


def extract_chat_id(source: dict[str, Any]) -> str:
    """Derive a stable conversation identifier from a LINE event source."""
    source_type = source.get("type", "unknown")
    if source_type == "user" and source.get("userId"):
        return f"user:{source['userId']}"
    if source_type == "group" and source.get("groupId"):
        return f"group:{source['groupId']}"
    if source_type == "room" and source.get("roomId"):
        return f"room:{source['roomId']}"
    return f"{source_type}:unknown"


def _maybe_start_loading(source: dict[str, Any], line_client: LineClient) -> None:
    if source.get("type") == "user" and source.get("userId"):
        asyncio.create_task(
            line_client.start_loading_animation(source["userId"], loading_seconds=60)
        )


async def handle_line_event(
    event: dict[str, Any],
    line_client: LineClient,
    qwen_service: QwenService,
) -> None:
    """Process a single LINE webhook event in the background."""
    reply_token = event.get("replyToken")
    if not reply_token or reply_token in DUMMY_REPLY_TOKENS:
        logger.debug("Skipping event without a real replyToken.")
        return

    event_type = event.get("type")
    source = event.get("source") or {}
    chat_id = extract_chat_id(source)

    if event_type == "follow":
        active_key = qwen_service.get_chat_model_key(chat_id)
        try:
            await line_client.reply_messages(
                reply_token,
                [
                    {
                        "type": "text",
                        "text": "Welcome! Pick an AI model below or send any message to start chatting. Type /model anytime to switch models.",
                    },
                    build_model_selector_flex(active_key, gemini_enabled=qwen_service.gemini_enabled),
                ],
            )
        except Exception:
            logger.exception("Failed to send follow welcome card for %s", chat_id)
        return

    if event_type == "postback":
        postback_data = str((event.get("postback") or {}).get("data", ""))
        params = parse_qs(postback_data)
        action = (params.get("action") or [""])[0]
        model_key = (params.get("model") or [""])[0]

        if action == "select_model":
            available = get_available_models(gemini_enabled=qwen_service.gemini_enabled)
            if model_key not in available:
                active_key = qwen_service.get_chat_model_key(chat_id)
                try:
                    await line_client.reply_messages(
                        reply_token,
                        [
                            {"type": "text", "text": "That model is unavailable. Please pick one below:"},
                            build_model_selector_flex(active_key, gemini_enabled=qwen_service.gemini_enabled),
                        ],
                    )
                except Exception:
                    logger.exception("Failed to send invalid-model fallback for %s", chat_id)
                return

            if available[model_key].provider == "local":
                _maybe_start_loading(source, line_client)
            try:
                started_at = time.monotonic()
                spec = await qwen_service.select_model(chat_id, model_key)
                elapsed = time.monotonic() - started_at
                logger.info("Switched %s to model '%s' in %.2fs", chat_id, spec.key, elapsed)
                await line_client.reply_text(
                    reply_token,
                    f"Switched to {spec.label} ({spec.tier}).\nSend a message to start chatting!",
                    quick_reply=build_default_quick_reply(),
                )
            except Exception:
                logger.exception("Error switching model to '%s' for %s", model_key, chat_id)
                try:
                    await line_client.reply_text(
                        reply_token,
                        "Sorry, I could not switch to that model right now. Please try again.",
                        quick_reply=build_default_quick_reply(),
                    )
                except Exception:
                    logger.exception("Failed to send model-switch error fallback for %s", chat_id)
        return

    if event_type != "message":
        return

    message = event.get("message") or {}
    message_type = message.get("type")
    giphy_api_key = getattr(getattr(qwen_service, "settings", None), "giphy_api_key", "")

    if message_type == "sticker":
        keywords = message.get("keywords") or []
        mood_hint = ", ".join(str(k) for k in keywords[:5]) if keywords else "playful"
        synthetic_prompt = (
            f"[The user sent you a LINE sticker with vibe/keywords: {mood_hint}. "
            "React in a fun, short, friendly way!]"
        )
        _maybe_start_loading(source, line_client)
        try:
            started_at = time.monotonic()
            ai_reply = await qwen_service.generate_reply(chat_id, synthetic_prompt)
            elapsed = time.monotonic() - started_at
            logger.info("Reacted to user sticker for %s in %.2fs", chat_id, elapsed)
            reply_msgs = await _build_ai_reply_messages(
                ai_reply,
                user_text="",
                giphy_api_key=giphy_api_key,
                force_sticker_mood=mood_hint,
            )
            await line_client.reply_messages(reply_token, reply_msgs)
        except Exception:
            logger.exception("Failed to react to user sticker for %s", chat_id)
            try:
                sticker_msg = build_sticker_message(mood_hint)
                sticker_msg["quickReply"] = build_default_quick_reply()
                await line_client.reply_messages(reply_token, [sticker_msg])
            except Exception:
                logger.exception("Failed to send fallback sticker for %s", chat_id)
        return

    if message_type != "text":
        try:
            await line_client.reply_text(
                reply_token,
                "Currently I support text messages and stickers. Send a message, /model, /gif, or /sticker!",
                quick_reply=build_default_quick_reply(),
            )
        except Exception:
            logger.exception("Failed to send non-text fallback reply for %s", chat_id)
        return

    user_text = str(message.get("text", "")).strip()
    if not user_text:
        return

    normalized = user_text.lower()

    if normalized in MODEL_MENU_COMMANDS:
        active_key = qwen_service.get_chat_model_key(chat_id)
        try:
            await line_client.reply_messages(
                reply_token,
                [build_model_selector_flex(active_key, gemini_enabled=qwen_service.gemini_enabled)],
            )
        except Exception:
            logger.exception("Failed to send model selector card for %s", chat_id)
        return

    if normalized in RESET_COMMANDS:
        qwen_service.reset_history(chat_id)
        try:
            await line_client.reply_text(
                reply_token,
                "Conversation history cleared.",
                quick_reply=build_default_quick_reply(),
            )
        except Exception:
            logger.exception("Failed to send reset confirmation for %s", chat_id)
        return

    for prefix in GIF_COMMAND_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix + " "):
            mood_arg = user_text[len(prefix) :].strip()
            try:
                gif_msg = await search_giphy_gif(mood_arg, giphy_api_key) if mood_arg else None
                if gif_msg is None:
                    _, gif_msg = build_gif_video_message(mood_arg)
                gif_msg["quickReply"] = build_default_quick_reply()
                await line_client.reply_messages(reply_token, [gif_msg])
            except Exception:
                logger.exception("Failed to send GIF for %s", chat_id)
            return

    for prefix in STICKER_COMMAND_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix + " "):
            mood_arg = user_text[len(prefix) :].strip()
            try:
                sticker_msg = build_sticker_message(mood_arg)
                sticker_msg["quickReply"] = build_default_quick_reply()
                await line_client.reply_messages(reply_token, [sticker_msg])
            except Exception:
                logger.exception("Failed to send sticker for %s", chat_id)
            return

    _maybe_start_loading(source, line_client)

    try:
        started_at = time.monotonic()
        ai_reply = await qwen_service.generate_reply(chat_id, user_text)
        elapsed = time.monotonic() - started_at
        active_key = qwen_service.get_chat_model_key(chat_id)
        logger.info(
            "Generated reply for %s using '%s' in %.2fs (%d chars)",
            chat_id,
            active_key,
            elapsed,
            len(ai_reply),
        )
        reply_msgs = await _build_ai_reply_messages(
            ai_reply,
            user_text=user_text,
            giphy_api_key=giphy_api_key,
        )
        await line_client.reply_messages(reply_token, reply_msgs)
    except Exception:
        logger.exception("Error generating or sending reply for %s", chat_id)
        try:
            await line_client.reply_text(
                reply_token,
                "Sorry, an error occurred while generating a response. Please try again.",
                quick_reply=build_default_quick_reply(),
            )
        except Exception:
            logger.exception("Failed to send error fallback message for %s", chat_id)


def create_app(
    settings: Settings | None = None,
    line_client: LineClient | None = None,
    qwen_service: QwenService | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.line_client = line_client or LineClient(
            channel_id=resolved_settings.line_channel_id,
            channel_secret=resolved_settings.line_channel_secret,
            static_access_token=resolved_settings.line_channel_access_token,
        )
        app.state.qwen_service = qwen_service or QwenService(resolved_settings)
        app.state.model_load_task = None

        async def _preload_model() -> None:
            try:
                await app.state.qwen_service.load()
            except Exception:
                logger.exception("Background Qwen model preload failed.")

        if not resolved_settings.qwen_lazy_load:
            if resolved_settings.qwen_preload_in_background:
                app.state.model_load_task = asyncio.create_task(_preload_model())
            else:
                await app.state.qwen_service.load()

        try:
            yield
        finally:
            if app.state.model_load_task and not app.state.model_load_task.done():
                app.state.model_load_task.cancel()
            if line_client is None:
                await app.state.line_client.aclose()
            if qwen_service is None:
                await app.state.qwen_service.aclose()

    app = FastAPI(
        title="LINE Qwen Webhook Server",
        version="1.1.0",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "service": "LINE Qwen Webhook Server",
            "webhook_endpoint": "/webhook",
            "health_endpoint": "/health",
        }

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        svc: QwenService = request.app.state.qwen_service
        cfg: Settings = request.app.state.settings
        return {
            "status": "ok",
            "backend": cfg.qwen_backend,
            "model": svc.active_model_name,
            "loaded_model_key": svc.loaded_model_key,
            "available_models": list(get_available_models(gemini_enabled=svc.gemini_enabled).keys()),
            "gemini_enabled": svc.gemini_enabled,
            "model_loaded": svc.is_loaded,
        }

    @app.post("/webhook")
    async def webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        x_line_signature: str | None = Header(default=None, alias="X-Line-Signature"),
    ) -> dict[str, str]:
        cfg: Settings = request.app.state.settings
        body = await request.body()

        if not x_line_signature:
            raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

        if not verify_signature(body, x_line_signature, cfg.line_channel_secret):
            raise HTTPException(status_code=403, detail="Invalid X-Line-Signature")

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

        events = payload.get("events", [])
        if not isinstance(events, list):
            raise HTTPException(status_code=400, detail="Invalid events field")

        for event in events:
            if isinstance(event, dict):
                background_tasks.add_task(
                    handle_line_event,
                    event,
                    request.app.state.line_client,
                    request.app.state.qwen_service,
                )

        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run("app.main:app", host=cfg.host, port=cfg.port, reload=False)
