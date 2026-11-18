import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from app.config import Settings, get_settings
from app.line_client import LineClient, verify_signature
from app.qwen_service import QwenService

logger = logging.getLogger(__name__)

DUMMY_REPLY_TOKENS = {
    "00000000000000000000000000000000",
    "ffffffffffffffffffffffffffffffff",
}
RESET_COMMANDS = {"/reset", "/clear"}


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


async def handle_line_event(
    event: dict[str, Any],
    line_client: LineClient,
    qwen_service: QwenService,
) -> None:
    """Process a single LINE webhook event in the background."""
    if event.get("type") != "message":
        return

    reply_token = event.get("replyToken")
    if not reply_token or reply_token in DUMMY_REPLY_TOKENS:
        logger.debug("Skipping event without a real replyToken.")
        return

    source = event.get("source") or {}
    chat_id = extract_chat_id(source)
    message = event.get("message") or {}
    message_type = message.get("type")

    if message_type != "text":
        try:
            await line_client.reply_text(
                reply_token,
                "Currently I only support text messages. Please send a text message!",
            )
        except Exception:
            logger.exception("Failed to send non-text fallback reply for %s", chat_id)
        return

    user_text = str(message.get("text", "")).strip()
    if not user_text:
        return

    if user_text.lower() in RESET_COMMANDS:
        qwen_service.reset_history(chat_id)
        try:
            await line_client.reply_text(reply_token, "Conversation history cleared.")
        except Exception:
            logger.exception("Failed to send reset confirmation for %s", chat_id)
        return

    # Show typing indicator in 1-on-1 chats while Qwen generates the reply
    if source.get("type") == "user" and source.get("userId"):
        asyncio.create_task(
            line_client.start_loading_animation(source["userId"], loading_seconds=60)
        )

    try:
        ai_reply = await qwen_service.generate_reply(chat_id, user_text)
        await line_client.reply_text(reply_token, ai_reply)
    except Exception:
        logger.exception("Error generating or sending reply for %s", chat_id)
        try:
            await line_client.reply_text(
                reply_token,
                "Sorry, an error occurred while generating a response. Please try again.",
            )
        except Exception:
            logger.exception("Failed to send error fallback message for %s", chat_id)


def create_app(
    settings: Settings | None = None,
    line_client: LineClient | None = None,
    qwen_service: QwenService | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    logging.basicConfig(
        level=getattr(logging, resolved_settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

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
        version="1.0.0",
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
