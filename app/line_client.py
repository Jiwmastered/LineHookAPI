import asyncio
import base64
import hashlib
import hmac
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

LINE_OAUTH_V3_TOKEN_URL = "https://api.line.me/oauth2/v3/token"
LINE_REPLY_MESSAGE_URL = "https://api.line.me/v2/bot/message/reply"
LINE_LOADING_ANIMATION_URL = "https://api.line.me/v2/bot/chat/loading/start"

LINE_MAX_TEXT_LENGTH = 5000
LINE_MAX_MESSAGES_PER_REPLY = 5
TOKEN_REFRESH_MARGIN_SECONDS = 60


def verify_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """Verify the X-Line-Signature header using HMAC-SHA256."""
    if not signature or not channel_secret:
        return False

    digest = hmac.new(
        channel_secret.encode("utf-8"),
        body,
        hashlib.sha256,
    ).digest()
    expected_signature = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected_signature, signature)


def split_text_messages(
    text: str,
    max_length: int = LINE_MAX_TEXT_LENGTH,
    max_messages: int = LINE_MAX_MESSAGES_PER_REPLY,
) -> list[dict[str, str]]:
    """Split a reply into LINE-compliant text message objects."""
    cleaned = (text or "").strip()
    if not cleaned:
        cleaned = "Sorry, I could not generate a response."

    chunks: list[str] = []
    remaining = cleaned

    while remaining and len(chunks) < max_messages:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            remaining = ""
            break

        # Prefer splitting on newline or whitespace near the limit
        split_idx = remaining.rfind("\n", 0, max_length)
        if split_idx < max_length // 2:
            split_idx = remaining.rfind(" ", 0, max_length)
        if split_idx <= 0:
            split_idx = max_length

        chunks.append(remaining[:split_idx].rstrip())
        remaining = remaining[split_idx:].lstrip()

    if remaining and chunks:
        # Indicate truncation if the total response exceeded LINE's 5-message cap
        suffix = "..."
        last = chunks[-1]
        if len(last) + len(suffix) > max_length:
            last = last[: max_length - len(suffix)]
        chunks[-1] = last + suffix

    return [{"type": "text", "text": chunk} for chunk in chunks]


class LineClient:
    """Async LINE Messaging API client with automatic Stateless Channel Access Token (v3) management."""

    def __init__(
        self,
        channel_id: str,
        channel_secret: str,
        static_access_token: str = "",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.channel_id = channel_id
        self.channel_secret = channel_secret
        self.static_access_token = static_access_token.strip()
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=30.0)
        self._token_lock = asyncio.Lock()
        self._cached_token: str | None = None
        self._token_expires_at: float = 0.0

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def get_access_token(self, force_refresh: bool = False) -> str:
        """Return a valid Channel Access Token, issuing a v3 stateless token when needed."""
        if self.static_access_token:
            return self.static_access_token

        now = time.monotonic()
        if not force_refresh and self._cached_token and now < self._token_expires_at:
            return self._cached_token

        async with self._token_lock:
            now = time.monotonic()
            if not force_refresh and self._cached_token and now < self._token_expires_at:
                return self._cached_token

            if not self.channel_id or not self.channel_secret:
                raise ValueError(
                    "LINE_CHANNEL_ID and LINE_CHANNEL_SECRET must be set to issue a stateless access token."
                )

            response = await self._http.post(
                LINE_OAUTH_V3_TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.channel_id,
                    "client_secret": self.channel_secret,
                },
            )
            response.raise_for_status()
            payload = response.json()

            token = payload["access_token"]
            expires_in = int(payload.get("expires_in", 899))
            ttl = max(expires_in - TOKEN_REFRESH_MARGIN_SECONDS, 30)

            self._cached_token = token
            self._token_expires_at = time.monotonic() + ttl
            logger.debug("Issued new LINE stateless channel access token (expires_in=%ss)", expires_in)
            return token

    async def _post_with_auth(self, url: str, json_body: dict[str, Any]) -> httpx.Response:
        token = await self.get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = await self._http.post(url, headers=headers, json=json_body)

        # Retry once with a refreshed token on 401 when using stateless tokens
        if response.status_code == 401 and not self.static_access_token:
            logger.warning("LINE API returned 401; refreshing stateless token and retrying.")
            token = await self.get_access_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            response = await self._http.post(url, headers=headers, json=json_body)

        return response

    async def start_loading_animation(self, chat_id: str, loading_seconds: int = 20) -> None:
        """Display the LINE typing/loading animation in a 1-on-1 chat (best-effort)."""
        clamped_seconds = max(5, min(60, loading_seconds))
        try:
            response = await self._post_with_auth(
                LINE_LOADING_ANIMATION_URL,
                {"chatId": chat_id, "loadingSeconds": clamped_seconds},
            )
            if response.status_code >= 400:
                logger.warning(
                    "Failed to start loading animation (status=%s): %s",
                    response.status_code,
                    response.text,
                )
        except Exception as exc:
            logger.warning("Error starting LINE loading animation: %s", exc)

    async def reply_messages(self, reply_token: str, messages: list[dict[str, Any]]) -> None:
        """Send arbitrary LINE message objects (text, flex, etc.) via the Reply Message API."""
        response = await self._post_with_auth(
            LINE_REPLY_MESSAGE_URL,
            {
                "replyToken": reply_token,
                "messages": messages[:LINE_MAX_MESSAGES_PER_REPLY],
            },
        )
        response.raise_for_status()

    async def reply_text(
        self,
        reply_token: str,
        text: str,
        quick_reply: dict[str, Any] | None = None,
    ) -> None:
        """Send one or more text messages using the LINE Reply Message API."""
        messages: list[dict[str, Any]] = split_text_messages(text)
        if quick_reply and messages:
            messages[-1]["quickReply"] = quick_reply
        await self.reply_messages(reply_token, messages)
