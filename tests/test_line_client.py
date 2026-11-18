import base64
import hashlib
import hmac
import json

import httpx
import pytest

from app.line_client import (
    LINE_OAUTH_V3_TOKEN_URL,
    LINE_REPLY_MESSAGE_URL,
    LineClient,
    split_text_messages,
    verify_signature,
)
from app.qwen_service import clean_model_output


def sign(body: bytes, secret: str) -> str:
    return base64.b64encode(
        hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode("utf-8")


def test_verify_signature_valid_and_invalid():
    secret = "test_channel_secret"
    body = b'{"events":[]}'
    valid_sig = sign(body, secret)

    assert verify_signature(body, valid_sig, secret) is True
    assert verify_signature(body, "invalid-signature", secret) is False
    assert verify_signature(body, valid_sig, "wrong_secret") is False
    assert verify_signature(body, "", secret) is False


def test_split_text_messages_handles_empty_and_long_text():
    empty_msgs = split_text_messages("   ")
    assert len(empty_msgs) == 1
    assert empty_msgs[0]["type"] == "text"
    assert empty_msgs[0]["text"]

    long_text = "word " * 3000  # 15,000 chars
    msgs = split_text_messages(long_text, max_length=5000, max_messages=5)
    assert 3 <= len(msgs) <= 5
    assert all(len(m["text"]) <= 5000 for m in msgs)

    huge_text = "x" * 30000
    capped = split_text_messages(huge_text, max_length=5000, max_messages=5)
    assert len(capped) == 5
    assert capped[-1]["text"].endswith("...")


def test_clean_model_output_strips_think_tags():
    raw = "<think>internal reasoning</think>\nHello from Qwen!"
    assert clean_model_output(raw) == "Hello from Qwen!"
    assert clean_model_output("Plain answer") == "Plain answer"


@pytest.mark.asyncio
async def test_stateless_token_caching_and_retry_on_401():
    token_calls = 0
    reply_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls, reply_calls
        if str(request.url) == LINE_OAUTH_V3_TOKEN_URL:
            token_calls += 1
            return httpx.Response(
                200,
                json={
                    "access_token": f"token-{token_calls}",
                    "expires_in": 899,
                    "token_type": "Bearer",
                },
            )
        if str(request.url) == LINE_REPLY_MESSAGE_URL:
            reply_calls += 1
            auth = request.headers.get("Authorization")
            if reply_calls == 1:
                assert auth == "Bearer token-1"
                return httpx.Response(401, json={"message": "expired"})
            assert auth == "Bearer token-2"
            payload = json.loads(request.content.decode("utf-8"))
            assert payload["replyToken"] == "real-token"
            assert payload["messages"] == [{"type": "text", "text": "Hi!"}]
            return httpx.Response(200, json={})
        raise AssertionError(f"Unexpected request URL: {request.url}")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = LineClient(
            channel_id="1234567890",
            channel_secret="secret123",
            http_client=http_client,
        )
        # First reply gets 401 -> refreshes token -> succeeds
        await client.reply_text("real-token", "Hi!")
        # Second token call uses cached token-2
        token = await client.get_access_token()
        assert token == "token-2"
        assert token_calls == 2
        assert reply_calls == 2
