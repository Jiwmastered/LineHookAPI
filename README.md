# LineQwen — LINE Webhook Server with Local Qwen AI

A FastAPI webhook server for the **LINE Messaging API** that receives user messages and responds using a **Qwen** AI model running directly inside the server.

## Features

- **LINE Signature Verification**: Verifies `X-Line-Signature` (HMAC-SHA256) on every incoming webhook request.
- **Automatic Stateless Channel Access Tokens (v3)**: Authenticates with LINE using only `LINE_CHANNEL_ID` and `LINE_CHANNEL_SECRET` via `POST https://api.line.me/oauth2/v3/token`, with automatic caching and refresh (or optional static `LINE_CHANNEL_ACCESS_TOKEN`).
- **In-Process Qwen AI Model**: Runs Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` (configurable to any Qwen2.5/Qwen3 model) locally inside the server on CPU, CUDA, or MPS. Also supports an optional local Ollama backend (`QWEN_BACKEND=ollama`).
- **Responsive UX**: Returns `200 OK` to LINE immediately, shows the LINE typing/loading animation in 1-on-1 chats while generating, and replies asynchronously.
- **Conversation Memory**: Keeps per-chat history (send `/reset` or `/clear` in LINE to reset context).
- **LINE Message Chunking**: Automatically splits long responses to respect LINE's 5,000-character and 5-message reply limits.

## Project Structure

```text
LineQwen/
├── app/
│   ├── config.py        # Environment settings
│   ├── line_client.py   # LINE signature check, v3 stateless token manager, reply & loading APIs
│   ├── qwen_service.py  # Local Qwen inference (transformers) + optional Ollama backend
│   └── main.py          # FastAPI app & /webhook, /health endpoints
├── tests/
│   ├── test_line_client.py
│   └── test_webhook.py
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Configure environment variables by copying `.env.example` to `.env` and filling in your credentials:

   ```powershell
   Copy-Item .env.example .env
   ```

   Key variables in `.env`:
   - `LINE_CHANNEL_ID` — your LINE Messaging API Channel ID
   - `LINE_CHANNEL_SECRET` — your LINE Messaging API Channel Secret
   - `QWEN_MODEL_NAME` — defaults to `Qwen/Qwen2.5-0.5B-Instruct` (change to e.g. `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-3B-Instruct` if you have more RAM/GPU)

## Run the Server

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> On first startup, the server downloads the configured Qwen model weights from Hugging Face and caches them locally.

Check health:

```powershell
curl http://127.0.0.1:8000/health
```

## Connect to LINE Developers Console

LINE requires a public **HTTPS** webhook URL. For local development, expose port `8000` with Cloudflare Tunnel or ngrok:

```powershell
# Using Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8000

# Or using ngrok
ngrok http 8000
```

Then in the **LINE Developers Console** → your Messaging API channel → **Messaging API** tab:
1. Set **Webhook URL** to `https://<your-public-domain>/webhook`.
2. Click **Verify** (should return `200 OK`).
3. Enable **Use webhook**.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
