# LineQwen — LINE Webhook Server with Local Qwen AI

A FastAPI webhook server for the **LINE Messaging API** that receives user messages and responds using a **Qwen** AI model running directly inside the server.

## Features

- **LINE Signature Verification**: Verifies `X-Line-Signature` (HMAC-SHA256) on every incoming webhook request.
- **Automatic Stateless Channel Access Tokens (v3)**: Authenticates with LINE using only `LINE_CHANNEL_ID` and `LINE_CHANNEL_SECRET` via `POST https://api.line.me/oauth2/v3/token`, with automatic caching and refresh (or optional static `LINE_CHANNEL_ACCESS_TOKEN`).
- **In-Process Qwen AI Model**: Runs Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` (configurable to any Qwen2.5/Qwen3 model) locally inside the server on CPU, CUDA, or MPS. Also supports an optional local Ollama backend (`QWEN_BACKEND=ollama`).
- **Container-Ready**: Includes `Dockerfile`, `docker-compose.yml`, `railway.toml`, and `render.yaml` with CPU-only PyTorch, baked-in model weights, and non-blocking startup (`QWEN_PRELOAD_IN_BACKGROUND=true`).
- **Responsive UX**: Returns `200 OK` to LINE immediately, shows the LINE typing/loading animation in 1-on-1 chats while generating, and replies asynchronously.
- **Conversation Memory**: Keeps per-chat history (send `/reset` or `/clear` in LINE to reset context).
- **LINE Message Chunking**: Automatically splits long responses to respect LINE's 5,000-character and 5-message reply limits.

> **Note on Hosting (Vercel vs. Container Hosts)**
> Because this server runs PyTorch + Hugging Face `transformers` + Qwen model weights (~1 GB) **locally in-process**, it requires a container/VM host with at least **2 GB RAM** (4 GB recommended for `1.5B`+ models). Serverless platforms like Vercel Functions have a 250 MB bundle limit and read-only filesystems and will fail with `500 FUNCTION_INVOCATION_FAILED`. Use **Railway**, **Hugging Face Spaces (Docker)**, **Render**, or **Docker / VPS** below.

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
├── Dockerfile           # Container image with CPU PyTorch + preloaded Qwen weights
├── docker-compose.yml   # Local / VPS Docker deployment
├── railway.toml         # Railway deployment config
├── render.yaml          # Render Blueprint config
├── .env.example
├── requirements.txt
└── README.md
```

## Deploy to a Container Host

Set these environment variables on your host:
- `LINE_CHANNEL_ID` — your LINE Messaging API Channel ID
- `LINE_CHANNEL_SECRET` — your LINE Messaging API Channel Secret
- `QWEN_MODEL_NAME` *(optional)* — defaults to `Qwen/Qwen2.5-0.5B-Instruct`

### Option 1: Railway (Recommended — Auto-detects `Dockerfile` & `railway.toml`)
1. Go to [Railway](https://railway.com) → **New Project** → **Deploy from GitHub repo** → select `Jiwmastered/LineHookAPI`.
2. In **Variables**, add `LINE_CHANNEL_ID` and `LINE_CHANNEL_SECRET`.
3. Under **Settings → Networking**, generate a public domain (e.g., `https://<app>.up.railway.app`).
4. Set your LINE Webhook URL to `https://<app>.up.railway.app/webhook`.

### Option 2: Hugging Face Spaces (Free 16 GB RAM CPU Docker Tier)
1. Create a new **Space** on Hugging Face and choose **Docker** (Blank) as the SDK.
2. Push this repository (or sync from GitHub) to the Space.
3. Add a `README.md` YAML header with `app_port: 8000` (or set `PORT=7860` in Space Variables).
4. In Space **Settings → Variables and secrets**, add `LINE_CHANNEL_ID` and `LINE_CHANNEL_SECRET` as Secrets.
5. Set your LINE Webhook URL to `https://<your-user>-<your-space>.hf.space/webhook`.

### Option 3: Render (`render.yaml`)
1. Go to [Render](https://render.com) → **New** → **Blueprint** → connect `Jiwmastered/LineHookAPI`.
2. Fill in `LINE_CHANNEL_ID` and `LINE_CHANNEL_SECRET` when prompted.
3. Set your LINE Webhook URL to `https://<your-service>.onrender.com/webhook`.

### Option 4: Docker / VPS
```bash
cp .env.example .env
# Edit .env with LINE_CHANNEL_ID and LINE_CHANNEL_SECRET
docker compose up -d --build
```
Then place a reverse proxy with HTTPS (Caddy, Nginx + Certbot, or Cloudflare Tunnel) in front of port `8000`.

## Local Development (Without Docker)

1. Create a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in `LINE_CHANNEL_ID` and `LINE_CHANNEL_SECRET`.
3. Start the server:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
4. Expose port `8000` over HTTPS for local testing:
   ```powershell
   cloudflared tunnel --url http://localhost:8000
   ```

## Connect to LINE Developers Console

In **LINE Developers Console** → your Messaging API channel → **Messaging API** tab:
1. Set **Webhook URL** to `https://<your-public-domain>/webhook`.
2. Click **Verify** (should return `200 OK`).
3. Enable **Use webhook**.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
