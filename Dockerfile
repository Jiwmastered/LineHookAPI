FROM python:3.12-slim

ARG PRELOAD_MODELS=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    QWEN_BACKEND=llamacpp \
    QWEN_DEFAULT_MODEL_KEY=qwen2.5-0.5b \
    QWEN_MODELS_DIR=/app/models \
    QWEN_N_CTX=1024 \
    QWEN_MAX_NEW_TOKENS=192 \
    QWEN_PRELOAD_IN_BACKGROUND=false \
    PORT=8000

WORKDIR /app

# libgomp1 is required by the prebuilt llama-cpp-python OpenMP CPU wheel
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser \
    && mkdir -p /app/.cache/huggingface /app/models \
    && chown -R appuser:appuser /app

# Install lightweight web stack + prebuilt CPU wheel for llama-cpp-python
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt \
    && pip install --no-cache-dir --only-binary=:all: \
       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
       llama-cpp-python

COPY --chown=appuser:appuser app ./app

USER appuser

# Pre-download all selectable Qwen GGUF models into /app/models so switching via LINE card is instant
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
      python -c "from huggingface_hub import hf_hub_download; from app.models_catalog import MODEL_CATALOG; [hf_hub_download(repo_id=m.gguf_repo, filename=m.gguf_filename, local_dir='/app/models') for m in MODEL_CATALOG.values()]"; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=5)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
