FROM python:3.12-slim

ARG QWEN_GGUF_REPO=Qwen/Qwen2.5-0.5B-Instruct-GGUF
ARG QWEN_GGUF_FILENAME=qwen2.5-0.5b-instruct-q4_0.gguf
ARG PRELOAD_MODEL=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    QWEN_BACKEND=llamacpp \
    QWEN_GGUF_REPO=${QWEN_GGUF_REPO} \
    QWEN_GGUF_FILENAME=${QWEN_GGUF_FILENAME} \
    QWEN_GGUF_PATH=/app/models/${QWEN_GGUF_FILENAME} \
    QWEN_N_CTX=1024 \
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

# Install lightweight web stack + prebuilt CPU wheel for llama-cpp-python (~450 MB total RAM at runtime)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt \
    && pip install --no-cache-dir --only-binary=:all: \
       --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
       llama-cpp-python

COPY --chown=appuser:appuser app ./app

USER appuser

# Pre-download the quantized Qwen GGUF file into /app/models so boot is instant and offline
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
      python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${QWEN_GGUF_REPO}', filename='${QWEN_GGUF_FILENAME}', local_dir='/app/models')"; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=5)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
