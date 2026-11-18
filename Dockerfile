FROM python:3.12-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG QWEN_MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct
ARG PRELOAD_MODEL=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    QWEN_MODEL_NAME=${QWEN_MODEL_NAME} \
    QWEN_PRELOAD_IN_BACKGROUND=true \
    PORT=8000

WORKDIR /app

# Create non-root user (UID 1000 works smoothly on Hugging Face Spaces, Railway, Render, and VPS)
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app

# Install lightweight CPU-only PyTorch first (keeps image small), then remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser app ./app

USER appuser

# Optionally bake the default Qwen weights into the image for fast startup (~2s)
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
      python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m='${QWEN_MODEL_NAME}'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)"; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=5)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
