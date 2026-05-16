FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    DETECTOR_MODEL_PATH=/app/models/yolo11n.onnx \
    DETECTOR_MODEL_URL=https://huggingface.co/webnn/yolo11n/resolve/main/onnx/yolo11n.onnx \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN mkdir -p /app/models \
    && curl -L --fail "$DETECTOR_MODEL_URL" -o "$DETECTOR_MODEL_PATH"

COPY app ./app

RUN useradd --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app

USER app

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
