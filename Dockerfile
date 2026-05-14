FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    YOLO_CONFIG_DIR=/app/data/.ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

EXPOSE 9107

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:9107", "api:app"]
