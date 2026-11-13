# FaceLab container image — works on Render.com and anywhere Docker runs.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FACELAB_ENV=prod \
    FACELAB_HOST=0.0.0.0 \
    PORT=8010

WORKDIR /app

# opencv-python is a transitive dependency of insightface; it links libGL.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8010

# Give the first boot time to fetch the model pack before declaring it unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

CMD ["sh", "/app/scripts/start.sh"]
