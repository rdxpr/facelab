# FaceLab container image — for self-hosting (a VPS, or any Docker host).
#
# The image runs with FACELAB_ENV=prod, so uploaded photos and generated
# results go to an ephemeral directory (see FACELAB_EPHEMERAL_DIR) that is
# wiped on every start. Model weights are fetched to ./models on first boot;
# mount a volume at /app/models if you want to keep them between restarts.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    FACELAB_ENV=prod \
    FACELAB_HOST=0.0.0.0 \
    PORT=8010 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# opencv-python is a transitive dependency of insightface; it links libGL.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Use uv so [tool.uv] override-dependencies keeps the CPU onnxruntime wheel out
# of the environment that already has onnxruntime-gpu (plain pip ignores
# pyproject.toml, installing both). --no-dev keeps pytest out of the image.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

COPY . .
RUN useradd -r -u 10001 -d /tmp facelab && chown -R facelab:facelab /app
USER facelab

EXPOSE 8010

# Give the first boot time to fetch the model pack before declaring it unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

CMD ["sh", "/app/scripts/start.sh"]
