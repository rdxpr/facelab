#!/usr/bin/env sh
# Container entrypoint: make sure the default model pack exists, then start the
# server. Fetching is skipped when the pack is already on disk; on an ephemeral
# host (e.g. Hugging Face Spaces) that download happens again on each cold start.
set -eu

cd "$(dirname "$0")/.."

MODEL="${FACELAB_DEFAULT_MODEL:-buffalo_l}"
if [ "${FACELAB_FETCH_MODELS:-1}" = "1" ]; then
  python scripts/fetch_models.py --model "$MODEL" \
    || echo "[facelab] warning: could not fetch model '$MODEL'; starting anyway"
fi

exec python server.py --host "${FACELAB_HOST:-0.0.0.0}" --port "${PORT:-8010}"
