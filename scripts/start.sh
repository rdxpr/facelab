#!/usr/bin/env sh
# Container entrypoint: make sure the default model pack exists, then start the
# server. Fetching is skipped when the pack is already on disk; on a host with
# ephemeral storage (or a fresh container) it downloads again on each start.
set -eu

cd "$(dirname "$0")/.."

MODEL="${FACELAB_DEFAULT_MODEL:-buffalo_l}"
FETCH="${FACELAB_FETCH_MODELS:-1}"
if [ "$FETCH" = "all" ]; then
  python scripts/fetch_models.py --all \
    || echo "[facelab] warning: could not fetch all models; starting anyway"
elif [ "$FETCH" = "1" ]; then
  python scripts/fetch_models.py --model "$MODEL" \
    || echo "[facelab] warning: could not fetch model '$MODEL'; starting anyway"
fi

exec python server.py --host "${FACELAB_HOST:-0.0.0.0}" --port "${PORT:-8010}"
