#!/usr/bin/env sh
# Container entrypoint: make sure the default model pack exists, then start the
# server. Fetching is skipped when the pack is already on disk; on a host with
# ephemeral storage (or a fresh container) it downloads again on each start.
set -eu

cd "$(dirname "$0")/.."

MODEL="${FACELAB_DEFAULT_MODEL:-buffalo_l}"
FETCH="${FACELAB_FETCH_MODELS:-1}"
STRICT="${FACELAB_FETCH_STRICT:-0}"
if [ "$FETCH" = "all" ]; then
  if ! python scripts/fetch_models.py --all; then
    if [ "$STRICT" = "1" ]; then
      echo "[facelab] error: could not fetch all models (FACELAB_FETCH_STRICT=1)" >&2
      exit 1
    fi
    echo "[facelab] warning: could not fetch all models; starting anyway" >&2
  fi
elif [ "$FETCH" = "1" ]; then
  if ! python scripts/fetch_models.py --model "$MODEL"; then
    if [ "$STRICT" = "1" ]; then
      echo "[facelab] error: could not fetch model '$MODEL' (FACELAB_FETCH_STRICT=1)" >&2
      exit 1
    fi
    echo "[facelab] warning: could not fetch model '$MODEL'; starting anyway" >&2
  fi
fi

exec python server.py --host "${FACELAB_HOST:-0.0.0.0}" --port "${PORT:-8010}"
