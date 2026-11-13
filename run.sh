#!/usr/bin/env bash
# Start FaceLab locally.
#
#   ./run.sh                 # http://127.0.0.1:8010
#   ./run.sh --port 9000
set -euo pipefail
cd "$(dirname "$0")"

# Prefer the project's own virtualenv, fall back to `uv run`, then plain python.
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python server.py "$@"
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run python server.py "$@"
fi
exec python3 server.py "$@"
