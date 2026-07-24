#!/usr/bin/env bash
# Start the ML Playground on http://127.0.0.1:8000
#
#   ./run.sh              # normal start
#   ./run.sh --dev        # auto-reload on source changes
#   PORT=9000 ./run.sh    # different port

set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c "import fastapi, sklearn" >/dev/null 2>&1; then
  echo "Missing dependencies. Install them with:"
  echo "    $PYTHON -m pip install -r requirements.txt"
  exit 1
fi

RELOAD=()
if [[ "${1:-}" == "--dev" ]]; then
  RELOAD=(--reload --reload-dir backend --reload-dir frontend)
fi

echo "ML Playground → http://${HOST}:${PORT}"
exec "$PYTHON" -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT" "${RELOAD[@]}"
