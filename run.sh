#!/usr/bin/env bash
# Start the ML Playground on http://127.0.0.1:8000
#
#   ./run.sh              # normal start
#   ./run.sh --dev        # auto-reload on source changes
#   PORT=9000 ./run.sh    # different port

set -euo pipefail
cd "$(dirname "$0")"

# Fail legibly if the script has been moved out of the project root; otherwise
# uvicorn reports this as a bare ModuleNotFoundError several frames deep.
if [[ ! -f backend/app/main.py ]]; then
  echo "run.sh must sit in the project root; backend/app/main.py was not found next to it." >&2
  exit 1
fi

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

# Prefer the project venv so the script works whether or not the shell activated
# it. An explicit PYTHON=... still wins.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
  else
    PYTHON=python3
  fi
fi

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
