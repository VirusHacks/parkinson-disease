#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/parkinsons_Motor"
WEB_DIR="${PROJECT_DIR}/static/myosuite_demo"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "Project directory not found: ${PROJECT_DIR}" >&2
  exit 1
fi

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    echo "uv already installed: $(uv --version)"
    return
  fi

  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "uv installed but not found in PATH. Add ${HOME}/.local/bin to PATH and rerun." >&2
    exit 1
  fi
}

install_frontend_dependencies() {
  if [[ ! -d "${WEB_DIR}" ]]; then
    echo "Skipping frontend setup: ${WEB_DIR} not found."
    return
  fi

  if ! command -v npm >/dev/null 2>&1; then
    echo "npm not found. Skipping npm install (frontend assets may already be vendored)."
    return
  fi

  if [[ -f "${WEB_DIR}/package-lock.json" ]]; then
    echo "Installing frontend dependencies with npm ci..."
    (cd "${WEB_DIR}" && npm ci)
  elif [[ -f "${WEB_DIR}/package.json" ]]; then
    echo "Installing frontend dependencies with npm install..."
    (cd "${WEB_DIR}" && npm install)
  fi
}

ensure_uv

echo "Syncing Python dependencies with uv..."
uv sync --project "${PROJECT_DIR}"

install_frontend_dependencies

cat <<EOF

Setup complete.

Run the app with:
  uv run --project "${PROJECT_DIR}" server

Open in browser:
  http://localhost:8000/web
  http://localhost:8000/viewer
EOF
