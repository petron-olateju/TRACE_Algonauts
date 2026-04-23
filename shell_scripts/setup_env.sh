#!/bin/bash
set -euo pipefail

VENV_NAME="${1:-.venv}"

echo ">>> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo ">>> Syncing environment from pyproject.toml..."
uv sync

echo ">>> Done! To activate run:"
echo "  source ${VENV_NAME}/bin/activate"