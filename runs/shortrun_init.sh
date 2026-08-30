#!/bin/bash

export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/remote/.nanochat-cache}
mkdir -p $NANOCHAT_BASE_DIR

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv

uv sync --frozen --extra gpu
source .venv/bin/activate

python -m nanochat.dataset -n 8
python -m scripts.tok_train
python -m scripts.tok_eval
