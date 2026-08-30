#!/bin/bash

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/remote/.nanochat-cache}

source .venv/bin/activate

NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d " ")

FLAGS=(
    --depth 24
    --device-batch-size 16
    --num-iterations 100
    --eval-every 50
    --eval-tokens 2097152
    --model-tag shortrun
    --core-metric-every -1
    --sample-every -1
)

case "$ARCH" in
    9.0)  FLAGS+=(--fp8) ;;
    12.0) FLAGS+=(--fp8 --window-pattern L) ;;
esac

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- "${FLAGS[@]}" "$@"
