#!/bin/bash

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/remote/.nanochat-cache}

source .venv/bin/activate

FLAGS=(
    --depth 24
    --device-batch-size 16
    --num-iterations 100
    --eval-every -1
    --core-metric-every -1
    --sample-every -1
    --save-every -1
    --fp8
)

torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- "${FLAGS[@]}" "$@"
