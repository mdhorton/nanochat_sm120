#!/bin/bash

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/remote/.nanochat-cache}

source .venv/bin/activate

FLAGS=(
    --depth 12
    --device-batch-size 8
    --total-batch-size 524288
    --eval-every 50
    --eval-tokens 2097152
    --num-iterations 100
    --core-metric-every -1
    --sample-every -1
#    --window-pattern L
    --fp8
)

torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- "${FLAGS[@]}" "$@"
