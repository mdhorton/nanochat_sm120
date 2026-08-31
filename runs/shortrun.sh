#!/bin/bash

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/remote/.nanochat-cache}

NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
SM_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d " ")
VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%.0f", $1/1024}')

# Model size is picked from VRAM so the same script runs on big and small hosts.
if [ "${VRAM_GB:-0}" -ge 70 ]; then
    DEPTH=24
    DBS=16
else
    DEPTH=12
    DBS=8
fi

FLAGS=(
    --depth $DEPTH
    --device-batch-size $DBS
    --num-iterations 50
    --eval-every 25
    --eval-tokens 2097152
    --model-tag shortrun
    --core-metric-every -1
    --sample-every -1
)

case "$SM_ARCH" in
    9.0|10.0) FLAGS+=(--fp8) ;;
    12.0) FLAGS+=(--nvfp4) ;;&
    12.0) FLAGS+=() ;;
esac

echo "$NPROC GPU(s), arch $SM_ARCH, ${VRAM_GB}GB -> d$DEPTH dbs$DBS"
echo "${FLAGS[@]}"

source .venv/bin/activate
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- "${FLAGS[@]}" "$@"
