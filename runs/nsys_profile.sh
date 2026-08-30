#!/bin/bash

# nsys profile of a few steady-state training steps, one .nsys-rep per rank.
# Usage: bash runs/nsys_profile.sh [--all-ranks] [base_train overrides...]

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/remote/.nanochat-cache}

source .venv/bin/activate

NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d " ")

# How many ranks to trace. Rank 0 only by default; --all-ranks traces every rank,
# which is what shows load imbalance and collective wait attribution.
RANKS=1
if [ "$1" = "--all-ranks" ]; then RANKS=$NPROC; shift; fi

OUTDIR=${NANOCHAT_BASE_DIR}/nsys/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUTDIR"

PROFILE_START=15
PROFILE_STEPS=3

FLAGS=(
    --depth 24
    --device-batch-size 16
    --num-iterations $((PROFILE_START + PROFILE_STEPS + 2))
    --eval-every -1
    --core-metric-every -1
    --sample-every -1
    --model-tag shortrun_nsys
    --profile-start $PROFILE_START
    --profile-steps $PROFILE_STEPS
)

case "$ARCH" in
    9.0)  FLAGS+=(--fp8) ;;
    10.0) FLAGS+=(--fp8) ;;
    12.0) FLAGS+=(--fp8 --window-pattern L) ;;
esac

echo "$NPROC GPU(s), arch $ARCH, tracing $RANKS rank(s) -> $OUTDIR"

# torchrun --no-python so each rank execs its own nsys; RANK is set per child process.
torchrun --standalone --nproc_per_node=$NPROC --no-python \
    bash -c '
        if [ "${RANK:-0}" -lt "'"$RANKS"'" ]; then
            exec nsys profile \
                --output="'"$OUTDIR"'/rank${RANK}" \
                --force-overwrite=true \
                --trace=cuda,nvtx,cublas \
                --sample=none --cpuctxsw=none \
                --capture-range=cudaProfilerApi --capture-range-end=stop \
                python -m scripts.base_train "$@"
        else
            exec python -m scripts.base_train "$@"
        fi
    ' _ "${FLAGS[@]}" "$@"
