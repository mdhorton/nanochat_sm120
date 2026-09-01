#!/bin/bash

# nsys profile of a few steady-state training steps, one .nsys-rep per rank.
# Usage: bash runs/nsys_profile.sh [--all-ranks] [base_train overrides...]

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

# How many ranks to trace. Rank 0 only by default; --all-ranks traces every rank.
RANKS=1
LAUNCHER_NOTE=""
if [ "$1" = "--all-ranks" ]; then
    RANKS=$NPROC
    # Concurrent nsys instances fault in _StaticCudaLauncher's driver-API kernel launches.
    export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
    LAUNCHER_NOTE=" (static cuda launcher disabled)"
    shift
fi

OUTDIR=${NANOCHAT_BASE_DIR}/nsys/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUTDIR"

PROFILE_START=15
PROFILE_STEPS=3

FLAGS=(
    --depth $DEPTH
    --device-batch-size $DBS
    --num-iterations $((PROFILE_START + PROFILE_STEPS + 2))
    --eval-every -1
    --core-metric-every -1
    --sample-every -1
    --model-tag shortrun_nsys
    --profile-start $PROFILE_START
    --profile-steps $PROFILE_STEPS
)

case "$SM_ARCH" in
    9.0)  FLAGS+=(--fp8) ;;
    10.0) FLAGS+=(--fp8) ;;
    # Opt in to the windowed-flash fast path so a profile shows the kernels sm120 actually runs.
    12.0) export NANOCHAT_FA2_SWINDOW=1 ;;
esac

echo "$NPROC GPU(s), arch $SM_ARCH, ${VRAM_GB}GB -> d$DEPTH dbs$DBS, tracing $RANKS rank(s)$LAUNCHER_NOTE -> $OUTDIR"

source .venv/bin/activate
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
