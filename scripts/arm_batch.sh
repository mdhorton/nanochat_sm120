#!/usr/bin/bash
# Run a set of base_train arms back to back under one protocol: one process per arm, an optional
# cooldown to a common start temperature between arms, per-arm logs, and an append-only progress
# file. See dev/perf-log.md, "The batched-arm protocol" -- arms run this way reproduce to ~0.4%,
# against the ~2% bar that applies to runs compared across sessions.
#
# Throughput arms need --cooldown. Numerics-only arms do not: bpb does not depend on thermal
# state, so cooling between them buys nothing and costs the cap per arm.
set -u

usage() {
  cat <<'EOF'
usage: scripts/arm_batch.sh --out DIR [options] -- NAME "FLAGS" [NAME "FLAGS" ...]

  --out DIR             per-arm logs and progress.txt land here (required)
  --base "FLAGS"        flags every arm shares
  --nproc N             torchrun --nproc_per_node (default 2)
  --base-dir PATH       NANOCHAT_BASE_DIR for the children (default: inherited)
  --cooldown SECONDS    max wait for both dies to reach --cooldown-target before each arm
                        (default 0, i.e. no cooldown)
  --cooldown-target C   temperature to wait for, in C (default 40)

Each NAME gets $DIR/$NAME.log, carrying its start temperature, its flags and its exit code.
"done NAME" is appended to $DIR/progress.txt as each arm finishes, then "BATCH COMPLETE".
EOF
}

OUT="" BASE="" NPROC=2 BASE_DIR="" COOLDOWN=0 TARGET=40
while [ $# -gt 0 ]; do
  case "$1" in
    --out)             OUT=$2; shift 2;;
    --base)            BASE=$2; shift 2;;
    --nproc)           NPROC=$2; shift 2;;
    --base-dir)        BASE_DIR=$2; shift 2;;
    --cooldown)        COOLDOWN=$2; shift 2;;
    --cooldown-target) TARGET=$2; shift 2;;
    --) shift; break;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage; exit 1;;
  esac
done

[ -n "$OUT" ] || { echo "--out is required" >&2; exit 1; }
[ $# -ge 2 ] || { echo "need at least one NAME \"FLAGS\" pair after --" >&2; exit 1; }
[ $(($# % 2)) -eq 0 ] || { echo "arms must be NAME \"FLAGS\" pairs" >&2; exit 1; }

cd "$(dirname "$0")/.." || exit 1   # .venv/bin/torchrun and -m scripts.base_train are repo-relative
mkdir -p "$OUT"
: > "$OUT/progress.txt"

cooldown () {
  local t=0 max
  while [ "$t" -lt "$COOLDOWN" ]; do
    max=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -rn | head -1)
    [ "$max" -le "$TARGET" ] && return
    sleep 10; t=$((t + 10))
  done
}

while [ $# -gt 0 ]; do
  name=$1; flags=$2; shift 2
  [ "$COOLDOWN" -gt 0 ] && cooldown
  {
    echo "### arm=$name"
    echo "### start_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr '\n' ' ')"
    echo "### flags=$BASE $flags"
    date -Is
  } > "$OUT/$name.log"
  # Two things in the line below are load-bearing. `--standalone` picks the single-node
  # rendezvous, and the `--` stops torchrun claiming the script's flags as its own -- without it
  # it dies on "ambiguous option: --run could match --run-path". scripts/run.sh has both.
  # Do NOT put a comment between these continuation lines: a `#` after a backslash continuation
  # opens a comment that swallows the rest of the command, silently dropping the env prefix.
  env ${BASE_DIR:+NANOCHAT_BASE_DIR="$BASE_DIR"} OMP_NUM_THREADS=1 \
    .venv/bin/torchrun --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- $BASE $flags \
    >> "$OUT/$name.log" 2>&1
  echo "### exit=$? $(date -Is)" >> "$OUT/$name.log"
  echo "done $name" >> "$OUT/progress.txt"
done
echo "BATCH COMPLETE" >> "$OUT/progress.txt"
