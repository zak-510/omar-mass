#!/usr/bin/env bash
# Checkpointed bench: grow n 1..TARGET, cycling all 4 methods each step, then one
# batched grade pass per n. Resumable. Usage: run_round_robin.sh [N] [SEED] [flags]
set -euo pipefail

TARGET="${1:-100}"
SEED="${2:-0}"
shift "$(( $# >= 2 ? 2 : $# ))"
EXTRA=("$@")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${OMAR_DIR:-$HOME/.omar}/mass/runs"
mkdir -p "$OUTDIR"

# Prefer a prebuilt binary; fall back to cargo run.
RUN=(cargo run --release -q -p omar-mass --)
for cand in "$ROOT/target/release/omar-mass" "$ROOT/../target/release/omar-mass"; do
  if [[ -x "$cand" ]]; then
    RUN=("$cand")
    break
  fi
done

METHODS=(cot self-refine sc5 debate)

CSV=$(IFS=,; echo "${METHODS[*]}")

for ((n = 1; n <= TARGET; n++)); do
  for m in "${METHODS[@]}"; do
    echo "[round-robin] solve method=$m n=$n seed=$SEED"
    "${RUN[@]}" bench --method "$m" --n "$n" --seed "$SEED" --resume --no-grade \
      --out "$OUTDIR/$m.seed$SEED.json" ${EXTRA[@]+"${EXTRA[@]}"}
  done
  echo "[round-robin] grade n=$n seed=$SEED"
  "${RUN[@]}" grade --methods "$CSV" --seed "$SEED" --dir "$OUTDIR"
done

echo "[round-robin] done; scores in $OUTDIR/graded.seed$SEED.json"
