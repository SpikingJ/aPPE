#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
GPU=${1:-0}
ZOO=${ZOO:-zoo_sb}
TEST=${TEST:-deep_prior}
OUT=${OUT:-outputs_sb}
RESULTS=${RESULTS:-results_sb}
LOGS=${LOGS:-logs_sb}
TBAR=${TBAR:-4000}
mkdir -p "$OUT" "$RESULTS" "$LOGS"

python build_history.py --zoo "$ZOO" --out "$OUT/history.npz" --key slot_ok --holdout names --names $TEST \
  > "$LOGS/build_history.log" 2>&1 || { cat "$LOGS/build_history.log"; exit 1; }
cat "$LOGS/build_history.log"
[ -f "$OUT/factor.npz" ] || CUDA_VISIBLE_DEVICES=$GPU python factor_model.py --history "$OUT/history.npz" \
  --out "$OUT/factor.npz" --cv > "$LOGS/factor.log" 2>&1
grep -E "selected|candidate|$TEST" "$LOGS/factor.log"

HISTORY="$OUT/history.npz" FACTOR="$OUT/factor.npz" RESULTS="$RESULTS" LOGS="$LOGS" PATH_CANDS="$TEST" PATH_TBAR=$TBAR \
  ONLY="ch3 ch4 ch4_rho1 ch3_paths ch3_uniform_paths ch4_paths ch4_paths_S1" bash run_experiments.sh "$GPU"
