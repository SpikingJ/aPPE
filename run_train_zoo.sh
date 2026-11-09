#!/usr/bin/env bash
set -e
SCRIPT=${SCRIPT:-train_zoo.py}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
DATA=${1:-data/radioml2018}
OUT=${2:-outputs/zoo}
GPUS=${3:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
mkdir -p logs "$OUT"
if ls "$OUT"/*.lock >/dev/null 2>&1; then
  echo "lock files found in $OUT (left by killed workers?) -- remove them if no worker is running:"
  ls "$OUT"/*.lock
  exit 1
fi
for g in ${GPUS//,/ }; do
  log="logs/$(basename "${SCRIPT%.py}")_gpu$g.log"
  CUDA_VISIBLE_DEVICES=$g nohup python "$SCRIPT" --data "$DATA" --out "$OUT" > "$log" 2>&1 &
  echo "GPU $g: pid $! -> $log"
  sleep 30
done
