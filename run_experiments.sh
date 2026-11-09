#!/usr/bin/env bash
set -u
GPUS=${1:-0,1,2,3}
HISTORY=${HISTORY:-outputs/history.npz}
FACTOR=${FACTOR:-outputs/factor.npz}
RESULTS=${RESULTS:-results}
LOGS=${LOGS:-logs}
PATH_CANDS=${PATH_CANDS-0 6 10}
PATH_TBAR=${PATH_TBAR:-0.2}
ONLY=${ONLY:-}
COMMON="--history $HISTORY --factor $FACTOR --trials 200"
W="--widths 0.1 0.05 0.02"
PATHS="--candidates $PATH_CANDS --save_traj 200"
JOBS=(
  "ch3|ch3_fixed_budget.py $COMMON --budgets 0.025 0.05 0.1 0.2"
  "ch4|ch4_anytime_valid.py $COMMON $W"
  "ch4_uniform_eta0.2|ch4_anytime_valid.py $COMMON $W --reliance 0 --weights 1 --tau 1 --eta 0.2"
  "ch4_uniform_eta0.1|ch4_anytime_valid.py $COMMON $W --reliance 0 --weights 1 --tau 1 --eta 0.1"
  "ch4_uniform_eta0.05|ch4_anytime_valid.py $COMMON $W --reliance 0 --weights 1 --tau 1 --eta 0.05"
  "ch4_rho1|ch4_anytime_valid.py $COMMON $W --reliance 1 --weights 1"
  "ch4_rho0|ch4_anytime_valid.py $COMMON $W --reliance 0 --weights 1"
  "ch4_S3|ch4_anytime_valid.py $COMMON $W --reliance 0 0.5 1"
  "ch4_eta0.3|ch4_anytime_valid.py $COMMON $W --eta 0.3"
  "ch4_eta0.2|ch4_anytime_valid.py $COMMON $W --eta 0.2"
  "ch4_etamix|ch4_anytime_valid.py $COMMON $W --eta 0.2 0.5 0.8"
  "ch3_uniform|ch3_fixed_budget.py $COMMON --budgets 0.025 0.05 0.1 0.2 --no_prediction"
)
[ -n "$PATH_CANDS" ] && JOBS+=(
  "ch3_paths|ch3_fixed_budget.py $COMMON --budgets $PATH_TBAR $PATHS"
  "ch3_uniform_paths|ch3_fixed_budget.py $COMMON --budgets $PATH_TBAR $PATHS --no_prediction"
  "ch4_paths|ch4_anytime_valid.py $COMMON $W $PATHS --run_until $PATH_TBAR"
  "ch4_paths_S1|ch4_anytime_valid.py $COMMON $W $PATHS --reliance 1 --weights 1 --run_until $PATH_TBAR"
)

IFS=, read -ra G <<< "$GPUS"
mkdir -p "$LOGS" "$RESULTS"
for i in "${!G[@]}"; do
  (
    for ((j = i; j < ${#JOBS[@]}; j += ${#G[@]})); do
      name=${JOBS[j]%%|*}
      cmd=${JOBS[j]#*|}
      [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && continue
      if [ -f "$RESULTS/$name.csv" ]; then echo "skip $name (done)"; continue; fi
      echo "$(date +%T) GPU ${G[i]}: start $name"
      if CUDA_VISIBLE_DEVICES=${G[i]} python $cmd --out "$RESULTS/$name.csv" > "$LOGS/$name.log" 2>&1; then
        echo "$(date +%T) GPU ${G[i]}: done  $name"
      else
        echo "$(date +%T) GPU ${G[i]}: FAILED $name (see $LOGS/$name.log)"
      fi
    done
  ) &
done
wait
echo "all jobs finished"
