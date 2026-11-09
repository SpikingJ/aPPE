#!/usr/bin/env bash
set -u
OUT=${OUT:-outputs_nostra}
RESULTS=${RESULTS:-results_nostra}
LOGS=${LOGS:-logs_nostra}
CANDS=${CANDS:-collab gb_d3 mlp_64x64 rf_leaf20 logreg_bal nb}
PATH_CAND=${PATH_CAND:-collab}
TRIALS=${TRIALS:-200}
ETA=${ETA:-0.9}
TBAR=${TBAR:-400}
WIDTHS=${WIDTHS:-0.2 0.15 0.1}
W="--widths $WIDTHS"
TMAX="--T_max 1.0"
mkdir -p "$OUT" "$RESULTS" "$LOGS"

python build_history_nostra.py --campaign campaign --models models/models.csv --out "$OUT" \
  --candidates $CANDS --leave_one_out > "$LOGS/build_history.log" 2>&1 || { cat "$LOGS/build_history.log"; exit 1; }
python build_history_nostra.py --campaign campaign --models models/models.csv --out "$OUT" \
  --candidates gb_d3 --leave_one_out --snapshot_candidates --suffix _checkpoints >> "$LOGS/build_history.log" 2>&1
cat "$LOGS/build_history.log"

run() {
  local dir=$1 name=$2; shift 2
  mkdir -p "$RESULTS/$dir" "$LOGS/$dir"
  if [ -f "$RESULTS/$dir/$name.csv" ]; then echo "skip $dir/$name"; return; fi
  echo "$(date +%T) $dir/$name"
  python "$@" --out "$RESULTS/$dir/$name.csv" > "$LOGS/$dir/$name.log" 2>&1 || echo "FAILED $dir/$name (see $LOGS/$dir/$name.log)"
}

for c in $CANDS gb_d3_checkpoints; do
  h="$OUT/history_$c.npz"
  f="$OUT/factor_$c.npz"
  [ -f "$f" ] || python factor_model.py --history "$h" --out "$f" --cv --d_grid 1 2 3 4 6 \
    --kappa_grid 0.1 1 10 > "$LOGS/factor_$c.log" 2>&1
  COMMON="--history $h --factor $f --trials $TRIALS"
  run "$c" ch4 ch4_anytime_valid.py $COMMON $W $TMAX --eta $ETA
  run "$c" ch4_rho1 ch4_anytime_valid.py $COMMON $W $TMAX --eta $ETA --reliance 1 --weights 1
  run "$c" ch4_eta0.5 ch4_anytime_valid.py $COMMON $W $TMAX --eta 0.5
  run "$c" ch3 ch3_fixed_budget.py $COMMON --budgets $TBAR 1.0 --stop_widths $WIDTHS
done

c=$PATH_CAND; h="$OUT/history_$c.npz"; f="$OUT/factor_$c.npz"
COMMON="--history $h --factor $f --trials $TRIALS --save_traj $TRIALS"
run "$c" ch3_paths ch3_fixed_budget.py $COMMON --budgets $TBAR
run "$c" ch3_uniform_paths ch3_fixed_budget.py $COMMON --budgets $TBAR --no_prediction
run "$c" ch4_paths ch4_anytime_valid.py $COMMON $W $TMAX --run_until $TBAR --eta $ETA
run "$c" ch4_paths_S1 ch4_anytime_valid.py $COMMON $W $TMAX --run_until $TBAR --eta $ETA --reliance 1 --weights 1
echo "all jobs finished"
