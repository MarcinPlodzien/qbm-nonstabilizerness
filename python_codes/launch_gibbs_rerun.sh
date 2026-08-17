#!/usr/bin/env bash
# =============================================================================
# launch_gibbs_rerun.sh -- self-contained-schema rerun of the SK-Gibbs data.
# 12 single-threaded workers; each worker runs N=8 (L=8) first (fast), then
# N=10 (L=12). Non-uniform beta grid: fine (0.2) near the magic peak, coarse
# (0.4) in the tail. Produces the new fully self-contained .npz files
# (loss trajectory + entangler gates + all params + target + init).
# =============================================================================
set -u
cd "$(dirname "$0")"

# single-thread each worker so 12 workers ~= 12 cores (no oversubscription)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"

# beta grid: 0.0..4.0 step 0.2, then 4.4..8.0 step 0.4  (31 points)
BETA_LIST=$(python3 -c "
import numpy as np
b=[round(x,2) for x in np.arange(0,4.001,0.2)]+[round(x,2) for x in np.arange(4.4,8.001,0.4)]
print(','.join(str(x) for x in b))")
echo "BETA_LIST=$BETA_LIST"

NW=12
COMMON="SG_MODEL=SK DISORDER_J_MEAN=0.0 DISORDER_J_STD=1.0 N_DIS=10 N_REAL=10 ANSATZ=main"
LOGDIR=/tmp/gibbs_rerun; mkdir -p "$LOGDIR"

for wid in $(seq 0 $((NW-1))); do
  ( env $COMMON GRID_N=8  GIBBS_L=8  BETA_LIST="$BETA_LIST" N_WORKERS=$NW WORKER_ID=$wid \
        python3 run_gibbs.py > "$LOGDIR/n8_w${wid}.log" 2>&1
    env $COMMON GRID_N=10 GIBBS_L=12 BETA_LIST="$BETA_LIST" N_WORKERS=$NW WORKER_ID=$wid \
        python3 run_gibbs.py > "$LOGDIR/n10_w${wid}.log" 2>&1
  ) &
done
echo "launched $NW worker chains (N=8 then N=10); logs in $LOGDIR"
wait
echo "ALL GIBBS RERUN WORKERS DONE"
