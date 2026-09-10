"""
get_data_table_1.py -- data for Manuscript Table I (source: run_twod_chains.py).
Three independent depth-curriculum chains per (2D target, entangler). Workers run in parallel;
about 5 h on 14 CPU workers. Set N_WORKERS to the number of cores to use.
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
nw = int(os.environ.get("N_WORKERS", "1"))
env = dict(os.environ, OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_cpu_multi_thread_eigen=false",
           SUMMARY_ON_SAVE="0", N_REAL="8", N_EPOCHS="1000", N_CHAINS="3", N_WORKERS=str(nw))
procs = [subprocess.Popen([sys.executable, os.path.join(HERE, "run_twod_chains.py")], env=dict(env, WORKER_ID=str(w)))
         for w in range(nw)]
for p in procs:
    p.wait()
print("table-1 data done")
