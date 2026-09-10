"""
run_twod_chains.py -- independent depth-curriculum chains for the 2D targets, all entanglers.

The main-text 2D results come from ONE warm-start lineage per (target, entangler): every depth is
seeded from the best realization of the previous depth, so realizations within a chain are not
independent and a single chain cannot separate entangler effects from lineage luck. This driver
runs N_CHAINS additional, fully independent chains per (target, entangler): each chain has its own
base seed (entangler instances, cold initializations, warm-start tails), the same curriculum
L = 1,2,4,6,8,10,12, N_REAL realizations per depth (N_COLD of them cold), and N_EPOCHS epochs.
Files are written with sweep name 'twodch<c>' so nothing collides with the main 'twod' records.

Run:  JAX_PLATFORMS=cpu N_REAL=8 N_EPOCHS=1000 N_CHAINS=3 N_WORKERS=14 WORKER_ID=w python3 run_twod_chains.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import run_sweep as R
from utils.store import runs_dir, run_filename

TARGETS    = ["mm2d", "rings", "moons", "stripes", "cross", "spiral"]
ENTANGLERS = ["ising", "cluster", "haar", "matchgate"]
L_CURRIC   = [1, 2, 4, 6, 8, 10, 12]
N_CHAINS   = int(os.environ.get("N_CHAINS", "3"))
N_COLD     = int(os.environ.get("N_COLD", "2"))
NQ, KDEPTH = 10, 20
BASE0      = 12000                      # chain 0 = main-text lineage (sweep 'twod'); chains start at 1


def _meta_for(sweep, target, e, L):
    return dict(sweep=sweep, N=NQ, K=KDEPTH, L=L, entangler=e, n_t=0, target=target,
                n_peaks="na", sigma_frac="na", n_t_target=0)


def read_best_angles(sweep, target, e, L):
    path = os.path.join(runs_dir(), run_filename(_meta_for(sweep, target, e, L)))
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    b = int(z["best_real"]) if "best_real" in z.files else int(np.argmin(z["kld"]))
    return np.asarray(z["params"])[b]


def warm_init(prev_best, Lp, L, nreal, seed):
    N = prev_best.shape[2]
    rng = np.random.default_rng(seed)
    out = []
    for r in range(nreal):
        if r < nreal - N_COLD:
            tail = 0.05 * rng.standard_normal((L - Lp, 3, N))
            out.append(np.concatenate([prev_best, tail], axis=0))
        else:
            out.append(rng.uniform(0, 2 * np.pi, (L, 3, N)))
    return jnp.asarray(np.stack(out))


def run_chain(target, e, chain, group):
    sweep = f"twodch{chain}"
    R.BASE = BASE0 + 1000 * chain        # independent entangler instances and initializations
    prev_best, prev_L = None, None
    for L in L_CURRIC:
        cfg = dict(sweep=sweep, N=NQ, K=KDEPTH, L=L, entangler=e, n_t=0, target=target)
        init = None if prev_best is None else warm_init(
            prev_best, prev_L, L, R.N_REAL, R.BASE + 7 * L + 97 * ENTANGLERS.index(e) + 11 * TARGETS.index(target))
        R.run_one(cfg, group, init_override=init)
        b = read_best_angles(sweep, target, e, L)
        if b is not None:
            prev_best, prev_L = b, L


def main():
    group = R.load_clifford_group(verbose=False)
    chains = [(t, e, c) for c in range(1, N_CHAINS + 1) for t in TARGETS for e in ENTANGLERS]
    if os.environ.get("N_WORKERS"):
        wid, nw = int(os.environ.get("WORKER_ID", "0")), int(os.environ["N_WORKERS"])
        mine = [c for i, c in enumerate(chains) if i % nw == wid]
        print(f"[chains worker {wid}/{nw}] {len(mine)} chains", flush=True)
    else:
        mine = chains
    for t, e, c in mine:
        print(f"  chain target={t} entangler={e} chain={c}", flush=True)
        run_chain(t, e, c, group)
    print("CHAINS DONE", flush=True)


if __name__ == "__main__":
    main()
