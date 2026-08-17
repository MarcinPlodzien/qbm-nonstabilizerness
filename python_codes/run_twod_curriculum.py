"""Depth-curriculum warm-start driver for the 2D targets (fixes the L>=6 trainability barrier).

For each (target, entangler) chain we train L = 1,2,4,6,8,10,12 in order; each depth is
warm-started from the previous depth's best rotation angles (first L_prev layers = converged
angles, added layers = small random), which unlocks the deeper circuit's expressivity that a
random init cannot reach (cold L=6 KLD ~0.54 -> warm ~0.01). Entangler stays fixed;
only the rotations (which carry all the nonstabilizerness) are warm-started.

Writes the same self-describing npz schema as run_sweep (sweep='twod'), so the figure scripts
read it unchanged. Chains are independent -> sharded across processes via WORKER_ID/N_WORKERS.

Run:  JAX_PLATFORMS=cpu N_WORKERS=12 WORKER_ID=w python3 run_twod_curriculum.py
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import run_sweep as R
from utils.store import runs_dir, run_filename

TARGETS = os.environ.get("CURRIC_TARGETS", "mm2d,rings,moons,stripes").split(",")   # spiral dropped; stripes cleaner
ENTANGLERS = ["ising", "cluster", "haar", "matchgate"]
L_CURRIC = [1, 2, 4, 6, 8, 10, 12]     # deeper L hits the trainability barrier; not needed
N_COLD = 3                     # a few fully-random seeds each depth as insurance vs the warm ones
NQ = int(os.environ.get("GRID_N", "10"))   # register size (even); N=12 -> 64x64 2D grid
KDEPTH = 2 * NQ


def _meta_for(target, e, L):
    return dict(sweep="twod", N=NQ, K=KDEPTH, L=L, entangler=e, n_t=0, target=target,
                n_peaks="na", sigma_frac="na", n_t_target=0)


def read_best_angles(target, e, L):
    """Best-seed trained rotation angles (L,3,N) from the saved npz, or None if absent."""
    path = os.path.join(runs_dir(), run_filename(_meta_for(target, e, L)))
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    b = int(z["best_real"]) if "best_real" in z.files else int(np.argmin(z["kld"]))
    return np.asarray(z["params"])[b]                       # (L,3,N)


def warm_init(prev_best, Lp, L, nreal, seed):
    """(nreal,L,3,N): warm seeds = prev_best in first Lp layers + small random tail;
    plus N_COLD fully-random seeds. Warm seeds share the prefix but differ in the tail."""
    N = prev_best.shape[2]
    rng = np.random.default_rng(seed)
    out = []
    for r in range(nreal):
        if r < nreal - N_COLD:                              # warm
            tail = 0.05 * rng.standard_normal((L - Lp, 3, N))
            out.append(np.concatenate([prev_best, tail], axis=0))
        else:                                               # cold insurance
            out.append(rng.uniform(0, 2 * np.pi, (L, 3, N)))
    return jnp.asarray(np.stack(out))


def run_chain(target, e, group):
    prev_best, prev_L = None, None
    for L in L_CURRIC:
        if L * 3 * NQ >= 2 ** NQ:                            # keep under-param (3NL < 2^N)
            break
        cfg = dict(sweep="twod", N=NQ, K=KDEPTH, L=L, entangler=e, n_t=0, target=target)
        # ENTANGLERS.index keeps the warm-start seed reproducible (built-in hash() is salted per process)
        init = None if prev_best is None else warm_init(prev_best, prev_L, L, R.N_REAL, 7 * L + 97 * ENTANGLERS.index(e))
        R.run_one(cfg, group, init_override=init)
        b = read_best_angles(target, e, L)
        if b is not None:
            prev_best, prev_L = b, L


def main():
    group = R.load_clifford_group(verbose=False)
    chains = [(t, e) for t in TARGETS for e in ENTANGLERS]
    if os.environ.get("N_WORKERS"):
        wid, nw = int(os.environ.get("WORKER_ID", "0")), int(os.environ["N_WORKERS"])
        mine = [c for i, c in enumerate(chains) if i % nw == wid]
        print(f"[curric worker {wid}/{nw}] {len(mine)} chains", flush=True)
    else:
        mine = chains
    for t, e in mine:
        print(f"  chain target={t} entangler={e}", flush=True)
        run_chain(t, e, group)
    print("CURRIC DONE", flush=True)


if __name__ == "__main__":
    main()
