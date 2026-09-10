"""
compute_gibbs_randphase.py -- SRE of random-phase encodings for the SK Gibbs sweep
-------------------------------------------------------------------------------
For every (beta, disorder realization) of the main N=10 SK sweep with the CLIFFORD
entangler, take the lowest-loss run, rebuild its output distribution q_theta, and
evaluate the SRE of (i) the trained state, (ii) sqrt(q_theta) with i.i.d. uniform
random phases, (iii) sqrt(p_beta) with i.i.d. uniform random phases. Averages over
N_DRAWS phase draws. Output: data/runs/gibbs_randphase__N10__SK.npz, read by
fig_gibbs_kld.py to overlay the random-phase references on panel (b).

Run:  JAX_PLATFORMS=cpu python3 compute_gibbs_randphase.py
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from utils.store import runs_dir
from utils.sre import stabilizer_renyi_entropy
from utils.born_machine import make_forward

N, MODEL, L_MAIN, N_DRAWS = 10, "SK", 12, 3
R = runs_dir()
OUT = os.path.join(R, f"gibbs_randphase__N{N}__{MODEL}.npz")

recs = {}
for p in glob.glob(os.path.join(R, f"gibbs__N{N}__{MODEL}__*main__L{L_MAIN}__clifford__*.npz")):
    z = np.load(p, allow_pickle=True); m = json.loads(str(z["meta"]))
    recs.setdefault(float(m["beta"]), []).append((m, z))
betas = sorted(recs)
rng = np.random.default_rng(2026)
fwd = None
out = {k: [] for k in ("beta", "realization", "kld_best", "m2_trained", "m2_randphase_q", "m2_randphase_p", "m2min")}
for b in betas:
    for m, z in sorted(recs[b], key=lambda t: t[0]["realization_idx"]):
        pairs = tuple(map(tuple, np.asarray(z["pairs"]).tolist()))
        if fwd is None:
            fwd = make_forward(N, m["L"], pairs)
        best = int(z["best_idx"])
        probs, psi = fwd(jnp.asarray(z["params"][best]), jnp.asarray(z["entangler_gates"][best]))
        q = np.array(probs, dtype=float); q /= q.sum()
        p = np.asarray(z["target_p"], dtype=float)
        rq = [float(stabilizer_renyi_entropy(jnp.asarray(np.sqrt(q) * np.exp(1j * rng.uniform(0, 2 * np.pi, q.size))), N)) for _ in range(N_DRAWS)]
        rp = [float(stabilizer_renyi_entropy(jnp.asarray(np.sqrt(p) * np.exp(1j * rng.uniform(0, 2 * np.pi, p.size))), N)) for _ in range(N_DRAWS)]
        out["beta"].append(b); out["realization"].append(m["realization_idx"])
        out["kld_best"].append(float(z["kld"][best]))
        out["m2_trained"].append(float(stabilizer_renyi_entropy(psi, N)))
        out["m2_randphase_q"].append(np.mean(rq)); out["m2_randphase_p"].append(np.mean(rp))
        out["m2min"].append(float(z["m2min_target"]))
    i = np.array(out["beta"]) == b
    print(f"beta={b:.2f} n={i.sum()} trained={np.mean(np.array(out['m2_trained'])[i]):.2f} "
          f"rand(q)={np.mean(np.array(out['m2_randphase_q'])[i]):.2f} rand(p)={np.mean(np.array(out['m2_randphase_p'])[i]):.2f}", flush=True)
np.savez(OUT, **{k: np.asarray(v) for k, v in out.items()}, n_draws=N_DRAWS)
print("WROTE", OUT, flush=True)
