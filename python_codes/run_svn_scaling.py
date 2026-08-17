"""Half-chain entanglement scaling vs N for a single application of the fixed entangler, U_S|+>.
Clifford/doped/Haar reach the Page coefficient (volume law); the free-fermion matchgate saturates well
below it. This is a property of the fixed unitary and involves no training. Writes svn_scaling.npz, read by
fig_svn_scaling.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax
from utils.born_machine import brickwork_pairs, plus_state, apply_scrambler
from utils.sre import half_chain_entropy, page_entropy
from utils.cliffords import load_clifford_group, sample_scrambler_gates, doped_clifford_2q_gates
from utils.store import runs_dir

LAD = ["matchgate", "clifford", "doped", "haar"]
NS = [4, 6, 8, 10, 12, 14, 16, 18, 20]
def nseeds(N): return 6 if N <= 12 else 4 if N <= 16 else 2
group = load_clifford_group(verbose=False)

single = {e: [] for e in LAD}
page = []
for N in NS:
    page.append(page_entropy(N))
    K = 2 * N; pairs = brickwork_pairs(N, K); ng = len(pairs)
    for e in LAD:
        vals = []
        for s in range(nseeds(N)):
            k = jax.random.PRNGKey(7000 + 13 * N + s)
            g = (doped_clifford_2q_gates(k, ng, group, 4) if e == "doped"
                 else sample_scrambler_gates(k, e, ng, group, 0))
            vals.append(float(half_chain_entropy(apply_scrambler(plus_state(N), g, pairs, N), N)))
        single[e].append(np.mean(vals))
    print(f"  N={N}: " + " ".join(f"{e}={single[e][-1]:.3f}" for e in LAD) + f"  Page={page[-1]:.3f}", flush=True)

out = os.path.join(runs_dir(), "svn_scaling.npz")
kw = dict(NS=np.array(NS, dtype=float), page=np.array(page, dtype=float))
for e in LAD:
    kw[f"single_{e}"] = np.array(single[e], dtype=float)     # markers: one U_S on |+>
np.savez(out, **kw)
print("saved svn-scaling data ->", out, " NS =", NS)
