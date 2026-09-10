"""
================================================================================
 verify_operator_entanglement.py -- standalone check of the Appendix B claim
================================================================================
Recomputes, from the gate definitions alone and with no stored data, the operator
entanglement of each fixed entangler U_S:

    U_S = sum_k s_k A_k (x) B_k     (operator Schmidt decomposition across a cut)

    E_op(j) = -sum_k p_k ln p_k,  p_k = s_k^2 / sum s^2,   r(j) = #{k : s_k > 0}

reported (i) across every bipartition j at the operating depth K = 2N, and
(ii) across the middle cut as a function of the brickwork depth K, which shows the
Floquet revival of the kicked-Ising circuit.

Expected (N = 10, K = 2N): kicked Ising E_op = ln4 = 1.386 and r = 4 at every cut;
cluster E_op = 5 ln4 = 6.931 and r = 1024 at the middle cut; local Haar 5.94;
matchgate 2.61. Clifford circuits give a flat Schmidt spectrum, so E_op = ln r there.

Run:  python3 verify_operator_entanglement.py            # N = 10 (a few minutes)
      GRID_N=8 python3 verify_operator_entanglement.py   # N = 8  (fast)
================================================================================
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
from utils.cliffords import _G_ISING, _G_CLUSTER, sample_scrambler_gates, load_clifford_group
from utils.born_machine import brickwork_pairs

N = int(os.environ.get("GRID_N", "10"))


def build(N, gates, pairs):
    """Dense U_S, applying each two-qubit gate to the row axes only."""
    d = 2 ** N
    U = np.eye(d, dtype=complex).reshape((2,) * N + (d,))
    for g, (i, j) in zip(gates, pairs):
        g4 = np.asarray(g).reshape(2, 2, 2, 2)
        U = np.moveaxis(U, (i, j), (0, 1))
        sh = U.shape
        U = np.tensordot(g4, U.reshape(2, 2, -1), axes=([2, 3], [0, 1])).reshape(sh)
        U = np.moveaxis(U, (0, 1), (i, j))
    return U.reshape(d, d)


def op_ent(U, N, cut):
    dA, dB = 2 ** cut, 2 ** (N - cut)
    M = U.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA * dA, dB * dB)
    s = np.linalg.svd(M, compute_uv=False)
    p = s ** 2
    p = p / p.sum()
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p))), int(np.sum(s > 1e-9))


def main():
    K = 2 * N
    pairs = brickwork_pairs(N, K)
    key = jax.random.PRNGKey(0)
    fams = {
        "kicked-Ising": np.broadcast_to(_G_ISING, (len(pairs), 4, 4)),
        "cluster":      np.broadcast_to(_G_CLUSTER, (len(pairs), 4, 4)),
        "local-Haar":   np.asarray(sample_scrambler_gates(key, "haar", len(pairs))),
        "matchgate":    np.asarray(sample_scrambler_gates(key, "matchgate", len(pairs))),
    }
    print(f"N = {N}, K = 2N = {K};  ln4 = {np.log(4):.4f},  N ln2 = {N*np.log(2):.4f}", flush=True)
    print("\nE_op across every cut j (and operator Schmidt rank):", flush=True)
    for name, g in fams.items():
        U = build(N, g, pairs)
        assert np.allclose(U.conj().T @ U, np.eye(2 ** N), atol=1e-9), f"{name} not unitary"
        res = [op_ent(U, N, j) for j in range(1, N)]
        print(f"  {name:13s} E_op " + " ".join(f"{e:6.3f}" for e, _ in res), flush=True)
        print(f"  {'':13s} rank " + " ".join(f"{r:6d}" for _, r in res), flush=True)

    print("\nkicked-Ising, middle cut, vs brickwork depth K (Floquet revival):", flush=True)
    for k in range(1, 3 * N + 1):
        e, r = op_ent(build(N, np.broadcast_to(_G_ISING, (len(brickwork_pairs(N, k)), 4, 4)),
                            brickwork_pairs(N, k)), N, N // 2)
        tag = "   <-- operating depth K = 2N" if k == K else ""
        print(f"  K = {k:2d}   E_op = {e:6.3f}   rank = {r:5d}{tag}", flush=True)


if __name__ == "__main__":
    main()
