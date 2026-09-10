"""
fig_twod_openent.py -- Appendix figure: what separates the kicked-Ising entangler from the rest.

(a) Operator entanglement of the fixed brickwork U_S across every bipartition (q < j | q >= j),
    i.e. the entanglement entropy of the Choi state of U_S, computed from the dense 2^N x 2^N
    unitary rebuilt from the stored gates. This is a property of the entangler alone: it bounds,
    through the operator Schmidt rank, how much entanglement one layer can add or remove across
    that cut. The cluster values coincide with the maximum, 2 min(j, N-j) ln 2.
(b) Entropy across the x|y register cut (qubits 0-4 | 5-9, the two coordinate registers of the
    2D targets) after each of the L = 12 layers of the trained circuit, replayed from the stored
    angles and gates with born_machine.layer_states. Mean over the six targets and the three
    independent chains of the lowest-loss realization. Dashed line, Page value.

USAGE:  python3 fig_twod_openent.py   OUTPUT: figures/fig_twod_openent.{pdf,png}
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from utils.store import runs_dir
from utils.born_machine import layer_states
from utils.sre import page_entropy
import figstyle as fs

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
TARGETS = ["mm2d", "rings", "moons", "stripes", "cross", "spiral"]
ENT = ["ising", "cluster", "haar", "matchgate"]
NQ, KDEPTH, LFIN, CHAINS = 10, 20, 12, (1, 2, 3)


def path(c, L, e, t):
    return os.path.join(runs_dir(), f"twodch{c}__N{NQ}__K{KDEPTH}__L{L}__{e}__nt0__{t}__modesna__signa.npz")


def dense_scrambler(gates, pairs):
    """Rebuild the 2^N x 2^N brickwork unitary by pushing the identity through it."""
    D = 2 ** NQ
    A = np.eye(D, dtype=complex).reshape((2,) * NQ + (D,))
    for i, (a, b) in enumerate(pairs):
        a, b = int(a), int(b)
        g = np.asarray(gates[i]).reshape(2, 2, 2, 2)
        A = np.moveaxis(A, [a, b], [0, 1])
        sh = A.shape
        A = np.tensordot(g, A.reshape(2, 2, -1), axes=([2, 3], [0, 1])).reshape(sh)
        A = np.moveaxis(A, [0, 1], [a, b])
    return A.reshape(D, D)


def operator_entanglement(U, j):
    """Entropy of the Choi state of U across the cut (q < j | q >= j), in nats."""
    dA, dB = 2 ** j, 2 ** (NQ - j)
    T = U.reshape(dA, dB, dA, dB).transpose(0, 2, 1, 3).reshape(dA * dA, dB * dB) / np.sqrt(2 ** NQ)
    M = T @ T.conj().T if T.shape[0] <= T.shape[1] else T.conj().T @ T
    p = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    p = p / p.sum()
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


def cut_entropy(psi, j):
    s = np.linalg.svd(np.asarray(psi).reshape(2 ** j, -1), compute_uv=False)
    p = np.clip(s ** 2, 1e-300, None)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def main():
    page = page_entropy(NQ)
    cuts = np.arange(1, NQ)

    op = {}
    for e in ENT:
        z = np.load(path(1, LFIN, e, "moons"), allow_pickle=True)
        U = dense_scrambler(np.asarray(z["scrambler_gates"])[0], np.asarray(z["pairs"]))
        op[e] = np.array([operator_entanglement(U, j) for j in cuts])

    trace = {}
    for e in ENT:
        acc = []
        for t in TARGETS:
            for c in CHAINS:
                z = np.load(path(c, LFIN, e, t), allow_pickle=True)
                b = int(np.argmin(np.asarray(z["kld"])))
                gates = np.asarray(z["scrambler_gates"])[b]
                acc.append([cut_entropy(psi, NQ // 2) for psi in
                            layer_states(np.asarray(z["params"])[b], gates,
                                         np.asarray(z["pairs"]), NQ, LFIN)])
        trace[e] = np.mean(np.array(acc), axis=0)

    fs.setup()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=fs.STACK2,
                                   gridspec_kw={"hspace": 0.30})
    for e in ENT:
        ax0.plot(cuts, op[e], "-", marker=fs.MARKER[e], color=fs.COLOR[e], label=fs.LABEL[e])
    ax0.set_xlabel(r"cut position $j$")
    ax0.set_ylabel(r"$E_{\mathrm{op}}(\hat U_S)$")
    ax0.set_xticks(cuts)
    ax0.set_ylim(0.0, 10.2)
    ax0.legend(loc="upper left", ncol=2, fontsize=fs.FS_ANNOT, handlelength=1.6, columnspacing=1.0, frameon=False)
    fs.panel_tag(ax0, "(a)", x=0.035, y=0.10)

    ax1.axhline(page, ls="--", color=fs.GREY, lw=1.0)
    for e in ENT:
        ax1.plot(np.arange(LFIN + 1), trace[e], "-", marker=fs.MARKER[e],
                 color=fs.COLOR[e], label=fs.LABEL[e])
    ax1.set_xlabel(r"layer $\ell$")
    ax1.set_ylabel(r"$S_{x|y}$ ")
    ax1.set_xticks(range(0, LFIN + 1, 2))
    fs.panel_tag(ax1, "(b)", x=0.93, y=0.14)
    fs.save(fig, os.path.join(FIGS, "fig_twod_openent"))

    print("(a) operator entanglement across each cut")
    for e in ENT:
        print(f"    {e:10s} " + " ".join(f"{v:.3f}" for v in op[e]))
    print(f"(b) Page = {page:.3f};  register-cut entropy after layer l, mean over 6 targets x 3 chains")
    for e in ENT:
        print(f"    {e:10s} " + " ".join(f"{v:.2f}" for v in trace[e]))


if __name__ == "__main__":
    main()
