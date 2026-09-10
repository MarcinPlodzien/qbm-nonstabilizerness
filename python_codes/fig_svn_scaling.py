"""fig_svn_scaling.py -- half-chain von Neumann entropy vs qubit number N for a single application of the
entangler U_S|+>, all four families, against the Page value. Reads svn_scaling.npz written by
run_svn_scaling.py. Output: figures/fig_svn_scaling.{pdf,png}."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from utils.store import runs_dir
import figstyle as fs

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
DATA = os.path.join(runs_dir(), "svn_scaling.npz")
LAD  = ["matchgate", "clifford", "doped", "haar"]


def main():
    if not os.path.exists(DATA):
        print("missing", DATA, "-- run run_svn_scaling.py first"); return
    d = np.load(DATA, allow_pickle=True)
    NS = d["NS"]; page = d["page"]
    print(f"svn-scaling: N={list(NS.astype(int))}", flush=True)

    fs.setup()
    fig, ax = plt.subplots(figsize=fs.SINGLE)
    ax.plot(NS, page, "--", color=fs.GREY, lw=1.2, label="Page value")
    for e in LAD:
        ax.plot(NS, d[f"single_{e}"], marker=fs.MARKER[e], color=fs.COLOR[e], ls="-", label=fs.LABEL[e])

    ax.set_xlabel(r"qubits $N$")
    ax.set_ylabel(r"half-chain $S_{vN}$")
    ax.set_xticks(NS[::2].astype(int) if len(NS) > 8 else NS.astype(int))
    ax.set_ylim(top=ax.get_ylim()[1] * 1.42)
    ax.legend(loc="upper left", ncol=3, fontsize=fs.FS_ANNOT, frameon=False,
              handlelength=1.5, columnspacing=1.0, handletextpad=0.5)
    fs.save(fig, os.path.join(FIGS, "fig_svn_scaling"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
