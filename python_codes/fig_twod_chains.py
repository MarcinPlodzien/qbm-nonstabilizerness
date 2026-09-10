"""
fig_twod_chains.py -- Appendix figure: KLD vs depth for four entanglers on the six 2D targets,
three independent depth-curriculum chains per (target, entangler). Reads the 'twodch<c>' records.

Each panel: lowest-loss KLD at every curriculum depth, mean over chains (line) and one standard
deviation over chains (band). Cold start at L=1; from L=2 on, six of eight realizations are warm-started.

USAGE:  python3 fig_twod_chains.py   OUTPUT: figures/fig_twod_chains.{pdf,png}
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.store import runs_dir
import figstyle as fs

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
TARGETS = [("mm2d", "four lobes"), ("rings", "two rings"), ("moons", "two moons"),
           ("stripes", "three stripes"), ("cross", "diagonal cross"), ("spiral", "two spirals")]
ENT = ["ising", "cluster", "haar", "matchgate"]


def load():
    d = {}
    for p in glob.glob(os.path.join(runs_dir(), "twodch*__*.npz")):
        z = np.load(p, allow_pickle=True); m = json.loads(str(z["meta"]))
        if m["N"] != 10:
            continue
        c = int(m["sweep"].replace("twodch", ""))
        d.setdefault((m["target"], m["entangler"]), {}).setdefault(m["L"], {})[c] = float(np.min(z["kld"]))
    return d


def main():
    d = load()
    fs.setup()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.22, "wspace": 0.10})
    for ax, (t, lab), tag in zip(axes.ravel(), TARGETS, "abcdef"):
        for e in ENT:
            byL = d.get((t, e), {})
            Ls = sorted(byL)
            mu = np.array([np.mean(list(byL[L].values())) for L in Ls])
            sd = np.array([np.std(list(byL[L].values()), ddof=1) if len(byL[L]) > 1 else 0.0 for L in Ls])
            ax.plot(Ls, mu, "-", marker=fs.MARKER[e], color=fs.COLOR[e], label=fs.LABEL[e], ms=3.2)
            lo = np.clip(mu - sd, mu * 0.2, None)
            ax.fill_between(Ls, lo, mu + sd, color=fs.COLOR[e], alpha=0.15, lw=0)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=2.0, subs=(1.0,), numticks=10))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
        ax.text(0.97, 0.95, lab, transform=ax.transAxes, ha="right", va="top", fontsize=fs.FS_ANNOT)
        fs.panel_tag(ax, f"({tag})", x=0.035, y=0.12)
    for ax in axes[1]:
        ax.set_xlabel(r"layers $L$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"KLD")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=fs.FS_ANNOT,
               handlelength=1.8, columnspacing=1.4, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(bottom=0.22)
    out = os.path.join(FIGS, "fig_twod_chains")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
