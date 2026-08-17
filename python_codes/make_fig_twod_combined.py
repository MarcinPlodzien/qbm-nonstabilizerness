r"""make_fig_twod_combined.py -- two-dimensional target figures.

make_fig_twod_combined()   : full-width horizontal 2 x 6 grid, top row targets (a)-(f),
                             bottom row trained-Born output (g)-(l) (32x32, magma, best seed, L=12).
                             Saves figures/fig_twod_combined.{pdf,png}, width=\textwidth.
make_fig_twod_convergence(): single-column KLD-vs-epoch panel, one line per target (log2 y-axis).
                             Saves figures/fig_twod_convergence.{pdf,png}.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.ticker as mticker

from pra_style import apply, grid, save, FONT_SIZE_PANEL_LABEL, FONT_BOOST

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = os.path.join(ROOT, "data", "runs")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

# Target order and labels (must match the npz meta "target")
TGT2D = [
    ("mm2d",    "four lobes"),
    ("rings",   "two rings"),
    ("moons",   "two moons"),
    ("stripes", "three stripes"),
    ("cross",   "diagonal cross"),
    ("spiral",  "two spirals"),
]

# Per-target colours
COL2D = {
    "mm2d":    "#E69F00",
    "rings":   "#56B4E9",
    "moons":   "#009E73",
    "stripes": "#F0E442",
    "cross":   "#0072B2",
    "spiral":  "#CC79A7",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _load_twod(entangler="ising", N=10):
    """Return {target: {L: (kld_history[seeds,epochs], target_p, probs_final[seeds], kld[seeds])}}.

    Each twod__*.npz already bundles all seeds for one (entangler, target, L): kld_history has
    shape (seeds, epochs), probs_final (seeds, 2^N), kld (seeds,)."""
    out = {}
    for path in glob.glob(os.path.join(RUNS, "twod__*.npz")):
        z = np.load(path, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        if m.get("entangler") != entangler or m.get("N") != N:
            continue
        tgt = m.get("target")
        L   = int(m.get("L", 0))
        if tgt is None:
            continue
        out.setdefault(tgt, {})[L] = (
            np.asarray(z["kld_history"]) if "kld_history" in z.files else None,
            np.asarray(z["target_p"])   if "target_p"   in z.files else None,
            np.asarray(z["probs_final"]) if "probs_final" in z.files else None,
            np.asarray(z["kld"])         if "kld"         in z.files else None,
        )
    return out


def _log2_yaxis(ax):
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=2.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=2.0, subs=np.linspace(0.1, 0.9, 9), numticks=12)
    )
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


# --------------------------------------------------------------------------
# build figure
# --------------------------------------------------------------------------
def make_fig_twod_combined(ep_max=700):
    data = _load_twod("ising")
    avail = [(t, lab) for t, lab in TGT2D if t in data and data[t]]
    assert avail, "No kicked-Ising 2D data found."

    nc  = len(avail)   # 6 target columns
    N2D = 10
    nx  = 2 ** (N2D // 2)   # 32

    k = apply(14.0)

    # full-width horizontal: 2 rows (target / trained) x 6 cols (targets)
    fig = plt.figure(figsize=(14.0, 4.9))
    gs  = mgridspec.GridSpec(
        2, nc, figure=fig, height_ratios=[1.0, 1.0],
        hspace=0.05, wspace=0.05,
        left=0.045, right=0.997, top=0.995, bottom=0.005,
    )

    im_kw = dict(origin="lower", cmap="magma", interpolation="nearest",
                 extent=(0, 1, 0, 1), vmin=0)

    for j, (tgt, lab) in enumerate(avail):
        L = 12 if 12 in data[tgt] else max(data[tgt])
        _, tp, qf, kl = data[tgt][L]
        best = int(np.argmin(kl)) if kl is not None else 0
        q_best = qf[best] if qf is not None else np.zeros(nx * nx)
        tp_img = np.asarray(tp).reshape(nx, nx)
        q_img  = q_best.reshape(nx, nx)

        for row, img in enumerate([tp_img, q_img]):
            ax = fig.add_subplot(gs[row, j])
            ax.imshow(img / img.max(), **im_kw)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            lbl = chr(ord("a") + row * nc + j)           # (a)-(f) top, (g)-(l) bottom
            ax.text(0.06, 0.9, f"({lbl})", transform=ax.transAxes, color="white",
                    fontweight="bold", fontsize=FONT_SIZE_PANEL_LABEL * k * 0.8,
                    ha="left", va="top")
            if j == 0:                                   # row label (target / trained)
                ax.set_ylabel("target" if row == 0 else "trained",
                              fontsize=plt.rcParams["axes.labelsize"] * 0.95, labelpad=4 * k)

    paths = save(fig, FIGS, "fig_twod_combined")
    print(f"Saved: {paths[0]}")


def make_fig_twod_convergence(ep_max=700):
    """Single-column KLD-vs-epoch figure, one line per 2D target (kicked-Ising, best seed mean)."""
    data = _load_twod("ising")
    avail = [(t, lab) for t, lab in TGT2D if t in data and data[t]]
    assert avail, "No kicked-Ising 2D data found."
    apply(3.4)
    plt.rcParams.update({"axes.labelsize": 11, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 9.5, "legend.fontsize": 8.5})
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    for tgt, lab in avail:
        L = 12 if 12 in data[tgt] else max(data[tgt])
        kld_h, *_ = data[tgt][L]
        if kld_h is None:
            continue
        med = np.mean(kld_h, axis=0)[:ep_max]
        ax.plot(np.arange(len(med)), med, color=COL2D.get(tgt, "#999999"), label=lab, lw=1.9)
    _log2_yaxis(ax)
    grid(ax)
    ax.set_xlim(0, ep_max)
    ax.set_xlabel("epoch")
    ax.set_ylabel("KLD")
    ax.legend(frameon=False, ncol=2, loc="upper right",
              handlelength=1.2, columnspacing=0.9, labelspacing=0.25)
    fig.tight_layout(pad=0.3)
    paths = save(fig, FIGS, "fig_twod_convergence")
    print(f"Saved: {paths[0]}")


if __name__ == "__main__":
    make_fig_twod_combined()
    make_fig_twod_convergence()
