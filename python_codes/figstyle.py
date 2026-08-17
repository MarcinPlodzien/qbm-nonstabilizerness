"""
================================================================================
 figstyle.py -- single source of truth for publication-quality paper figures
================================================================================
Every figure script imports this so fonts, colors, line widths, panel tags, and
saving are IDENTICAL across the paper. Tuned for a single PRA column (~3.4 in).
Palette is the colorblind-safe Okabe-Ito set, fixed per entangler family.

Usage in a per-figure script:
    import figstyle as fs
    fs.setup()
    fig, ax = plt.subplots(figsize=fs.SINGLE)
    ax.plot(x, y, color=fs.COLOR["clifford"], label=fs.LABEL["clifford"])
    fs.panel_tag(ax, "(a)")
    fs.save(fig, os.path.join(FIGS, "fig_name"))
================================================================================
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- canvas sizes (inches) ----
SINGLE   = (3.40, 2.55)      # one single-column panel
SINGLE_T = (3.40, 2.10)      # short single-column panel
STACK2   = (3.40, 4.20)      # two stacked single-column panels
STACK3   = (3.40, 6.00)      # three stacked single-column panels
DOUBLE   = (7.00, 2.60)      # two-column (figure*)

# ---- colorblind-safe palette (Okabe-Ito), fixed per entangler family ----
COLOR = {
    "clifford":  "#009E73",  # green
    "doped":     "#CC79A7",  # pink
    "haar":      "#0072B2",  # blue
    "matchgate": "#E69F00",  # orange
    "cluster":   "#56B4E9",  # light blue
    "ising":     "#009E73",  # green (kicked-Ising: Clifford-point analogue)
}
LABEL = {
    "clifford": "Clifford", "doped": "doped-Clifford", "haar": "Haar",
    "matchgate": "matchgate", "cluster": "cluster", "ising": "kicked-Ising",
}
MARKER = {"clifford": "s", "doped": "^", "haar": "D",
          "matchgate": "o", "cluster": "v", "ising": "P"}
GREY = "0.45"

# ---- font sizes (points) ----
FS_LABEL, FS_TICK, FS_LEGEND, FS_TAG, FS_ANNOT = 9.5, 8.0, 7.5, 10.0, 7.5
LW, MS = 1.6, 3.0


def setup():
    """Apply the common rcParams. Call once at the top of every figure script."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL, "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "lines.linewidth": LW, "lines.markersize": MS,
        "legend.frameon": False, "legend.handlelength": 1.3,
        "legend.labelspacing": 0.25, "legend.borderaxespad": 0.3,
        "legend.columnspacing": 1.0,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def panel_tag(ax, s, x=0.035, y=0.955):
    """Small bold (a)/(b)/(c) in a corner (default upper-left)."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=FS_TAG,
            fontweight="bold", va="top", ha="left")


def band(ax, x, mu, sd, color, alpha=0.14):
    """Shaded +/- band."""
    import numpy as np
    x = np.asarray(x); mu = np.asarray(mu); sd = np.asarray(sd)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=alpha, lw=0)


def save(fig, path_noext):
    """Write <path>.pdf and <path>.png (300 dpi) and close the figure."""
    fig.savefig(path_noext + ".pdf")
    fig.savefig(path_noext + ".png", dpi=300)
    plt.close(fig)
