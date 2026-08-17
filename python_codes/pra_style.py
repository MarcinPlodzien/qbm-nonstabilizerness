"""PRA figure style for the nonstabilizerness paper, ported from the accepted QSBM
paper's PRA_figure_codes/pra_style.py so the two papers share one look.

Recipe (identical to QSBM Figs. 2-11): draw on a wide canvas and \\includegraphics at
\\columnwidth (246 pt). A constant of size s on a W-inch canvas prints at s*246/(72*W),
so apply(W) scales every rc constant by W/14 to hold Figure-3's *printed* type/stroke
weight on any canvas -- a 7 in single-panel and a 14 in three-panel land at the same
apparent size on the page. Panel labels (a)/(b)/(c), no panel titles, no suptitles.
save() writes BOTH pdf (vector, for the manuscript) and png at 300 dpi.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- reference canvas / fonts (QSBM get_fig_3.py) ---
FIG_WIDTH = 14
FIG_DPI = 300
FONT_SIZE_GLOBAL = 24
FONT_SIZE_AXIS_LABEL = 22
FONT_SIZE_TICK_LABEL = 22
FONT_SIZE_LEGEND = 18
FONT_SIZE_PANEL_LABEL = 22
MARKER_SIZE = 6
CAPSIZE = 3
LINEWIDTH = 3
GRID_MAJOR_ALPHA = 0.5
GRID_MINOR_STYLE = ":"
GRID_MINOR_ALPHA = 0.3
FONT_BOOST = 1.18                     # QSBM boost so small-canvas constants read at 8-11 pt

# ---- entangler ladder palette (Okabe-Ito colourblind-safe) + markers ----
# Two zero-SRE analog/discrete Clifford entanglers (ising, cluster), the free-fermion
# matchgate, the zero-SRE random Clifford control, the tunable doped-Clifford, and Haar.
COL = {"matchgate": "#E69F00", "clifford": "#009E73", "doped": "#CC79A7", "haar": "#0072B2",
       "cluster": "#56B4E9", "ising": "#009E73"}
MRK = {"matchgate": "o", "clifford": "s", "doped": "^", "haar": "D",
       "cluster": "v", "ising": "P"}
LBL = {"matchgate": "matchgate", "clifford": "Clifford", "doped": "doped-Clifford",
       "haar": "Haar", "cluster": "cluster", "ising": "kicked-Ising"}

RC = {
    "font.size": FONT_SIZE_GLOBAL,
    "axes.titlesize": FONT_SIZE_GLOBAL,
    "axes.labelsize": FONT_SIZE_AXIS_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK_LABEL,
    "ytick.labelsize": FONT_SIZE_TICK_LABEL,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "lines.linewidth": LINEWIDTH,
    "lines.markersize": MARKER_SIZE,
    "figure.constrained_layout.use": True,
    "axes.spines.top": True,
    "axes.spines.right": True,
}
_FONT_KEYS = {"font.size", "axes.titlesize", "axes.labelsize",
              "xtick.labelsize", "ytick.labelsize", "legend.fontsize"}


def apply(fig_width=FIG_WIDTH):
    """Apply the shared style rescaled for a canvas of width `fig_width`; returns the
    scale k = fig_width/14 so callers can scale capsize/labelpad/extra linewidths too."""
    k = fig_width / FIG_WIDTH
    rc = {}
    for key, val in RC.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            rc[key] = val
        else:
            rc[key] = val * k * (FONT_BOOST if key in _FONT_KEYS else 1.0)
    plt.rcParams.update(rc)
    return k


_TAG_POS = {
    "upper right": (0.95, 0.95, "right", "top"),
    "upper left": (0.05, 0.95, "left", "top"),
    "lower left": (0.05, 0.05, "left", "bottom"),
    "lower right": (0.95, 0.05, "right", "bottom"),
}


def panel_tag(ax, tag, loc="upper left", k=1.0):
    """Bold (a)/(b)/(c) panel label; no panel titles anywhere (PRA style)."""
    x, y, ha, va = _TAG_POS[loc]
    ax.text(x, y, tag, transform=ax.transAxes,
            fontsize=FONT_SIZE_PANEL_LABEL * k * FONT_BOOST,
            fontweight="bold", ha=ha, va=va)


def grid(ax):
    ax.grid(False)                        # no in-panel guide lines (removed per house style)


def hline(ax, y, k=1.0, color="red", label=None):
    """Dashed reference line (e.g. Page value / KLD floor), matched to QSBM's Haar line."""
    ax.axhline(y, color=color, linestyle="--", linewidth=5 * k, alpha=0.7, label=label)


def save(fig, outdir, stem):
    import os
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths
