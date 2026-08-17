"""
================================================================================
 fig_combined.py -- 1D five-peak target: distribution overlay + KLD vs depth
================================================================================
Dedicated headline figure on the common style (figstyle.py). Single-column, two
stacked panels (N=10, 5-mode target):
  (a) target distribution (shaded) with the trained Born output of every
      entangler family overlaid (mean +/- std over realizations) -- identical.
  (b) converged KLD vs circuit depth L (log2), four families coincident.
All labels/annotations inside the frame.

USAGE:  python3 fig_combined.py   OUTPUT: figures/fig_combined.{pdf,png}
================================================================================
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.store import runs_dir
import figstyle as fs

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
RUNS = runs_dir()
LAD  = ["matchgate", "clifford", "doped", "haar"]


def load(sweep, **filters):
    out = []
    for path in glob.glob(os.path.join(RUNS, f"{sweep}__*.npz")):
        z = np.load(path, allow_pickle=True); m = json.loads(str(z["meta"]))
        if any(m.get(k) != v for k, v in filters.items()):
            continue
        rec = dict(m)
        rec["kld"] = z["kld"]
        for key in ("target_p", "probs_final"):
            if key in z.files:
                rec[key] = z[key]
        out.append(rec)
    return out


def _coarsegrain(v, nb=256):
    g = len(v) // nb
    return v[: g * nb].reshape(nb, g).sum(1)


def _log2_yaxis(ax):
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=2.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=2.0, subs=np.linspace(0.1, 0.9, 9), numticks=12))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def main():
    recs = load("decoupling", N=10) + load("deepl", N=10)
    assert recs, "no decoupling/deepl N=10 records"

    dist_recs = [r for r in recs if "probs_final" in r]
    Lmax = max(r["L"] for r in dist_recs)
    by_ent = {r["entangler"]: r for r in dist_recs if r["L"] == Lmax}
    nb = 256
    x = np.arange(nb) / (nb - 1)
    pt = _coarsegrain(np.asarray(by_ent[LAD[0]]["target_p"]))

    kld_by_ent = {}
    for e in LAD:
        rows = sorted([r for r in recs if r["entangler"] == e], key=lambda r: r["L"])
        if not rows:
            continue
        kld_by_ent[e] = ([r["L"] for r in rows],
                         np.array([float(np.mean(r["kld"])) for r in rows]),
                         np.array([float(np.std(r["kld"])) for r in rows]))
    print(f"combined: dist at L={Lmax}, KLD families={list(kld_by_ent)}", flush=True)

    fs.setup()
    fig, (ax_d, ax_k) = plt.subplots(2, 1, figsize=(3.4, 4.4),
                                     gridspec_kw={"height_ratios": [1.0, 1.35], "hspace": 0.34})

    # (a) distribution overlay -- all four families lie on top of the target, so no
    # per-family legend here (it would be four indistinguishable curves); the family
    # legend lives in panel (b) with the same colours.
    ax_d.fill_between(x, pt, color="0.82", lw=0)
    for e in LAD:
        if e not in by_ent:
            continue
        q = np.asarray(by_ent[e]["probs_final"]); qn = q / q.sum(1, keepdims=True)
        qc = np.stack([_coarsegrain(qn[i]) for i in range(qn.shape[0])])
        mn, sd = qc.mean(0), qc.std(0)
        ax_d.plot(x, mn, color=fs.COLOR[e], lw=1.2)
        ax_d.fill_between(x, mn - sd, mn + sd, color=fs.COLOR[e], alpha=0.16, lw=0)
    ax_d.set_xlim(0, 1); ax_d.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_d.set_yticks([])
    ax_d.set_xlabel(r"$x$"); ax_d.set_ylabel("probability")
    ax_d.set_ylim(top=pt.max() * 1.18)
    ax_d.text(0.965, 0.93, "shaded: target", transform=ax_d.transAxes,
              ha="right", va="top", fontsize=fs.FS_ANNOT, color="0.35")
    fs.panel_tag(ax_d, "(a)")

    # (b) KLD vs L (log2) -- carries the shared family legend
    for e in LAD:
        if e not in kld_by_ent:
            continue
        Ls, mean, std = kld_by_ent[e]
        lo = np.clip(mean - std, mean * 0.05, None)
        ax_k.errorbar(Ls, mean, yerr=[mean - lo, std], marker=fs.MARKER[e],
                      color=fs.COLOR[e], label=fs.LABEL[e], capsize=2, lw=fs.LW)
    _log2_yaxis(ax_k)
    ax_k.set_xlabel(r"layers $L$"); ax_k.set_ylabel(r"KLD")
    ax_k.legend(loc="upper right", ncol=2)
    fs.panel_tag(ax_k, "(b)", x=0.035, y=0.14)

    fs.save(fig, os.path.join(FIGS, "fig_combined"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
