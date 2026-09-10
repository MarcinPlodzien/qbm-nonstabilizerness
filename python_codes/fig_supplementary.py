"""
================================================================================
 fig_supplementary.py -- the four claims previously carried as "(not shown)"
================================================================================
(a) Target sweep: converged KLD against the target floor M_2,min, over the peak-width and
    peak-number scans (sweep 'richness'), one point per (target, family) at N=10, L=10.
(b) Paired differences KLD_X - KLD_Clifford against depth L at N=10, formed on the shared
    initialization and target, mean and standard error over the 20 realizations.
(c) Trained-state half-chain entropy relative to the Page value against depth L, N=10, all
    four families, showing that every family reaches volume law under reuse.
(d) SK Gibbs matchgate offset against circuit depth L at beta = 2, disorder-averaged, showing
    the offset closing as the circuit deepens.

USAGE:  python3 fig_supplementary.py   OUTPUT: figures/fig_supplementary.{pdf,png}
================================================================================
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from utils.store import runs_dir
from utils.sre import page_entropy
import figstyle as fs

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
R = runs_dir()
FAM = ["clifford", "doped", "haar", "matchgate"]


def meta(z):
    return json.loads(str(z["meta"]))


def panel_a(ax):
    """converged KLD vs target floor over the richness (peak width / peak number) scans"""
    pts = {}
    for p in glob.glob(os.path.join(R, "richness__N10__K20__L10__*.npz")):
        z = np.load(p, allow_pickle=True); m = meta(z)
        pts.setdefault(m["entangler"], []).append(
            (float(np.mean(z["magic_cost_Mp"])), float(np.mean(z["kld"]))))
    for e in FAM:
        if e not in pts: continue
        v = np.array(sorted(pts[e]))
        ax.plot(v[:, 0], v[:, 1], marker=fs.MARKER[e], color=fs.COLOR[e], ls="none",
                ms=4.5, mfc="none", label=fs.LABEL[e])
    ax.set_xlabel(r"target floor $M_{2,\min}$"); ax.set_ylabel(r"KLD")
    fs.panel_tag(ax, "(a)", x=0.035, y=0.90)


def panel_b(ax):
    """paired differences vs depth, N=10"""
    byL = {}
    for sw in ("decoupling", "deepl"):
        for p in glob.glob(os.path.join(R, f"{sw}__N10__K20__L*__*mm*.npz")):
            z = np.load(p, allow_pickle=True); m = meta(z)
            byL.setdefault(m["L"], {})[m["entangler"]] = np.asarray(z["kld"])
    Ls = sorted(L for L, d in byL.items() if "clifford" in d and len(d) == 4)
    for e in FAM[1:]:
        mu, se = [], []
        for L in Ls:
            d = byL[L][e] - byL[L]["clifford"]
            mu.append(d.mean()); se.append(d.std(ddof=1) / np.sqrt(len(d)))
        ax.errorbar(Ls, mu, yerr=se, marker=fs.MARKER[e], color=fs.COLOR[e],
                    label=fs.LABEL[e], capsize=2, lw=fs.LW)
    ax.axhline(0.0, color=fs.GREY, lw=0.8)
    ax.set_xlabel(r"layers $L$"); ax.set_ylabel(r"$\langle\mathrm{KLD}_X-\mathrm{KLD}_{\mathrm{Cl}}\rangle$")
    fs.panel_tag(ax, "(b)", x=0.035, y=0.90)


def panel_c(ax):
    """trained half-chain entropy / Page vs depth, N=10"""
    pg = page_entropy(10); byL = {}
    for sw in ("decoupling", "deepl"):
        for p in glob.glob(os.path.join(R, f"{sw}__N10__K20__L*__*mm*.npz")):
            z = np.load(p, allow_pickle=True); m = meta(z)
            if "s_half_final" in z.files:
                byL.setdefault(m["entangler"], {})[m["L"]] = float(np.mean(z["s_half_final"])) / pg
    for e in FAM:
        if e not in byL: continue
        Ls = sorted(byL[e])
        ax.plot(Ls, [byL[e][L] for L in Ls], marker=fs.MARKER[e], color=fs.COLOR[e], label=fs.LABEL[e])
    ax.axhline(1.0, ls="--", color=fs.GREY, lw=1.0)
    ax.set_xlabel(r"layers $L$"); ax.set_ylabel(r"$S_{vN}/S_{\mathrm{Page}}$")
    ax.set_ylim(0.4, 1.12)
    fs.panel_tag(ax, "(c)", x=0.035, y=0.16)


def panel_d(ax):
    """matchgate minus Clifford on the SK target at beta=2, vs depth"""
    d = {}
    for p in glob.glob(os.path.join(R, "gibbs__N10__SK__*main__L*__*.npz")):
        if "couplings" in p: continue
        z = np.load(p, allow_pickle=True); m = meta(z)
        if m["entangler"] not in ("clifford", "matchgate") or abs(float(m["beta"]) - 2.0) > 1e-9: continue
        d.setdefault(m["L"], {}).setdefault(m["entangler"], {})[m["realization_idx"]] = float(np.median(z["kld"]))
    Ls, mu, se = [], [], []
    for L in sorted(d):
        c, g = d[L].get("clifford", {}), d[L].get("matchgate", {})
        common = sorted(set(c) & set(g))
        if len(common) < 3: continue
        v = np.array([g[r] - c[r] for r in common])
        Ls.append(L); mu.append(v.mean()); se.append(v.std(ddof=1) / np.sqrt(len(v)))
    ax.errorbar(Ls, mu, yerr=se, marker=fs.MARKER["matchgate"], color=fs.COLOR["matchgate"],
                capsize=2, lw=fs.LW)
    ax.axhline(0.0, color=fs.GREY, lw=0.8)
    ax.set_xlabel(r"layers $L$"); ax.set_ylabel(r"$\langle\mathrm{KLD}_{\mathrm{mg}}-\mathrm{KLD}_{\mathrm{Cl}}\rangle$")
    fs.panel_tag(ax, "(d)", x=0.035, y=0.90)


def main():
    fs.setup()
    fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6), gridspec_kw={"hspace": 0.42, "wspace": 0.34})
    panel_a(ax[0, 0]); panel_b(ax[0, 1]); panel_c(ax[1, 0]); panel_d(ax[1, 1])
    h, l = ax[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=fs.FS_ANNOT, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.19)
    out = os.path.join(FIGS, "fig_supplementary")
    fig.savefig(out + ".pdf", bbox_inches="tight"); fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("DONE", flush=True)


if __name__ == "__main__":
    main()
