"""
================================================================================
 fig_gibbs_kld.py -- KLD & trained magic vs temperature (SK-Gibbs), one figure
================================================================================
Dedicated script for the entangler-comparison figure on the spin-glass Gibbs
target. Reads the per-config sweep data/runs/gibbs__N{N}__{model}__*.npz and
draws three stacked single-column panels, disorder-averaged, in the QSBM house
style (figstyle.py): all labels/annotations INSIDE the frame, no panel titles.

  (a) KLD vs beta per entangler family (mean +/- disorder std).
  (b) trained-state M2 (best seed) vs beta, with the target floor M2,min.
  (c) paired difference <KLD_X - KLD_Clifford>_disorder (the decoupling test).

USAGE:  GRID_N=10 MODEL=SK ANSATZ=main python3 fig_gibbs_kld.py
OUTPUT: figures/fig_gibbs_kld_N{N}_{model}_{ansatz}.{pdf,png}
================================================================================
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from utils.store import runs_dir
import figstyle as fs

N      = int(os.environ.get("GRID_N", "10"))
MODEL  = os.environ.get("MODEL", "SK")
ANSATZ = os.environ.get("ANSATZ", "main")
FAMS   = ["clifford", "doped", "haar", "matchgate"]
FIGS   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def load():
    from collections import defaultdict
    kld = defaultdict(lambda: defaultdict(dict))
    m2tr = defaultdict(lambda: defaultdict(dict))
    m2min = defaultdict(list)
    for p in glob.glob(os.path.join(runs_dir(), f"gibbs__N{N}__{MODEL}__*.npz")):
        if "couplings" in p:
            continue
        z = np.load(p, allow_pickle=True); m = json.loads(str(z["meta"]))
        if m.get("ansatz") != ANSATZ:
            continue
        fam, r, b = m["entangler"], int(m["realization_idx"]), float(m["beta"])
        kld[fam][b][r] = float(np.median(np.asarray(z["kld"])))
        m2tr[fam][b][r] = float(z["m2_trained"])
        if fam == "clifford":
            m2min[b].append(float(z["m2min_target"]))
    return kld, m2tr, m2min


def main():
    kld, m2tr, m2min = load()
    betas = sorted(m2min)
    if not betas:
        print("no data for", N, MODEL, ANSATZ); return
    nd = max(len(v) for v in kld["clifford"].values())
    print(f"N={N} {MODEL} {ANSATZ}: {len(betas)} betas (<= {betas[-1]}), up to {nd} disorders/pt", flush=True)

    kmu = {f: np.array([np.mean(list(kld[f][b].values())) for b in betas]) for f in FAMS}
    ksd = {f: np.array([np.std(list(kld[f][b].values()))  for b in betas]) for f in FAMS}
    mmu = {f: np.array([np.mean(list(m2tr[f][b].values())) for b in betas]) for f in FAMS}
    m2m = np.array([np.mean(m2min[b]) for b in betas])
    dmu, dse = {}, {}
    for f in FAMS[1:]:
        vals = [[kld[f][b][r] - kld["clifford"][b][r] for r in set(kld[f][b]) & set(kld["clifford"][b])] for b in betas]
        dmu[f] = np.array([np.mean(v) if v else np.nan for v in vals])
        dse[f] = np.array([np.std(v) / np.sqrt(max(1, len(v))) if v else np.nan for v in vals])

    fs.setup()
    fig, (a, b2, c) = plt.subplots(3, 1, figsize=fs.STACK3, sharex=True,
                                   gridspec_kw={"hspace": 0.10})

    # (a) KLD vs beta
    for f in FAMS:
        a.plot(betas, kmu[f], "-", color=fs.COLOR[f], label=fs.LABEL[f])
        fs.band(a, betas, kmu[f], ksd[f], fs.COLOR[f], alpha=0.10)
    a.set_ylabel(r"$\mathrm{KLD}(p\,\Vert\,q_\theta)$")
    a.set_ylim(bottom=0)
    a.legend(loc="lower right")
    fs.panel_tag(a, "(a)")

    # (b) trained M2 + target floor  (annotations inside frame)
    for f in FAMS:
        b2.plot(betas, mmu[f], "-", color=fs.COLOR[f])
    b2.plot(betas, m2m, "--", color="0.25", lw=1.3, label=r"$M_{2,\min}$ (target)")
    b2.axhline(N - 2, color=fs.GREY, ls=":", lw=1.0)
    b2.text(0.97, N - 2 - 0.10, r"$N-2$", transform=b2.get_yaxis_transform(),
            ha="right", va="top", fontsize=fs.FS_ANNOT, color="0.35")
    b2.set_ylabel(r"trained $M_2(\psi_\theta)$")
    b2.set_ylim(bottom=0)
    b2.legend(loc="center right")
    fs.panel_tag(b2, "(b)")

    # (c) paired difference
    c.axhline(0, color=fs.GREY, lw=0.8)
    for f in FAMS[1:]:
        c.plot(betas, dmu[f], "-o", color=fs.COLOR[f], lw=1.3, ms=2.8, label=fs.LABEL[f])
        fs.band(c, betas, dmu[f], dse[f], fs.COLOR[f], alpha=0.15)
    c.set_xlabel(r"inverse temperature $\beta$")
    c.set_ylabel(r"$\langle\mathrm{KLD}_X-\mathrm{KLD}_{\mathrm{Cl}}\rangle$")
    c.legend(loc="upper left", ncol=3)
    fs.panel_tag(c, "(c)", y=0.86)

    fig.align_ylabels([a, b2, c])
    fs.save(fig, os.path.join(FIGS, f"fig_gibbs_kld_N{N}_{MODEL}_{ANSATZ}"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
