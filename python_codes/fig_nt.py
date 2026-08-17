"""
================================================================================
 fig_nt.py -- doping (entangler SRE) has no effect; trained M2 saturates at N-2
================================================================================
Dedicated figure on the common style (figstyle.py). Single-column, two panels
(different x-axes, so side by side):
  (a) converged KLD vs probe-state SRE M2(U_S|+>), tuned by T-doping at fixed
      entanglement (N=10, L=10) -- flat.
  (b) trained-state M2 vs N at the deepest converged L -- saturates at N-2.
All labels/annotations inside the frame.

USAGE:  python3 fig_nt.py    OUTPUT: figures/fig_nt.{pdf,png}
================================================================================
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from utils.store import runs_dir
import figstyle as fs

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
RUNS = runs_dir()


def _meta(z):
    return json.loads(str(z["meta"]))


def doping():
    """Group the dedicated doping sweep (N=10, L=10) by n_t -> (probe M2, KLD mean, KLD std)."""
    by = {}
    for p in glob.glob(os.path.join(RUNS, "nt__*.npz")):
        z = np.load(p, allow_pickle=True); m = _meta(z)
        if m.get("sweep") != "nt" or m.get("N") != 10:
            continue
        d = by.setdefault(m.get("n_t"), {"m2": [], "kld": []})
        if "m2_scrambler" in z.files:
            d["m2"].append(float(np.mean(z["m2_scrambler"])))
        d["kld"].extend(float(v) for v in np.asarray(z["kld"]))
    nts = sorted(by)
    xm = np.array([np.mean(by[n]["m2"]) for n in nts])
    ym = np.array([np.mean(by[n]["kld"]) for n in nts])
    ye = np.array([np.std(by[n]["kld"]) for n in nts])
    return xm, ym, ye


def trained_m2_vs_N():
    Ns, m2N = [], []
    for N in (8, 10, 12):
        vals = []
        for sw in ("decoupling", "deepl"):
            for p in glob.glob(os.path.join(RUNS, f"{sw}__*.npz")):
                z = np.load(p, allow_pickle=True); m = _meta(z)
                if m.get("N") != N or "m2_final" not in z.files:
                    continue
                vals.append((m.get("L"), float(np.mean(z["m2_final"]))))
        if vals:
            Lmax = max(v[0] for v in vals)
            Ns.append(N); m2N.append(np.mean([v[1] for v in vals if v[0] == Lmax]))
    return np.array(Ns), np.array(m2N)


def main():
    xm, ym, ye = doping()
    Ns, m2N = trained_m2_vs_N()
    print(f"doping: M2 0..{xm.max():.1f}, KLD~{ym.mean():.3f}; trained M2(N)={[round(v,1) for v in m2N]}", flush=True)

    fs.setup()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(3.4, 1.85),
                                 gridspec_kw={"wspace": 0.42})

    # (a) KLD vs probe-state SRE, doping
    a0.errorbar(xm, ym, yerr=ye, marker="o", ms=3.0, color=fs.COLOR["doped"],
                capsize=2, lw=fs.LW)
    a0.axhline(float(np.mean(ym)), ls="--", color=fs.GREY, lw=1.0)
    a0.set_ylim(0, float(np.max(ym + ye)) * 1.35)
    a0.set_xlabel(r"entangler $M_2(U_S|{+}\rangle)$")
    a0.set_ylabel(r"KLD")
    fs.panel_tag(a0, "(a)")

    # (b) trained M2 vs N
    a1.plot(Ns, [n - 2 for n in Ns], "--", color="0.35", lw=1.2, label=r"$N-2$")
    a1.plot(Ns, m2N, "s", ms=4.5, color=fs.COLOR["clifford"], ls="none", label="trained")
    a1.set_xlabel(r"qubits $N$")
    a1.set_ylabel(r"$M_2(\psi_\theta)$")
    a1.set_xticks(Ns)
    a1.legend(loc="lower right")
    fs.panel_tag(a1, "(b)")

    fs.save(fig, os.path.join(FIGS, "fig_nt"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
