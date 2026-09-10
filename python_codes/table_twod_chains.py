"""
table_twod_chains.py -- Appendix table: 2D targets, four entanglers, independent curriculum chains.

For every (target, entangler) and every independent chain (sweep 'twodch<c>'), take the final
depth L=12 and record the lowest-loss realization's KLD and the median over realizations. Report
per (target, entangler) the mean and standard deviation over chains of the lowest-loss KLD, and
per target the chain-to-chain scatter versus the spread between entanglers.

Run:  python3 table_twod_chains.py   -> prints a LaTeX tabular body + summary lines
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from utils.store import runs_dir

TARGETS = [("mm2d", "four lobes"), ("rings", "two rings"), ("moons", "two moons"),
           ("stripes", "three stripes"), ("cross", "diagonal cross"), ("spiral", "two spirals")]
ENT = [("ising", "kicked Ising"), ("cluster", "cluster"), ("haar", "local Haar"), ("matchgate", "matchgate")]
L_FINAL = int(os.environ.get("L_FINAL", "12"))


def load():
    d = {}
    for p in glob.glob(os.path.join(runs_dir(), "twodch*__*.npz")):
        z = np.load(p, allow_pickle=True); m = json.loads(str(z["meta"]))
        if m["L"] != L_FINAL or m["N"] != 10:
            continue
        c = int(m["sweep"].replace("twodch", ""))
        k = np.asarray(z["kld"])
        d.setdefault((m["target"], m["entangler"]), {})[c] = (float(k.min()), float(np.median(k)))
    return d


def main():
    d = load()
    print(f"% L={L_FINAL}; entries: mean +- std over chains of the lowest-loss KLD (median over realizations in parentheses)")
    rows = []
    for t, tl in TARGETS:
        cells = []; best_means = []
        for e, el in ENT:
            ch = d.get((t, e), {})
            if not ch:
                cells.append("--"); continue
            b = np.array([v[0] for v in ch.values()]); md = np.array([v[1] for v in ch.values()])
            best_means.append(b.mean())
            cells.append(f"${b.mean():.3f}\\pm{b.std(ddof=1) if len(b)>1 else 0:.3f}$ ({md.mean():.2f})")
        rows.append(f"{tl} & " + " & ".join(cells) + r" \\")
        # chain scatter vs entangler spread
        sc = [np.std([v[0] for v in d[(t, e)].values()], ddof=1) for e, _ in ENT if len(d.get((t, e), {})) > 1]
        print(f"% {t}: n_chains={[len(d.get((t,e),{})) for e,_ in ENT]} chain-scatter(mean std)={np.mean(sc):.3f} "
              f"entangler-spread(max-min of means)={max(best_means)-min(best_means):.3f}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
