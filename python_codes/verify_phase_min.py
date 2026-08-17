"""
================================================================================
 verify_phase_min.py -- is phi=0 the minimizer of M_2 for the reported targets?
================================================================================
M_{2,min}(p) = min_phi M_2( sum_x sqrt(p_x) e^{i phi_x}|x> ) is non-convex, and we
report the phi=0 branch M_2(sqrt p) (an UPPER bound). This script checks that no
nonzero phase improves on phi=0 for the targets where it is load-bearing (the
SK-Gibbs targets), by running the multi-start phase descent (magic_cost, which
also seeds phi=0) and comparing to magic_cost_upper. If descent gives no
improvement, the upper bound is tight and M_{2,min} = M_2(sqrt p).

USAGE:  JAX_PLATFORMS=cpu python3 verify_phase_min.py
================================================================================
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from utils.targets import magic_cost, magic_cost_upper

N = 10
BETAS = ["0p60", "1p00", "1p40", "2p00", "3p00", "6p00"]   # around and beyond the magic peak


def main():
    print(f"phi=0 optimality check, SK-Gibbs targets, N={N} (upper = M2(sqrt p); "
          f"descent = multi-start incl. phi=0)", flush=True)
    worst = 0.0
    for b in BETAS:
        fs = glob.glob(f"../data/runs/gibbs__N{N}__SK__*clifford__realization_idx0__beta{b}.npz")
        if not fs:
            continue
        p = np.asarray(np.load(fs[0], allow_pickle=True)["target_p"])
        up = magic_cost_upper(p, N)
        opt = magic_cost(p, N, steps=400, n_restarts=4)
        imp = up - opt                                   # >0 => a nonzero phase beat phi=0
        worst = max(worst, imp)
        print(f"  beta={b.replace('p','.'):>5}: M2(sqrt p)={up:6.3f}   descent min={opt:6.3f}   "
              f"improvement={imp:+.4f}", flush=True)
    verdict = "phi=0 IS the minimizer (bound tight)" if worst < 1e-2 else "phi=0 NOT optimal"
    print(f"\nmax improvement over phi=0: {worst:.4f}  ->  {verdict}", flush=True)


if __name__ == "__main__":
    main()
