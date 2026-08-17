"""Filename-key integrity: the .npz filename MUST be a faithful, injective key of a run's config.
A collision silently overwrites/skips data (this exact bug corrupted the targetmagic sweep once).
Run:  python3 tests/test_filenames.py     (or: pytest tests/test_filenames.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_sweep as R
from run_sweep import _cfg_filename, _ID_FIELDS


def test_every_sweep_has_unique_filenames():
    """No two configs in any sweep (any N in the plan) may share a filename."""
    for sweep, N in R.OVERNIGHT_PLAN + [("kscan", 8), ("richness", 12), ("nt", 8)]:
        cfgs = R.grid(sweep, N)
        names = [_cfg_filename(c) for c in cfgs]
        assert len(set(names)) == len(cfgs), (
            f"COLLISION in sweep={sweep} N={N}: {len(cfgs)} configs -> {len(set(names))} filenames")


def test_filename_injective_on_each_id_field():
    """Changing ANY identifying field must change the filename (else that field can collide)."""
    base = dict(sweep="targetmagic", N=10, K=20, L=10, entangler="clifford", n_t=0,
                target="tdoped", n_peaks="na", sigma_frac="na", n_t_target=0)
    base_name = _cfg_filename(base)
    variants = dict(N=8, K=16, L=6, entangler="haar", n_t=3, target="mm",
                    n_peaks=5, sigma_frac=0.05, n_t_target=7)
    for field, val in variants.items():
        c = dict(base); c[field] = val
        assert _cfg_filename(c) != base_name, f"filename ignores field {field!r} -> silent collision risk"


def test_targetmagic_regression():
    """The specific bug: all 9 T-doping levels must map to 9 distinct files."""
    cfgs = R.grid("targetmagic", 10)
    names = [_cfg_filename(c) for c in cfgs]
    assert len(set(names)) == len(cfgs) == 18, "targetmagic n_t_target collision regressed"


if __name__ == "__main__":
    test_every_sweep_has_unique_filenames()
    test_filename_injective_on_each_id_field()
    test_targetmagic_regression()
    print("OK: all filename-key integrity tests passed "
          f"(checked uniqueness across sweeps + injectivity on {len(_ID_FIELDS)} id fields)")
