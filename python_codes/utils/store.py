"""
store.py -- one self-describing file per config (metadata + per-epoch history).

Every trained configuration is saved to its own .npz under data/runs/, containing:
  * meta        : JSON string with ALL config + hyperparameters
                  (sweep, N, K entangler depth, L, entangler kind, n_t, target name,
                   n_peaks, sigma_frac, seed, lr, n_epochs, n_real, clip, base_seed, ...)
  * kld_history : float32 array (n_real, n_epochs) -- KLD at every epoch, every seed
  * finals      : per-realization final metrics (kld, m2_final, s_half_final, conv)
                  and per-config scalars (magic_cost_Mp, s_half_target, m2_scrambler,
                  s_half_scrambler).
Files are written the moment a config finishes -> crash-safe and incremental.
Filenames are human-readable and unique per config.
"""
import json
import os

import numpy as np

from utils.io import data_dir


def runs_dir(sweep=None):
    d = os.path.join(data_dir(), "runs")
    os.makedirs(d, exist_ok=True)
    return d


def _fmt(v):
    if isinstance(v, float):
        return ("%g" % v).replace(".", "p").replace("-", "m")
    return str(v)


def run_filename(meta):
    """Human-readable, unique-per-config filename."""
    parts = [meta["sweep"],
             f"N{meta['N']}", f"K{meta['K']}", f"L{meta['L']}",
             meta["entangler"], f"nt{meta.get('n_t', 0)}",
             meta["target"], f"modes{meta.get('n_peaks', 'na')}",
             f"sig{_fmt(meta.get('sigma_frac', 'na'))}"]
    # tdoped targets vary ONLY in n_t_target (the target's T-count); it MUST be in the name or
    # all 9 targetmagic points collide to one file. Appended only for tdoped so existing 'mm'
    # filenames are unchanged.
    if meta.get("target") == "tdoped":
        parts.append(f"ntt{meta.get('n_t_target', 0)}")
    return "__".join(parts) + ".npz"


# scalar summaries pulled into summary.csv: (key, reduction). medians of per-realization finals.
_SUMMARY_REDUCE = (("kld", "med"), ("kld", "best"), ("m2_final", "med"), ("mq_final", "med"),
                   ("s_half_final", "med"), ("conv_rel10", "med"))
_SUMMARY_SCALARS = ("magic_cost_Mp", "s_half_target", "m2_scrambler", "s_half_scrambler", "page_svn")


def save_run(meta, kld_history, finals):
    """Write one config's self-describing file (all arrays in `finals` + full kld_history),
    then REBUILD summary.csv from every file on disk. `kld_history` is (n_real, n_epochs).
    Rebuilding (rather than appending) makes summary.csv a pure projection of the .npz files,
    so it can never silently drift from them. Returns path."""
    path = os.path.join(runs_dir(), run_filename(meta))
    payload = {"meta": json.dumps(meta),
               "kld_history": np.asarray(kld_history, dtype=np.float32)}
    for k, v in finals.items():
        payload[k] = np.asarray(v)
    np.savez_compressed(path, **payload)
    # In parallel runs many workers save concurrently; set SUMMARY_ON_SAVE=0 and let the
    # driver rebuild summary.csv centrally (periodically + once at the end) to avoid every
    # worker re-globbing the whole directory on every save.
    if os.environ.get("SUMMARY_ON_SAVE", "1") != "0":
        rebuild_summary()
    return path


def _summary_row(meta, get, filename):
    """Build one summary row. `get(key)` returns the stored array for `key` or None."""
    row = dict(meta)
    for k, red in _SUMMARY_REDUCE:
        arr = get(k)
        if arr is not None and np.asarray(arr).size:
            arr = np.asarray(arr)
            row[f"{k}_{red}"] = float(np.median(arr) if red == "med" else np.min(arr))
    for k in _SUMMARY_SCALARS:
        v = get(k)
        if v is not None:
            row[k] = float(v)
    row["file"] = filename
    return row


def rebuild_summary():
    """Regenerate summary.csv from scratch by scanning every run file in data/runs/.
    Only metadata + small scalar/1-D finals are read (np.load is lazy on .npz members, so the
    large weight/gate arrays are never decompressed), making this cheap to call after each save.
    Tolerant of older files that lack newer columns. Returns the DataFrame."""
    import glob
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(os.path.join(runs_dir(), "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "meta" not in z.files:            # skip auxiliary outputs (e.g. xyz__*.npz) that aren't sweep runs
            continue
        meta = json.loads(str(z["meta"]))
        keys = set(z.files)
        rows.append(_summary_row(meta, lambda k: z[k] if k in keys else None, os.path.basename(f)))
    df = pd.DataFrame(rows)
    # atomic write (tmp + os.replace on same fs) so concurrent readers/writers never see a
    # partial file; last complete snapshot wins.
    import tempfile
    csv = os.path.join(runs_dir(), "summary.csv")
    fd, tmp = tempfile.mkstemp(dir=runs_dir(), suffix=".csv.tmp")
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv)
    return df


def load_df(sweep=None):
    """The pandas summary of all runs (one row per config). Filter by sweep."""
    import pandas as pd
    csv = os.path.join(runs_dir(), "summary.csv")
    if not os.path.exists(csv):
        return pd.DataFrame()
    df = pd.read_csv(csv)
    return df[df.sweep == sweep] if sweep is not None else df


def load_runs(sweep=None, **filters):
    """Load all run files (optionally filtered by sweep and meta key=value) into a list
    of dicts: {**meta, kld_history, **finals}. For building figures from stored data."""
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(runs_dir(), "*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "meta" not in z.files:            # skip auxiliary outputs (e.g. xyz__*.npz) that aren't sweep runs
            continue
        meta = json.loads(str(z["meta"]))
        if sweep is not None and meta.get("sweep") != sweep:
            continue
        if any(meta.get(k) != v for k, v in filters.items()):
            continue
        rec = dict(meta)
        rec["kld_history"] = z["kld_history"]
        for k in z.files:
            if k not in ("meta", "kld_history"):
                rec[k] = z[k]
        rec["_path"] = f
        out.append(rec)
    return out
