"""
================================================================================
 io.py -- results storage
================================================================================

CONVENTION
    * one tidy pandas DataFrame per experiment: ONE ROW PER RUN
      (scrambler kind, depth K, layers L, seed) x (all config parameters)
    * every model parameter appears BOTH in the filename and as a column,
      so a stray file is always self-describing and a reloaded frame never
      needs the config object to be interpreted
    * data/  <experiment_tag>.parquet   (fast, typed)  + .csv (human/grep-able)

    Paths are resolved relative to the project root, so the whole folder can be
    moved anywhere.
================================================================================
"""
import os

import pandas as pd

__all__ = ["project_root", "data_dir", "figures_dir", "save_results", "load_results"]


def project_root() -> str:
    """<...>/project_SRE_as_resource_for_generative_modeling"""
    here = os.path.dirname(os.path.abspath(__file__))          # python_codes/utils
    return os.path.abspath(os.path.join(here, "..", ".."))


def data_dir() -> str:
    d = os.path.join(project_root(), "data")
    os.makedirs(d, exist_ok=True)
    return d


def figures_dir() -> str:
    d = os.path.join(project_root(), "figures")
    os.makedirs(d, exist_ok=True)
    return d


def save_results(df: pd.DataFrame, tag: str) -> str:
    """Write <tag>.parquet and <tag>.csv into data/. Returns the parquet path."""
    base = os.path.join(data_dir(), tag)
    csv_path = base + ".csv"
    df.to_csv(csv_path, index=False)
    try:
        pq_path = base + ".parquet"
        df.to_parquet(pq_path, index=False)
        print(f"  saved {pq_path}")
    except Exception as e:                       # pyarrow missing -> csv is enough
        pq_path = csv_path
        print(f"  parquet unavailable ({type(e).__name__}); csv only")
    print(f"  saved {csv_path}   ({len(df)} rows)")
    return pq_path


def load_results(tag: str) -> pd.DataFrame:
    """Load by tag, preferring parquet. `tag` may omit the extension."""
    base = os.path.join(data_dir(), tag.replace(".parquet", "").replace(".csv", ""))
    if os.path.exists(base + ".parquet"):
        return pd.read_parquet(base + ".parquet")
    if os.path.exists(base + ".csv"):
        return pd.read_csv(base + ".csv")
    raise FileNotFoundError(f"no results for tag {tag!r} in {data_dir()}")
