"""
get_data_figure_2.py -- data for Manuscript Figure 2 (source: run_sweep.py (nt, deepl)).
"""
from _gen_common import run
run("run_sweep.py", SWEEP="nt", GRID_N=10)
run("run_sweep.py", SWEEP="deepl", GRID_N="8,10,12")
