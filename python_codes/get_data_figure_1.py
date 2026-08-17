"""
get_data_figure_1.py -- data for Manuscript Figure 1 (source: run_sweep.py (decoupling, deepl)).
"""
from _gen_common import run
run("run_sweep.py", SWEEP="decoupling", GRID_N=10)
run("run_sweep.py", SWEEP="deepl", GRID_N=10)
