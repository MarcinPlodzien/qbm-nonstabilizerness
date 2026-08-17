"""
get_data_figure_4.py -- data for Manuscript Figure 4 (source: run_twod_curriculum.py).
"""
from _gen_common import run
run("run_twod_curriculum.py", GRID_N=10, CURRIC_TARGETS="mm2d,rings,moons,stripes,cross,spiral")
