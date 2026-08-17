"""
get_data_figure_5.py -- data for Manuscript Figure 5 (source: run_twod_curriculum.py; same runs as Figure 4).
"""
from _gen_common import run
run("run_twod_curriculum.py", GRID_N=10, CURRIC_TARGETS="mm2d,rings,moons,stripes,cross,spiral")
