"""
get_data_figure_3.py -- data for Manuscript Figure 3 (source: run_gibbs.py).
"""
from _gen_common import run
import numpy as np
betas=[round(x,2) for x in np.arange(0,4.001,0.2)]+[round(x,2) for x in np.arange(4.4,8.001,0.4)]
run("run_gibbs.py", GRID_N=10, GIBBS_L=12, SG_MODEL="SK", N_DIS=10, N_REAL=10, ANSATZ="main", BETA_LIST=",".join(str(b) for b in betas))
run("compute_gibbs_randphase.py")   # random-phase SRE references for panel (b)
