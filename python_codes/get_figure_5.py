"""
get_figure_5.py -- reproduce Manuscript Figure 5: training convergence (KLD vs epoch) for the six 2D targets.
Run: python get_figure_5.py -> figures/fig_twod_convergence.pdf
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_fig_twod_combined
make_fig_twod_combined.make_fig_twod_convergence()
