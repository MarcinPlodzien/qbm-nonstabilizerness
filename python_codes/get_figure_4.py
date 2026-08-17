"""
get_figure_4.py -- reproduce Manuscript Figure 4: kicked-Ising (Clifford-point) entangler on six 2D targets (N=10, L=12).
Run: python get_figure_4.py -> figures/fig_twod_combined.pdf
"""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import make_fig_twod_combined
make_fig_twod_combined.make_fig_twod_combined()
