"""
get_figure_1.py -- reproduce Manuscript Figure 1: 1D five-peak: target with trained output, and KL divergence vs depth L (N=10).
Run: python get_figure_1.py -> figures/fig_combined.pdf
"""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import fig_combined
fig_combined.main()
