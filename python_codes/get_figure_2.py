"""
get_figure_2.py -- reproduce Manuscript Figure 2: T-doping sweep of KL divergence vs probe-state SRE at fixed entangling power, and trained-state M2 vs N.
Run: python get_figure_2.py -> figures/fig_nt.pdf
"""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import fig_nt
fig_nt.main()
