"""
get_figure_3.py -- reproduce Manuscript Figure 3: SK-Gibbs entangler comparison vs inverse temperature (N=10, L=12).
Run: python get_figure_3.py -> figures/fig_gibbs_kld_N10_SK_main.pdf
"""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import fig_gibbs_kld
fig_gibbs_kld.main()
