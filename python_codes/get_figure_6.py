"""
get_figure_6.py -- reproduce Manuscript Figure 6 (Appendix B): single-layer entangling power S_vN vs N,
for a single application of each fixed entangler U_S|+>.
Run: python get_figure_6.py -> figures/fig_svn_scaling.pdf
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_svn_scaling
fig_svn_scaling.main()
