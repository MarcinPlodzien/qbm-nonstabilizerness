"""
get_figure_7.py -- reproduce Manuscript Figure 7 (Appendix B): operator entanglement of each fixed entangler
across every bipartition, and register-cut entropy per layer of the trained circuit (N=10).
Run: python get_figure_7.py -> figures/fig_twod_openent.pdf
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_twod_openent
fig_twod_openent.main()
