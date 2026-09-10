"""
get_figure_6.py -- reproduce Manuscript Figure 6 (Appendix B): KLD vs depth for four entanglers on the six 2D
targets, three independent depth-curriculum chains each (N=10).
Run: python get_figure_6.py -> figures/fig_twod_chains.pdf
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_twod_chains
fig_twod_chains.main()
