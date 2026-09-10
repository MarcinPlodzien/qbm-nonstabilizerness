"""
get_table_1.py -- reproduce Manuscript Table I: 2D targets, four entanglers, independent curriculum chains (N=10, L=12).
Run: python get_table_1.py -> LaTeX tabular rows on stdout
"""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import table_twod_chains
table_twod_chains.main()
