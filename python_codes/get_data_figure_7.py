"""
get_data_figure_7.py -- data for Manuscript Figure 7 (source: run_twod_chains.py).
Same data set as get_data_table_1.py; the operator entanglement is computed from the stored entangler gates.
"""
import runpy, os
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "get_data_table_1.py"), run_name="__main__")
