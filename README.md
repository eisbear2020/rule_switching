# Python scripts to analyse neural data

# File structure

* main.py: main file for analysis. Calls methods from analysis_methods.py to analyse data
    * dynamic analysis
    * transition analysis
* analysis_methods.py: contains different analysis approaches and uses functions from comp_functions.py.
    * transition of manifold during rule switch
    * comparison of manifolds with different rules
    * analysing state transitions using difference vectors
* comp_functions.py: contains functions that compute parts of a thorough analysis:
    * computing activity map
    * multi dimensional scaling
    * population vector difference matrix
* select_data.py: importing/filtering/saving of data. Uses helper functions from filter_functions.py
* filter_functions.py: helper functions for selecting data
