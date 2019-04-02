# Python scripts to analyze neural data

# File structure

## main.py
main file for analysis. Calls methods from analysis_methods.py to analyse data 

* manifold transition (e.g. RULE A --> RULE B)
* manifold comparison (e.g RULE A vs. RULE B)
* state transition analysis
    
## manifold_methods.py
contains different analysis approaches for manifolds and uses functions from comp_functions.py and plotting_functions.py.

* Manifold: Base class containing basic attributes and methods
* singleManifold: Methods/attributes to analyze manifold for one condition (e.g. RULE A)
* ManifoldTransition: evaluates the manifold change e.g during rule switch
* ManifoldCompare:  compares results of two different conditions using dimensionality reduction and transforming each trial separately

## comp_functions.py
contains functions that compute parts of a thorough analysis:

* computing activity map
* multi dimensional scaling
* population vector difference matrix

## select_data.py
importing/filtering/saving of data. Uses helper functions from filter_functions.py
## filter_functions.py
helper functions for selecting data
