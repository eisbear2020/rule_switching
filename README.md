# Python scripts to analyze neural data

# File structure

## main.py
main file for analysis.

* data from the data_dir directory is selected according to the specifications in the 
data_selection_dictionary and stored in the temp_data directory
* in the param_dic dictionary all necessary parameters for the analysis are defined
* the file is split into 4 main sections for different analysis methods:

    * COMPARISON ANALYSIS:    comparison between data before the rule switch (_2,_4 first part)
and data after the sleep (_6)
    * TRANSITION ANALYSIS:    looks at the transition from rule A to rule B (_4 first part vs.
                                   _4 second part after rule switch)

    * STATE TRANSITION ANALYSIS:  evaluates the transition between states (population vectors)
    * COLLECT RESULTS FOR MULTIPLE SESSIONS: collects results of previously computed results
    
## manifold_methods.py
contains different analysis approaches for manifolds and uses functions from comp_functions.py and plotting_functions.py.

* Manifold: Base class containing basic attributes and methods

    * reduce_dimension: reduces dimension using defined method

    * plot_in_one_fig: plots results as scatter plot separating either trials (default) or rules

* singleManifold: Methods/attributes to analyze manifold for one condition (e.g. RULE A)

    *  concatenated_data:  using data from multiple trials for transformation (dim. reduction)
                                               and separating data afterwards

    * state_transition: analyzes the state transitions using difference vectors between two population
                                   states

* ManifoldTransition: evaluates the manifold change e.g during rule switch

    * separate_data_time_bins: using time bins, transforming every trial separately

* ManifoldCompare:  compares results of two different conditions using dimensionality reduction and transforming
                           each trial separately

     *  all_trials: compares two conditions (e.g. RULE A vs. RULE B) using all available trials from the data

     *  selected_trials: compares two conditions (e.g. RULE A vs. RULE B) using one trial for each condition

## quantification_methods.py
contains different analysis approaches to quantify the results from the 
manifold analysis

* BinDictionary: class for creation & modification of binned dictionaries. One dictionary contains one entry
                        per bin. One bin contains all population vectors as column vectors of different trials

     * check_and_create_dic: if dictionary does not exist yet --> create a new one

     * create_spatial_bin_dictionaries_transition: separates one data set into two binned dictionaries
                     depending on the new rule trial create "activity matrices" consisting of population vectors for
                     each rule

     * create_spatial_bin_dictionary: create "activity matrices" consisting of population vectors for
                     each rule

     * combine_bin_dictionaries: takes two dictionaries and combines them in one


* Analysis: class that analysis the data contained in binned dictionaries

     (1) Methods to analyze single cells:

     * cell_avg_rate_map: returns average rate map combining data from both dictionaries for each cell
                     and spatial bin

     * cell_rule_diff: calculate change in average firing rate and standard error of the mean for each
                     cell and bin between rules using cohens d: (avg1-avg2)/pooled std

     * plot_spatial_information: plots spatial information by sorting cells by peak firing rate

     * characterize_cells: performs different analysis steps to identify contribution/fingerprint of
                     single cells

     (2) Methods to analyze different rules:

     * cross_cos_diff:   calculates the pair-wise cos difference within each set and across both sets
           and compares the two distributions using the defined statistical method param_dic["stats_method"]
           using the data from all trials

           TODO: bonferroni corrections because spatial bins are not independent

     * cross_cos_diff_spat_trials: calculates within vs. across using all trials from both dictionaries
                     separating the data of single trials

     (3) Methods to analyze remapping characteristics (cell contribution):

     * remove_cells: cross_cos_diff and cross_cos_diff_trials with modified dictionaries leaving out
                     defined cells

     * leave_n_out_random: leaves different number of random cells out to estimate
                     contribution to the difference that is calculated as the average over trials. Changing variable
                     is the size of the subset (nr. of cells in subset)

     * leave_one_out: leaves out one cell after the other to estimate contribution
                     of single cells to the difference between rules (difference is calculated as the average over
                     trials)

       Methods to summarize results:

       * cell_contribution_cohen:  checks how many cells contribute to the difference by looking at how
                     many cells remap significantly using effect size/cohen's d

       * cell_contribution_leave_one_out(self, distance_measure): check how many cells contribute how much
                     to the difference between two conditions (e.g. RULES). Calls the leave_out_out method, leaves out
                     one cell after the other and sorts them according to contribution cumulative contributions are
                     plotted as a function of added cells

       * cell_contribution_subset_size: checks how many cells contribute how much to the
                     difference between two conditions (e.g. RULES) calls leave_n_out_random_average_over_trials
                     --> looks at different subsets of all cells and calculates difference between rules based on the
                     subset of cells. The variable is the size of the subset (nr. of cells in subset)

       * estimate_remapped_cell_number_cosine: check how many cells contribute how much to the difference
                     between two conditions (e.g. RULES) by "undoing" the occurred remapping and looking at how much
                     the overal difference reduces. Cells are then sorted by their impact (how much the reduce the
                     difference when they are removed). Nr. of cells to achieve 80% of the total difference is computed
                     --> many cells: more global remapping of many cells.
                     --> few cells: only some cells remap and cause the difference

              TODO: instead of "undoing" the remapping --> simulate firing rate of these neurons with
                           Poisson model


* StateTransitionAnalysis: class to analyze state transition between population vectors

    * filter_cells: filter cells that do not show any activity at all

    * distance: calculates distance between subsequent population vectors

    * angle: calculates angles between subsequent transitions

    * operations: calculates number of zeros (no change), +1 (activation) and -1 (inhibition) in
                     population difference vectors

        Methods to compare two rules:

    * compare_distance: compares distance between subsequent population vectors for two different data
                     sets (e.g. two different rules)

    * compare_operations: compares operations (silencing, activation, unchanged) between states for
                     two data sets (e.g. two different rules)

    * compare_angle: compares angles between subsequent population vectors for two data sets
                     (e.g. two rules)


* ResultsMultipleSessions: class that combines and saves all specified results to look at results
                                  from multiple sessions

    * check_and_create_dic: if dictionary does not exist yet --> create a new one

    * collect_and_save_data: collects and saves data

    * read_results: prints identifiers of all data that was collected

    * plot_results: plots results of all sessions separately

    * summarize: gets all results for type (either "COMPARISON" or "TRANSITION") and plots them in one 
                     plot. Selection of certain session can be defined.
                     
## comp_functions.py
contains functions that compute parts of a thorough analysis:

* computing activity map
* multi dimensional scaling
* population vector difference matrix

## select_data.py
importing/filtering/saving of data. Uses helper functions from filter_functions.py

# Results

## manifold_analysis.md

Contains all results of the qualitative manifold analysis.

## quantitative_analysis.md

Contains results of quantitative analysis.

## analysis_mjc189-1905-0517_sa_1_ga_3.md

Contains results for one specific session with specific constrains

## all_sessions.md

Summarizes results accross sessions.

## rm.py

Various methods to process data implemented by Michele Nadin.

## test.py

Experiments concerning distance measures and simple simulations to get insights concerning
number of remapped cells.