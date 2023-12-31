########################################################################################################################
#
#
#   NEURAL ACTIVITY ANALYSIS
#
#
#   Description:    main file for analyzing neural activity data
#
#                   1) data from the data_dir directory is selected according to the specifications in the
#                      data_selection_dictionary and stored in the temp_data directory
#
#                   2) in the param_dic dictionary all necessary parameters for the analysis are defined
#
#                   3) the file is split into 4 main sections for different analysis methods (see Structure)
#
#
#   Author: Lars Bollmann
#
#   Created: 07/03/2019
#
#   Structure:
#
#       (a) COMPARISON ANALYSIS:    comparison between data before the rule switch (_2,_4 first part)
#                                   and data after the sleep (_6)
#
#       (b) TRANSITION ANALYSIS:    looks at the transition from rule A to rule B (_4 first part vs.
#                                   _4 second part after rule switch)
#
#       (c) STATE TRANSITION ANALYSIS:  evaluates the transition between states (population vectors)
#
#
#       (d) COLLECT RESULTS FOR MULTIPLE SESSIONS: collects results of previously computed results
#
#
########################################################################################################################

import pickle
import os
import numpy as np

# pre-selection
# ----------------------------------------------------------------------------------------------------------------------
from select_data import save_selected_data

# manifold methods
# ----------------------------------------------------------------------------------------------------------------------
from manifold_methods import Manifold
from manifold_methods import SingleManifold
from manifold_methods import ManifoldTransition
from manifold_methods import ManifoldCompare

# quantification methods
# ----------------------------------------------------------------------------------------------------------------------
from quantification_methods import BinDictionary
from quantification_methods import Analysis
from quantification_methods import StateTransitionAnalysis
from quantification_methods import ResultsMultipleSessions

# data description dictionary
data_description_dictionary = {
    "1": "RULE EAST",
    "2": "RULE WEST",
    "3": "RULE LIGHT"
}


if __name__ == '__main__':

    ####################################################################################################################
    #   DEFINE DATA INPUT
    ####################################################################################################################

    data_selection_dictionary = {
        "data_dir": "../02 Data",
        # define session name

        # trajectory: 1 --> 3, rule: light --> west
        # -----------------------------------------
        # "session_name": "mjc189-1905-0517",
        # "session_name": "mjc190-1307-0517",
        # "session_name": "mjc190-1407-0617",
        "session_name": "mjc189-2005-0517",
        # "session_name": "mjc196-1202-0517",
        # "session_name": "mjc196-1302-0416",
        # "session_name": "mjc200-3003-0521",

        # trajectory: 1 --> 2, rule: light --> east
        # -----------------------------------------
        # "session_name": "mjc189-1705-0622",
        # "session_name": "mjc190-1507-0517",
        # "session_name": "mjc190-1607-0515",
        # "session_name": "mjc196-1002-0617",

        # select cell type:
        # p1: pyramidal cells of the HPC, p2 - p3: pyramidal cells of the PFC ,b1: inter-neurons of HPC
        # b2 - b3: inter-neurons of HPC
        "cell_type_array": ["p2", "p3"],
        "start_arm": [1],
        "goal_arm": [3],
        # select rule type:
        # 1: east, 2: west, 3: light
        "rule_type": [2, 3],
        "error_trial": [1]
    }

    # derive file and session name
    file_name = data_selection_dictionary["session_name"]+"_ct_"+str(data_selection_dictionary["cell_type_array"])\
                + "_sa_" + str(data_selection_dictionary["start_arm"]) + "_ga_" + \
                str(data_selection_dictionary["goal_arm"]) + "_et_" + str(data_selection_dictionary["error_trial"])

    session_name = data_selection_dictionary["session_name"]

    # check if data exists as pickle --> if not, create pickled data
    if not os.path.isfile("temp_data/"+file_name):
        save_selected_data(data_selection_dictionary)

    ####################################################################################################################
    #   LOADING THE DATA
    ####################################################################################################################

    infile = open("temp_data/" + file_name, 'rb')
    data = pickle.load(infile)
    infile.close()

    # spike data
    # ------------------------------------------------------------------------------------------------------------------
    res_data_set_2 = data["2"]["res"]
    res_data_set_4 = data["4"]["res"]
    res_data_set_6 = data["6"]["res"]

    # location data
    # ------------------------------------------------------------------------------------------------------------------
    whl_lin_data_set_2 = data["2"]["whl_lin"]
    whl_lin_data_set_4 = data["4"]["whl_lin"]
    whl_lin_data_set_6 = data["6"]["whl_lin"]

    # info
    # ------------------------------------------------------------------------------------------------------------------
    new_rule_trial = data["4"]["info"]["new_rule_trial"]
    rule_order = data["4"]["info"]["rule_order"]

    ####################################################################################################################
    #   PARAMETERS
    ####################################################################################################################

    # dictionary for all other parameters
    param_dic = {}

    # DATA DESCRIPTION
    # ------------------------------------------------------------------------------------------------------------------
    param_dic["data_descr"] = [data_description_dictionary[str(rule_order[0])],
                               data_description_dictionary[str(rule_order[1])]]

    # session and file name
    param_dic["session_name"] = session_name
    param_dic["file_name"] = file_name

    # ANALYSIS PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------

    # binning with "temporal","spatial"
    param_dic["binning_method"] = "spatial"

    # z-score bins
    param_dic["z_score"] = True

    # interval for temporal binning in s
    param_dic["time_bin_size"] = 2

    # number of temporal bins
    param_dic["nr_time_bins"] = 12

    # interval for spatial binning in cm
    param_dic["spatial_bin_size"] = 50

    # spatial bins to exclude: e.g. first 2 (e.g 0-10cm and 10-20cm) and last (190-200cm) --> [2,-1]
    param_dic["spat_bins_excluded"] = []

    # filter high synchrony events/immobility using the speed in cm/s --> cm/2
    # without filter --> set to -np.inf
    param_dic["speed_filter"] = -np.inf
    # exclude population vectors with all zero values
    param_dic["zero_filter"] = False

    # define method for dimensionality reduction
    # "MDS" multi dimensional scaling
    # "PCA" principal component analysis
    # "TSNE"
    # "isomap"
    param_dic["dr_method"] = "MDS"

    # first parameter of method:
    # MDS --> p1: difference measure ["jaccard","cos","euclidean"]
    # PCA --> p1 does not exist --> ""
    param_dic["dr_method_p1"] = "euclidean"

    # second parameter of method:
    # MDS --> p2: number of components
    # PCA --> p2: number of components
    param_dic["dr_method_p2"] = 3

    # number of trials to compare
    param_dic["nr_of_trials"] = 21
    # selected trial
    param_dic["sel_trial"] = 3

    # QUANTITATIVE ANALYSIS
    # ------------------------------------------------------------------------------------------------------------------
    # statistical method: Kruskal-Wallis --> "KW", Mann-Whitney-U --> "MWU"
    param_dic["stats_method"] = "MWU"

    # alpha value
    param_dic["stats_alpha"] = 0.01

    # remapping characteristic: percent of total distance that is used to compute the number of needed cells
    # (default: 0.8 --> 80%)
    param_dic["percent_of_total_distance"] = 0.8
    # how many times is the order permuted to compute cell contribution
    param_dic["nr_order_permutations"] = 500

    # PLOTTING PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------

    # saving figure
    param_dic["save_plot"] = False
    # lines in scatter plot
    param_dic["lines"] = True
    # sort cells according to peak for visualization
    param_dic["sort_cells"] = True

    # saving directory for bin dictionaries
    param_dic["saving_dir_bin_dic"] = "temp_data/binned_dictionaries/"

    # saving directory & name for results dictionary
    param_dic["result_dictionary_name"] = "RESULT_DIC"
    param_dic["saving_dir_result_dictionary"] = "temp_data/"

    # length of spatial segment for plotting (track [200cm] will be divided into equal length segments)
    # set to 20: TODO --> adapt for different lengths
    param_dic["spat_seg_plotting"] = 20

    # saving figure file name
    param_dic["plot_file_name"] = "trans_analysis"+"_"+param_dic["dr_method"]+"_"+ param_dic["dr_method_p1"]+"_"\
                                  +str(param_dic["dr_method_p2"])+"D"+ param_dic["binning_method"]

    # TODO: automatically use maximum value from all data for axis limits
    # axis limit for plotting
    # jaccard: [-0.2,0.2]
    # cos: [-1,1]
    # 3D: [-0.5,0.5]
    # tSNE 2D: -50,50
    # PCA: -10,10
    # axis_lim = np.zeros(6)
    # axis_lim[0] = axis_lim[2]= axis_lim[4]= -50
    # axis_lim[1] = axis_lim[3] = axis_lim[5] =50
    # param_dic["axis_lim"] = axis_lim
    param_dic["axis_lim"] =[]

########################################################################################################################
# (a)  COMPARISON ANALYSIS (RULE A vs. RULE B)
########################################################################################################################

    # MANIFOLD ANALYSIS
    ####################################################################################################################

    # look at one rule across multiple trials
    # ------------------------------------------------------------------------------------------------------------------
    # new_analysis = SingleManifold(res_data_set_2, whl_lin_data_set_2, param_dic)
    # new_analysis.state_analysis()
    # new_analysis.plot_in_one_fig_color_position()

    # compare two rules using dimensionality reduction for the combined data (reduce to 2 or 3 dimensions)
    # ------------------------------------------------------------------------------------------------------------------

    # new_comparison = ManifoldCompare([res_data_set_2, res_data_set_6], [whl_lin_data_set_2, whl_lin_data_set_6],
    #                                   param_dic)
    # new_comparison.state_analysis()
    # new_comparison.plot_in_one_fig_color_position()

    # QUANTITATIVE ANALYSIS
    ####################################################################################################################

    # create binned dictionaries
    # ------------------------------------------------------------------------------------------------------------------

    dic = BinDictionary(param_dic)
    dic.create_spatial_bin_dictionary(res_data_set_2, whl_lin_data_set_2, param_dic["data_descr"][0])
    dic.create_spatial_bin_dictionary(res_data_set_6, whl_lin_data_set_6, param_dic["data_descr"][1])
    dic.create_spatial_bin_dictionaries_transition(res_data_set_4, whl_lin_data_set_4, new_rule_trial,
                                                   param_dic["data_descr"][0], param_dic["data_descr"][1])

    dic.combine_bin_dictionaries(param_dic["data_descr"][0], "SWITCH_"+param_dic["data_descr"][0]
                                 , param_dic["data_descr"][0]+"_2_4")

    # compare RULE A and RULE B
    # ------------------------------------------------------------------------------------------------------------------
    # new_compare = Analysis("RULE LIGHT_2_4", "RULE WEST", param_dic)
    # new_compare.plot_spatial_information()
    # new_compare.cross_cos_diff()
    # new_compare.cross_cos_diff_spat_trials()
    # new_compare.characterize_cells()
    # new_compare.cell_contribution_leave_one_out("cos")
    # new_compare.remove_cells([69])
    # new_compare.cell_contribution_subset_size("cos", 500)
    # new_compare.cell_contribution_cohen()
    # new_compare.estimate_remapped_cell_number_cosine(True)

########################################################################################################################
# (b)  TRANSITION ANALYSIS (RULE A --> RULE B)
########################################################################################################################

    # MANIFOLD ANALYSIS
    ####################################################################################################################

    # new_analysis = ManifoldTransition(res_data_set_4, whl_lin_data_set_4, param_dic)
    # new_analysis.state_analysis()
    # new_analysis.plot_in_one_fig_color_position()
    # new_analysis.plot_in_one_fig(new_rule_trial)

    # QUANTITATIVE ANALYSIS
    ####################################################################################################################

    # new_transition = Analysis("RULE LIGHT_2_4", "SWITCH_RULE WEST", param_dic)
    # new_transition.cross_cos_diff()
    # new_transition.cross_cos_diff_spat_trials()
    # new_transition.characterize_cells()
    # new_transition.estimate_remapped_cell_number_cosine(True)
    # new_transition.remove_cells([69])
    # new_transition.remove_cells(np.arange(0, 60))


#######################################################################################################################
# (c)  STATE TRANSITION ANALYSIS
########################################################################################################################

    # MANIFOLD ANALYSIS
    ####################################################################################################################

    # looking at one rule
    # ------------------------------------------------------------------------------------------------------------------
    # state_transition_analysis = SingleManifold(res_data_set_1, whl_lin_data_set1, param_dic)
    # state_transition_analysis.state_transition_analysis()

    # comparing two rules
    # ------------------------------------------------------------------------------------------------------------------
    # new_comparison = ManifoldCompare([res_data_set_1, res_data_set_2],[whl_lin_data_set1,
    # whl_lin_data_set2], param_dic)
    # new_comparison.state_transition_analysis()

    # QUANTITATIVE ANALYSIS
    ####################################################################################################################

    # new_state_transition = StateTransitionAnalysis([res_data_set_2, res_data_set_6], [whl_lin_data_set_2,
    #                                                                                   whl_lin_data_set_6], param_dic)

    # euclidean distance between subsequent steps
    # ------------------------------------------------------------------------------------------------------------------
    # new_state_transition.compare_distance("cos")
    # new_state_transition.compare_distance("L1")
    # new_state_transition.compare_operations()

#######################################################################################################################
# (d)  COLLECT RESULTS FOR MULTIPLE SESSIONS
#######################################################################################################################

    # new_results = ResultsMultipleSessions(param_dic)
    # new_results.collect_and_save_data()
    # new_results.read_results()
    # new_results.plot_results()
    # new_results.summarize("COMPARISON", False)
    # new_results.summarize("TRANSITION", False)

    # dic = BinDictionary(param_dic)
    # dic.create_temporal_bin_dictionary(res_data_set_2, whl_lin_data_set_2, param_dic["data_descr"][0], param_dic["nr_time_bins"])
    # dic.create_temporal_bin_dictionary(res_data_set_6, whl_lin_data_set_6, param_dic["data_descr"][1], param_dic["nr_time_bins"])
    # dic.create_temporal_bin_dictionaries_transition(res_data_set_4, whl_lin_data_set_4, new_rule_trial,
    #                                             param_dic["data_descr"][0], param_dic["data_descr"][1], param_dic["nr_time_bins"])

    new_compare = Analysis("RULE LIGHT", "SWITCH_RULE LIGHT", param_dic, "SWITCH_RULE WEST", "RULE WEST")
    # new_compare.gradual_transition("cos")
    # new_compare.cross_cos_diff_temp_trials()
    # new_compare.plot_spatial_information()
    new_compare.cross_cos_diff_trials_all_sessions()
