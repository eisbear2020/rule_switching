########################################################################################################################
#
#   NEURAL ACTIVITY ANALYSIS
#
#   Description: main file for analysing neural activity data
#
#   Author: Lars Bollmann
#
#   Created: 07/03/2019
#
#   Structure:
#
#       - Manifold analysis
#           - transition analysis (RULE A --> RULE B)
#           - comparison analysis (RULE A vs. RULE B)
#           - state transition analysis (difference vectors between population states)
#
#   TODO: think about parameter dictionary --> maybe split into two: one for manifold and one for quantitative
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
from manifold_methods import SingleManifold
from manifold_methods import ManifoldTransition
from manifold_methods import ManifoldCompare

# quantification methods
# ----------------------------------------------------------------------------------------------------------------------
from quantification_methods import BinDictionary
from quantification_methods import Analysis
from quantification_methods import StateTransitionAnalysis

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
        "session_name": "mjc189-1905-0517",
        # select cell type:
        # p1: pyramidal cells of the HPC, p2 - p3: pyramidal cells of the PFC ,b1: inter-neurons of HPC
        # b2 - b3: inter-neurons of HPC
        "cell_type_array": ["p1"],
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

    # binning with "temporal","spatial" --> only for manifold analysis
    param_dic["binning_method"] = "spatial"

    # interval for temporal binning in s
    param_dic["time_bin_size"] = 0.1

    # interval for spatial binning in cm
    param_dic["spatial_bin_size"] = 10

    # spatial bins to exclude: e.g. first 2 (e.g 0-10cm and 10-20cm) and last (190-200cm) --> [2,-1]
    param_dic["spat_bins_excluded"] = [2, -1]

    # filter high synchrony events/immobility using the speed in cm/s --> cm/2
    # without filter --> set to []
    param_dic["speed_filter"] = 5
    # exclude population vectors with all zero values
    param_dic["zero_filter"] = False

    # define method for dimensionality reduction
    # "MDS" multi dimensional scaling
    # "PCA" principal component analysis
    # "TSNE"
    param_dic["dr_method"] = "MDS"

    # first parameter of method:
    # MDS --> p1: difference measure ["jaccard","cos","euclidean"]
    # PCA --> p1 does not exist --> ""
    param_dic["dr_method_p1"] = "cos"

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
#   COMPARISON ANALYSIS
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
    # dic = BinDictionary(param_dic)
    # dic.create_spatial_bin_dictionary(res_data_set_2, whl_lin_data_set_2, param_dic["data_descr"][0])
    # dic.create_spatial_bin_dictionary(res_data_set_6, whl_lin_data_set_6, param_dic["data_descr"][1])
    # dic.combine_bin_dictionaries("RULE LIGHT", "SWITCH_RULE LIGHT", "RULE LIGHT_2_4")

    # compare RULE A and RULE B
    # ------------------------------------------------------------------------------------------------------------------
    # new_compare = Analysis("RULE LIGHT", "RULE WEST", param_dic)
    # new_compare.plot_spatial_information()
    # new_compare.cross_cos_diff()
    # new_compare.characterize_cells()
    # new_compare.cell_contribution()
    # new_compare.cross_cos_diff_spat_trials()
    # new_compare.remove_cells([46,69])

########################################################################################################################
#   TRANSITION ANALYSIS (RULE A --> RULE B)
########################################################################################################################

    # MANIFOLD ANALYSIS
    ####################################################################################################################

    # new_analysis = ManifoldTransition(res_data_set_4, whl_lin_data_set_4, param_dic)
    # new_analysis.state_analysis()
    # new_analysis.plot_in_one_fig(new_rule_trial)
    # new_analysis.plot_in_one_fig_color_position()

    # QUANTITATIVE ANALYSIS
    ####################################################################################################################

    # create binned dictionaries
    # ------------------------------------------------------------------------------------------------------------------
    # dic = BinDictionary(param_dic)
    # dic.create_spatial_bin_dictionaries_transition(res_data_set_4, whl_lin_data_set_4, new_rule_trial,
    #                                                param_dic["data_descr"][0], param_dic["data_descr"][1])

    # new_transition = Analysis("SWITCH_RULE LIGHT", "SWITCH_RULE WEST", param_dic)
    # new_transition.cross_cos_diff()
    # new_transition.cross_cos_diff_spat_trials()
    # new_transition.characterize_cells()
    # new_transition.cell_contribution()
    # new_transition.remove_cells([46,69])
    # new_transition.remove_cells(np.arange(0, 60))


#######################################################################################################################
#   STATE TRANSITION ANALYSIS
########################################################################################################################

    # MANIFOLD ANALYSIS
    ####################################################################################################################

    # looking at one rule
    # ------------------------------------------------------------------------------------------------------------------
    # state_transition_analysis = SingleManifold(res_data_set_1, whl_lin_data_set1, param_dic)
    # state_transition_analysis.state_transition_analysis()

    # comparing two rules
    # ------------------------------------------------------------------------------------------------------------------
    # new_comparison = ManifoldCompare([res_data_set_1, res_data_set_2],[whl_lin_data_set1, whl_lin_data_set2], param_dic)
    # new_comparison.state_transition_analysis()

    # QUANTITATIVE ANALYSIS
    ####################################################################################################################

    new_state_transition = StateTransitionAnalysis([res_data_set_2, res_data_set_6], [whl_lin_data_set_2,
                                                                                      whl_lin_data_set_6], param_dic)
 
    # euclidean distance between subsequent steps
    # ------------------------------------------------------------------------------------------------------------------
    new_state_transition.compare_distance()
    # new_state_transition.compare_angle()
    # new_state_transition.compare_operations()
