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
#       - dynamic analysis
#       - transition analysis
#
########################################################################################################################

import pickle
import numpy as np
from analysis_methods import state_transition_analysis
from analysis_methods import SingleManifold
from analysis_methods import ManifoldTransition
from analysis_methods import ManifoldCompare
from comp_functions import calc_loc_and_speed



if __name__ == '__main__':

########################################################################################################################
#   PARAMETERS
########################################################################################################################

    # FILE PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    # HPC data, rule 3: light and rule 2: west
    file_rule_3 = "mjc189-1905-0517_2_ct_['p1']_sa_[1]_ga_[3]_rt_[3]_et_[1]"
    file_rule_2 = "mjc189-1905-0517_6_ct_['p1']_sa_[1]_ga_[3]_rt_[2]_et_[1]"
    file_rule_switch = "mjc189-1905-0517_4_ct_['p1']_sa_[1]_ga_[3]_rt_[2, 3]_et_[1]"

    # dictionary for all other parameters
    param_dic = {}

    # DATA DESCRIPTION
    # ------------------------------------------------------------------------------------------------------------------

    # description of data
    param_dic["data_descr"] = ["RULE LIGHT", "RULE WEST"]

    # ANALYSIS PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------

    # binning with "temporal","spatial"
    param_dic["binning_method"] = "spatial"

    # interval for temporal binning in s
    param_dic["time_bin_size"] = 0.1

    # interval for spatial binning in cm
    param_dic["spatial_bin_size"] = 10

    # spatial bins to exclude: e.g. first 2 (e.g 0-10cm and 10-20cm) and last (190-200cm) --> [2,-1]
    param_dic["spat_bins_excluded"] = [2,-1]

    # filter high synchrony events/immobility using the speed in cm/s --> cm/2
    # without filter --> set to []
    param_dic["speed_filter"] = 5
    # exclude population vectors with all zero values
    param_dic["zero_filter"] = True

    # define method for dimensionality reduction
    # "MDS" multi dimensional scaling
    # "PCA" principal component analysis
    # "TSNE"
    param_dic["dr_method"] = "MDS"

    # first parameter of method:
    # MDS --> p1: difference measure ["jaccard","cos"]
    # PCA --> p1 does not exist --> ""
    param_dic["dr_method_p1"] = "cos"

    # second parameter of method:
    # MDS --> p2: number of components
    # PCA --> p2: number of components
    param_dic["dr_method_p2"] = 2


    # number of trials to compare
    param_dic["nr_of_trials"] = 6
    # selected trial
    param_dic["sel_trial"] = 2

    # PLOTTING PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------

    # saving figure
    param_dic["save_plot"] = False

    # lines in scatter plot
    param_dic["lines"] = True


    # length of spatial segment for plotting (track [200cm] will be divided into equal length segments)
    # set to 20: TODO --> adapt for different lengths
    param_dic["spat_seg_plotting"] = 20

    # saving figure file name
    param_dic["plot_file_name"] = "man_transition_mds_cos_3D_rulesep"+"_"+param_dic["dr_method"]+"_"+ param_dic["dr_method_p1"]+"_"\
                                  +str(param_dic["dr_method_p2"])+"D"

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
#   LOADING THE DATA
########################################################################################################################

    # spike data
    # ------------------------------------------------------------------------------------------------------------------
    infile_3 = open("temp_data/res_"+file_rule_3, 'rb')
    res_rule_3 = pickle.load(infile_3)
    infile_3.close()
    infile_2 = open("temp_data/res_"+file_rule_2, 'rb')
    res_rule_2 = pickle.load(infile_2)
    infile_2.close()
    infile_23 = open("temp_data/res_"+file_rule_switch, 'rb')
    res_rule_23 = pickle.load(infile_23)
    infile_23.close()

    # location data
    # ------------------------------------------------------------------------------------------------------------------
    infile_3 = open("temp_data/whl_lin_" + file_rule_3, 'rb')
    whl_lin_rule_3 = pickle.load(infile_3)
    infile_3.close()
    infile_2 = open("temp_data/whl_lin_" + file_rule_2, 'rb')
    whl_lin_rule_2 = pickle.load(infile_2)
    infile_2.close()
    infile_23 = open("temp_data/whl_lin_" + file_rule_switch, 'rb')
    whl_lin_rule_23 = pickle.load(infile_23)
    infile_23.close()

########################################################################################################################
#   MANIFOLD TRANSITION ANALYSIS
########################################################################################################################

    # rule switching: transformation for each data set separately
    # ------------------------------------------------------------------------------------------------------------------
    #manifold_transition(res_rule_23, param_dic)

    # rule switching: transformation (reduction in dim.) using concatenated data
    # ------------------------------------------------------------------------------------------------------------------
    #manifold_transition_conc(res_rule_23, whl_lin_rule_23, param_dic)

    # trial with new rule
    # new_rule_trial = 7
    # new_analysis = ManifoldTransition(res_rule_23, whl_lin_rule_23, param_dic)
    # new_analysis.concatenated_data()
    # new_analysis.plot_in_one_fig(new_rule_trial)

########################################################################################################################
#   MANIFOLD COMPARISON
########################################################################################################################

    # look at one rule across multiple trials
    # ------------------------------------------------------------------------------------------------------------------
    # new_analysis = SingleManifold(res_rule_2, whl_lin_rule_2, param_dic)
    # new_analysis.concatenated_data()
    # new_analysis.plot_in_one_fig()

    # compare two rules using dimensionality reduction for both sets separately (reduce to 2 or 3 dimensions)
    # ------------------------------------------------------------------------------------------------------------------
    #manifold_compare([res_rule_3, res_rule_2], param_dic)

    # compare two rules using dimensionality for the combined data (reduce to 2 or 3 dimensions)
    # ------------------------------------------------------------------------------------------------------------------
    #manifold_compare_conc([res_rule_3, res_rule_2],[whl_lin_rule_3, whl_lin_rule_2], param_dic)

    # new_comparison = ManifoldCompare([res_rule_3, res_rule_2],[whl_lin_rule_3, whl_lin_rule_2], param_dic)
    # new_comparison.all_trials()


########################################################################################################################
#   STATE TRANSITION ANALYSIS
########################################################################################################################

    # using difference vectors
    # ------------------------------------------------------------------------------------------------------------------
    #state_transition_analysis([res_rule_23], whl_lin_rule_23, param_dic)

