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
from analysis_methods import manifold_transition
from analysis_methods import manifold_transition_conc
from analysis_methods import manifold_compare
from analysis_methods import manifold_compare_conc
from analysis_methods import state_transition_analysis
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

    # interval for temporal binning in s
    param_dic["time_bin_size"] = 0.1

    # filter high synchrony events/immobility using the speed in cm/s --> cm/2
    # without filter --> set to []
    param_dic["speed_filter"] = 5

    # exclude population vectors with all zero values
    param_dic["zero_filter"] = True

    # define method for dimensionality reduction
    # "MDS" multi dimensional scaling
    # "PCA" principal component analysis
    param_dic["dr_method"] = "MDS"

    # first parameter of method:
    # MDS --> p1: difference measure ["jaccard","cos"]
    # PCA --> p1 does not exist --> ""
    param_dic["dr_method_p1"] = "jaccard"

    # second parameter of method:
    # MDS --> p2: number of components
    # PCA --> p2: number of components
    param_dic["dr_method_p2"] = 3


    # number of trials to compare
    param_dic["nr_of_trials"] = 15
    # selected trial
    param_dic["sel_trial"] = 2

    # PLOTTING PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------

    # lines in scatter plot
    param_dic["lines"] = True

    # number of columns for subplots
    param_dic["c_p"] = 3

    # length of spatial segment for plotting (track [200cm] will be divided into equal length segments)
    # set to 20: TODO --> adapt for different lengths
    param_dic["spat_seg_plotting"] = 20

    # axis limit for plotting
    # jaccard: [-0.2,0.2]
    # cos: [-1,1]
    # 3D: [-0.5,0.5]
    # axis_lim = np.zeros(6)
    # axis_lim[0] = axis_lim[2]= axis_lim[4]= -0.5
    # axis_lim[1] = axis_lim[3] = axis_lim[5] =0.5
    # param_dic["axis_lim"] = axis_lim
    param_dic["axis_lim"] =[]

    plotting = False

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
#   DYNAMIC ANALYSIS
########################################################################################################################

    # rule switching: transformation for each data set separately
    # ------------------------------------------------------------------------------------------------------------------
    #manifold_transition(res_rule_23, param_dic)

    # rule switching: transformation (reduction in dim.) using concatenated data
    # ------------------------------------------------------------------------------------------------------------------
    manifold_transition_conc(res_rule_23, whl_lin_rule_23, param_dic)



    # compare two rules using dimensionality reduction for both sets separately (reduce to 2 or 3 dimensions)
    # ------------------------------------------------------------------------------------------------------------------
    #manifold_compare([res_rule_3, res_rule_2], param_dic)

    # compare two rules using dimensionality for the combined data (reduce to 2 or 3 dimensions)
    # ------------------------------------------------------------------------------------------------------------------
    #manifold_compare_conc([res_rule_3, res_rule_2],[whl_lin_rule_3, whl_lin_rule_2], param_dic)


########################################################################################################################
#   TRANSITION ANALYSIS
########################################################################################################################

    # using difference vectors
    # ------------------------------------------------------------------------------------------------------------------
    #state_transition_analysis([data_rule_3,data_rule_2], param_dic)



