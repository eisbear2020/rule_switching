########################################################################################################################
#
#   NEURAL ACTIVITY ANALYSIS
#
#   Description:
#
#   Author: Lars Bollmann
#
#   Created: 07/03/2019
#
#   Structure:
#
#
#
#
#
#
#
########################################################################################################################

import pickle
from analysis_methods import dimRed2DCompare
from analysis_methods import dimRed2D
from analysis_methods import dimRed3D
from analysis_methods import dimRedCombined
from analysis_methods import StateTransitionAnalysis
from analysis_methods import dimRed2DConc

if __name__ == '__main__':

########################################################################################################################
#   PARAMETERS
########################################################################################################################

    # FILE PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    # HPC data, rule 3: light and rule 2: west
    file_name_rule_3 = "res_mjc189-1905-0517_2_ct_['p1']_sa_[1]_ga_[3]_rt_[3]_et_[1]"
    file_name_rule_2 = "res_mjc189-1905-0517_6_ct_['p1']_sa_[1]_ga_[3]_rt_[2]_et_[1]"
    file_name_rule_switch = "res_mjc189-1905-0517_4_ct_['p1']_sa_[1]_ga_[3]_rt_[2, 3]_et_[1]"

    # dictionary for all other parameters
    param_dic = {}

    # DATA DESCRIPTION
    # ------------------------------------------------------------------------------------------------------------------

    # description of data
    param_dic["data_descr"] = ["RULE LIGHT", "RULE WEST"]

    # ANALYSIS PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------

    # interval for binning in s
    param_dic["bin_interval"] = 0.1

    # define method for dimensionality reduction
    param_dic["dr_method"] = "MDS" # multi dimensional scaling

    # first parameter of method:
    # MDS --> p1: difference measure ["jaccard","cos"]
    param_dic["dr_method_p1"] = "cos"

    # second parameter of method:
    # MDS --> p2: number of components
    param_dic["dr_method_p2"] = 3


    # number of trials to compare
    param_dic["nr_of_trials"] = 6
    # first trial to start analysis with
    param_dic["first_trial"] = 0
    # selected trial
    param_dic["sel_trial"] = 1

    # PLOTTING PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------

    # plotting options
    param_dic["lines"] = True

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
    infile_3 = open("temp_data/"+file_name_rule_3, 'rb')
    data_rule_3 = pickle.load(infile_3)
    infile_3.close()
    infile_2 = open("temp_data/"+file_name_rule_2, 'rb')
    data_rule_2 = pickle.load(infile_2)
    infile_2.close()
    infile_23 = open("temp_data/"+file_name_rule_switch, 'rb')
    data_rule_23 = pickle.load(infile_23)
    infile_23.close()

########################################################################################################################
#   DYNAMIC ANALYSIS
########################################################################################################################

    # rule switching: transformation for each data set separately
    # ------------------------------------------------------------------------------------------------------------------
    # if param_dic["dr_method_p2"] == 2:
    #     dimRed2D(data_rule_23, param_dic)
    # elif param_dic["dr_method_p2"] == 3:
    #     dimRed3D([data_rule_23], param_dic)

    # rule switching: transformation (reduction in dim.) using concatenated data
    # ------------------------------------------------------------------------------------------------------------------
    #dimRed2DConc(data_rule_23, param_dic)



    # compare two rules using dimensionality reduction for both sets separately (reduce to 2 or 3 dimensions)
    # ------------------------------------------------------------------------------------------------------------------
    # if param_dic["dr_method_p2"] == 2:
    #     dimRed2DCompare([data_rule_3, data_rule_2], param_dic)
    # elif param_dic["dr_method_p2"] == 3:
    #     dimRed3D([data_rule_3, data_rule_2], param_dic)

    # compare two rules using dimensionality for the combined data (reduce to 2 or 3 dimensions)
    # ------------------------------------------------------------------------------------------------------------------
    #dimRedCombined([data_rule_3, data_rule_2], param_dic)


########################################################################################################################
#   TRANSITION ANALYSIS
########################################################################################################################

    # using difference vectors
    # ------------------------------------------------------------------------------------------------------------------
    #StateTransitionAnalysis([data_rule_3], param_dic)





