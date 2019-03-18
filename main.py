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
import scipy.stats as sp
import numpy as np
import matplotlib.pyplot as plt
from helper_func import getActivityMat
from helper_func import plotActMat
#import seaborn as sns; sns.set()
from helper_func import multiDimScaling
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from helper_func import plot2DscatterLines
from helper_func import dimRed2DCompare
from helper_func import dimRed2D
from helper_func import dimRed3DCompare
from helper_func import dimRedCombined
from collections import OrderedDict
from helper_func import StateTransitionAnalysis

if __name__ == '__main__':

    # params
    #-------------------------------------------------------------------------------------------------------------------

    # HPC data, rule 3: light and rule 2: west
    file_name_rule_3 = "res_mjc189-1905-0517_2_ct_['p1']_sa_[1]_ga_[3]_rt_[3]_et_[1]"
    file_name_rule_2 = "res_mjc189-1905-0517_6_ct_['p1']_sa_[1]_ga_[3]_rt_[2]_et_[1]"
    file_name_rule_switch = "res_mjc189-1905-0517_4_ct_['p1']_sa_[1]_ga_[3]_rt_[2, 3]_et_[1]"

    # dictionary for all other parameters
    param_dic = {}

    # description of data
    param_dic["data_descr"] = ["RULE LIGHT","RULE WEST"]
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
    param_dic["nr_of_trials"] = 30

    # selected trial
    param_dic["sel_trial"] = 3

    # axis limit for plotting
    # jaccard: [-0.2,0.2]
    # cos: [-1,1]
    # 3D: [-0.5,0.5]
    axis_lim = np.zeros(6)
    axis_lim[0] = axis_lim[2]= axis_lim[4]= -1
    axis_lim[1] = axis_lim[3] = axis_lim[5] =1
    param_dic["axis_lim"] = axis_lim


    # options
    #-------------------------------------------------------------------------------------------------------------------
    plotting = False

    # get data
    #-------------------------------------------------------------------------------------------------------------------
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

    # rule switching
    # ------------------------------------------------------------------------------------------------------------------
    # if param_dic["dr_method_p2"] == 2:
    #     dimRed2D(data_rule_23, param_dic)
    # elif param_dic["dr_method_p2"] == 3:
    #     dimRed3DCompare(data_rule_23, param_dic)


    # compare two rules using dimensionality reduction for both sets separately (reduce to 2 or 3 dimensions)
    # ------------------------------------------------------------------------------------------------------------------
    # if param_dic["dr_method_p2"] == 2:
    #     dimRed2DCompare([data_rule_3, data_rule_2],param_dic)
    # elif param_dic["dr_method_p2"] == 3:
    #     dimRed3DCompare([data_rule_3, data_rule_2],param_dic)

    # compare two rules using dimensionality for the combined data (reduce to 2 or 3 dimensions)
    # ------------------------------------------------------------------------------------------------------------------
    #dimRedCombined([data_rule_3, data_rule_2], param_dic)

########################################################################################################################
#   TRANSITION ANALYSIS
########################################################################################################################

    # using difference vectors
    # ------------------------------------------------------------------------------------------------------------------
    StateTransitionAnalysis(data_rule_3,param_dic)



    if plotting:
        # plot activation matrix
        subplot(2,1,1)
        plotActMat(act_mat_r3,bin_interval)
        subplot(2,1,2)
        plotActMat(act_mat_r2,bin_interval)
        plt.show()



