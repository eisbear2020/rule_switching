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
from helper_func import dimRedCompare

if __name__ == '__main__':

    # params
    #-------------------------------------------------------------------------------------------------------------------
    param_dic = {}


    # HPC data, rule: 3 (light)
    file_name_rule_3 = "res_mjc189-1905-0517_2_ct_['p1']_sa_[1]_ga_[3]_rt_[3]_et_[1]"
    file_name_rule_2 = "res_mjc189-1905-0517_6_ct_['p1']_sa_[1]_ga_[3]_rt_[2]_et_[1]"

    # interval for binning in s
    param_dic["bin_interval"] = 0.1

    # axis limit for plotting
    # jaccard: [-0.2,0.2]
    # cos: [-1,1]
    axis_lim = np.zeros(6)
    axis_lim[0] = axis_lim[2]= -1
    axis_lim[1] = axis_lim[3] = 1
    axis_lim[4] = axis_lim[5] = 1
    param_dic["axis_lim"] = axis_lim

    # define method for dimensionality reduction
    param_dic["dr_method"] = "MDS" # multi dimensional scaling
    # first parameter of method:
    # MDS --> p1: difference measure ["jaccard","cos"]
    param_dic["dr_method_p1"] = "cos"
    # second parameter of method:
    # MDS --> p2: number of components
    param_dic["dr_method_p2"] = 3

    # select trial
    trial = 1

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

    # # pick trial
    # trial_r3 = list(data_rule_3.keys())[trial]
    # trial_r2 = list(data_rule_2.keys())[trial]
    #
    #
    # # calculate matrix of population vectors
    # act_mat_r3 = getActivityMat(data_rule_3,param_dic["bin_interval"],trial_r3)
    # act_mat_r2 = getActivityMat(data_rule_2,param_dic["bin_interval"],trial_r2)
    #
    # # multi dimensional scaling
    # # -------------------------------------------------------------------------------------------------------------------
    # mds_r3 = multiDimScaling(act_mat_r3,"jaccard",2)
    # mds_r2= multiDimScaling(act_mat_r2, "jaccard", 2)

    dimRedCompare([data_rule_3, data_rule_2],["LIGHT","WEST"],param_dic)



    # plotMDS(mds_r3,axis_lim,"light")
    # plt.show()




    if plotting:
        # plot activation matrix
        subplot(2,1,1)
        plotActMat(act_mat_r3,bin_interval)
        subplot(2,1,2)
        plotActMat(act_mat_r2,bin_interval)
        plt.show()



