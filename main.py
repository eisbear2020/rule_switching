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
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

if __name__ == '__main__':

    # params
    #-------------------------------------------------------------------------------------------------------------------

    # HPC data, rule: 3 (light)
    file_name_rule_3 = "res_mjc189-1905-0517_2_ct_['p1']_sa_[1]_ga_[3]_rt_[3]_et_[1]"
    file_name_rule_2 = "res_mjc189-1905-0517_6_ct_['p1']_sa_[1]_ga_[3]_rt_[2]_et_[1]"

    # interval for binning in s
    bin_interval = 0.1
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

    # pick trial
    trial_r3 = list(data_rule_3.keys())[trial]
    trial_r2 = list(data_rule_2.keys())[trial]


    # calculate matrix of population vectors
    act_mat_r3 = getActivityMat(data_rule_3,bin_interval,trial_r3)
    act_mat_r2 = getActivityMat(data_rule_2,bin_interval,trial_r2)

    # multi dimensional scaling
    # -------------------------------------------------------------------------------------------------------------------
    mds_r3 = multiDimScaling(act_mat_r3,"jaccard",2)
    mds_r2= multiDimScaling(act_mat_r2, "jaccard", 2)

    plt.subplot(2,1,1)
    colors = cm.rainbow(np.linspace(0, 1, mds_r3.shape[0]-1))
    for i,c in zip(range(0,mds_r3.shape[0]-1),colors):
        plt.plot(mds_r3[i:i+2,0],mds_r3[i:i+2,1],color=c)
    plt.title("Rule: light")
    plt.scatter(mds_r3[:, 0], mds_r3[:, 1], color="grey")
    plt.scatter(mds_r3 [0, 0], mds_r3 [0, 1],color="red", label = "start")
    plt.scatter(mds_r3[-1, 0], mds_r3[-1, 1], color="black", label="end")
    plt.legend()
    plt.xlim(np.amin([mds_r2.min(), mds_r3.min()]), np.amax([mds_r2.max(), mds_r3.max()]))
    plt.ylim(np.amin([mds_r2.min(), mds_r3.min()]), np.amax([mds_r2.max(), mds_r3.max()]))
    plt.subplot(2, 1, 2)
    colors = cm.rainbow(np.linspace(0, 1, mds_r2.shape[0]-1))
    for i,c in zip(range(0,mds_r2.shape[0]-1),colors):
        plt.plot(mds_r2[i:i+2,0],mds_r2[i:i+2,1],color=c)
    plt.title("Rule: go west")
    plt.scatter(mds_r2[:, 0], mds_r2[:, 1], color="grey")
    plt.scatter(mds_r2 [0, 0], mds_r2 [0, 1],color="red", label = "start")
    plt.scatter(mds_r2[-1, 0], mds_r2[-1, 1], color="black", label="end")
    plt.xlim(np.amin([mds_r2.min(), mds_r3.min()]), np.amax([mds_r2.max(), mds_r3.max()]))
    plt.ylim(np.amin([mds_r2.min(), mds_r3.min()]), np.amax([mds_r2.max(), mds_r3.max()]))
    plt.legend()
    plt.show()






    if plotting:
        # plot activation matrix
        subplot(2,1,1)
        plotActMat(act_mat_r3,bin_interval)
        subplot(2,1,2)
        plotActMat(act_mat_r2,bin_interval)
        plt.show()



