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
from helper_func import calcPopVectorEntropy
from helper_func import plotActMat
from helper_func import plotEntropy


if __name__ == '__main__':

    # params
    #-------------------------------------------------------------------------------------------------------------------
    #file_name = "res_mjc189-1905-0517ct_['p2', 'p3']_la_[2]_rt_[3]_et_[1]"
    file_name = "res_mjc189-1905-0517ct_['p1']_la_[2]_rt_[3]_et_[1]"

    # interval for binning in s
    bin_interval = 0.1
    # select trial
    trial = "trial10"

    # options
    #-------------------------------------------------------------------------------------------------------------------
    plotting = True


    # get data
    #-------------------------------------------------------------------------------------------------------------------
    infile = open("temp_data/"+file_name, 'rb')
    data = pickle.load(infile)
    infile.close()

    # calculate matrix of population vectors
    act_mat = getActivityMat(data,bin_interval,trial)

    # calculate entropy
    pop_vec_entropy = calcPopVectorEntropy(act_mat)






    if plotting:
        plt.subplot(2,1,1)
        # plot activation matrix
        plotActMat(act_mat,bin_interval)
        plt.subplot(2,1,2)
        # plot entropy
        plotEntropy(pop_vec_entropy,act_mat)
        plt.subplots_adjust(hspace=0.5)
        plt.show()



