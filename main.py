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
import numpy as np
from helper_func import getCellID
from helper_func import selTrials
from helper_func import getData

if __name__ == '__main__':

    # params
    #-------------------------------------------------------------------------------------------------------------------
    file_name = "res_mjc189-1905-0517ct_['p2', 'p3']_la_[2]_rt_[3]_et_[1]"

    # get data
    #-------------------------------------------------------------------------------------------------------------------
    infile = open("temp_data/"+file_name, 'rb')
    data = pickle.load(infile)
    infile.close()


    print(data.keys())
    print(data["trial3"]["cell27"])





