########################################################################################################################
#
#   NEURAL ACTIVITY ANALYSIS
#
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

import numpy as np


if __name__ == '__main__':

    # select experiment
    s_exp = "mjc189-1905-0517"

    # load data
    # to get rate map:
    # data["binmat"][SESSION][CELL][TIME INTERVAL]
    # TIME INTERVAL from data["tist"]
    data = np.load("../02 Data/"+s_exp+"/data.npy").item()
    x = data['binmat'][2][3][853:1514]
    print(x)






