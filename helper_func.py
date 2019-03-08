########################################################################################################################
#
#   Helper functions
#
#
#   Author: Lars Bollmann
#
#   Created: 08/03/2019
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


def getCellID(data_dir, s_exp,cell_type_array):
# returns cell IDs of selected cell type
#---------------------------------------
# select cells of one region
# p1: pyramidal cells of the HPC
# p2 - p3: pyramidal cells of the PFC
# b1: interneurons of HPC
# b2 - b3: interneurons of HPC

    with open(data_dir + "/" + s_exp + "/" + s_exp + ".des") as f:
        des = f.read()
    des = des.splitlines()

    cell_IDs = []

    for cell_type in cell_type_array:
        temp = [i + 2 for i in range(len(des)) if des[i] == cell_type]
        cell_IDs = cell_IDs + temp
    return cell_IDs
