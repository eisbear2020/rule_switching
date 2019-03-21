########################################################################################################################
#
#   SELECT DATA
#
#   Description:
#
#       - imports data from .clu, .res, .timestamp and saves conditional data to "temp_data" directory
#
#       - criteria for selection: directory, experiment, cell type, environment, trial conditions (e.g
#         success/fail)
#
#
#   Author: Lars Bollmann
#
#   Created: 11/03/2019
#
#
#
########################################################################################################################

import numpy as np
import pickle
from filter_functions import getCellID
from filter_functions import selTrials
from filter_functions import getData

if __name__ == '__main__':

    # data directory
    #-------------------------------------------------------------------------------------------------------------------
    data_dir = "../02 Data"

    # select experiment
    #-------------------------------------------------------------------------------------------------------------------
    s_exp = "mjc189-1905-0517"

    # select cell type
    #-------------------------------------------------------------------------------------------------------------------
    # p1: pyramidal cells of the HPC, p2 - p3: pyramidal cells of the PFC ,b1: interneurons of HPC
    # b2 - b3: interneurons of HPC

    cell_type_array = ["p1"]
    cell_IDs = getCellID(data_dir, s_exp, cell_type_array)

    # select environment:
    #-------------------------------------------------------------------------------------------------------------------
    # 2: first session
    # 4: session after sleep (with rule switch)
    # 6: last session

    env = "4"
    timestamps = np.loadtxt(data_dir+"/"+s_exp+"/"+s_exp+"_"+env+".timestamps").astype(int)

    # select trials
    #-------------------------------------------------------------------------------------------------------------------

    startarm = [1]
    goalarm = [3]
    ruletype = [2,3] #3: light
    errortrial = [1]

    trial_sel = {"startarm": startarm,"goalarm": goalarm, "ruletype":ruletype, "errortrial": errortrial}
    trial_IDs = selTrials(timestamps,trial_sel)

    # if no matching trials were found throw error
    if not trial_IDs:
        raise Exception("No matching trials found")



    # get data
    #-------------------------------------------------------------------------------------------------------------------

    # load cluster IDs and time of spikes
    clu = np.loadtxt(data_dir+"/"+s_exp+"/"+s_exp+"_"+env+".clu").astype(int)
    res = np.loadtxt(data_dir + "/" + s_exp + "/" + s_exp + "_" + env + ".res").astype(int)

    # extract data
    data = getData(trial_IDs,cell_IDs,clu,res,timestamps)

    # save data as pickle
    filename = "temp_data/"+"res_"+s_exp+"_"+env+"_"+"ct_"+str(cell_type_array)+"_sa_"+str(startarm)+"_ga_"+\
               str(goalarm)+"_rt_"+str(ruletype)+"_et_"+str(errortrial)
    outfile = open(filename, 'wb')
    pickle.dump(data,outfile)
    outfile.close()