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
from filter_functions import get_cell_ID
from filter_functions import sel_trials
from filter_functions import get_data
from filter_functions import get_location

if __name__ == '__main__':

    # data directory
    #-------------------------------------------------------------------------------------------------------------------
    data_dir = "../02 Data"

    # select experiment
    #-------------------------------------------------------------------------------------------------------------------
    #s_exp = "mjc189-1905-0517"
    s_exp = "mjc190-1607-0515"

    # select cell type
    #-------------------------------------------------------------------------------------------------------------------
    # p1: pyramidal cells of the HPC, p2 - p3: pyramidal cells of the PFC ,b1: interneurons of HPC
    # b2 - b3: interneurons of HPC

    cell_type_array = ["p1"]
    cell_IDs = get_cell_ID(data_dir, s_exp, cell_type_array)

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
    goalarm = [2]
    ruletype = [1,3] #3: light
    errortrial = [1]

    trial_sel = {"startarm": startarm,"goalarm": goalarm, "ruletype":ruletype, "errortrial": errortrial}
    trial_IDs = sel_trials(timestamps,trial_sel)

    # if no matching trials were found throw error
    if not trial_IDs:
        raise Exception("No matching trials found")


    # get location data
    #-------------------------------------------------------------------------------------------------------------------
    whl_rot = np.loadtxt(data_dir + "/" + s_exp + "/" + s_exp + "_" + env + ".whl_rot").astype(int)

    loc = get_location(trial_IDs, whl_rot, timestamps)

    # save data as pickle
    filename = "temp_data/"+"whl_lin_"+s_exp+"_"+env+"_"+"ct_"+str(cell_type_array)+"_sa_"+str(startarm)+"_ga_"+\
               str(goalarm)+"_rt_"+str(ruletype)+"_et_"+str(errortrial)
    outfile = open(filename, 'wb')
    pickle.dump(loc,outfile)
    outfile.close()


    # get spike data
    #-------------------------------------------------------------------------------------------------------------------

    # load cluster IDs and time of spikes
    clu = np.loadtxt(data_dir+"/"+s_exp+"/"+s_exp+"_"+env+".clu").astype(int)
    res = np.loadtxt(data_dir + "/" + s_exp + "/" + s_exp + "_" + env + ".res").astype(int)

    # extract data
    data = get_data(trial_IDs,cell_IDs,clu,res,timestamps)

    # save data as pickle
    filename = "temp_data/"+"res_"+s_exp+"_"+env+"_"+"ct_"+str(cell_type_array)+"_sa_"+str(startarm)+"_ga_"+\
               str(goalarm)+"_rt_"+str(ruletype)+"_et_"+str(errortrial)
    outfile = open(filename, 'wb')
    pickle.dump(data,outfile)
    outfile.close()