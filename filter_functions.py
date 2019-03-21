########################################################################################################################
#
#   Filter functions
#
#   Description:
#
#       - functions for importing & selecting data
#
#   Author: Lars Bollmann
#
#   Created: 21/03/2019
#
#   Structure:
#
#       - getCellID: returns cell IDs of selected cell type
#       - setTrials: returns trial IDs of trials that meet the conditions in trial_sel
#       - getData: returns dictionary with data for selected trials and selected cells
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


def selTrials(timestamps, trial_sel):
# returns trial IDs of trials that meet the conditions in trial_sel

    trial_intervals = []
    # go through all trials:
    for trial_ID, trial in enumerate(timestamps):
        # check if trial agrees with conditions
        #start,centrebegin,centreend,goalbegin,goalend,startarm,goalarm,control,lightarm,ruletype,errortrial
        if trial[9] in trial_sel["ruletype"] and trial[10] in trial_sel["errortrial"] and \
        trial[5] in trial_sel["startarm"] and trial[6] in trial_sel["goalarm"]:
            trial_intervals.append(trial_ID)
    return trial_intervals

def getData(trial_IDs, cell_IDs, clu, res,timestamps):
# returns dictionary with data for selected trials and selected cells

    data = {}
    # go through selected trials
    for trial_ID in trial_IDs:
        # create entry in dictionary
        data["trial"+str(trial_ID)] = {}
        #-----------------------------------------------
        # timestamps: 20kHz/512 --> 25.6 ms per time bin
        # res data: 20kHz
        # time interval for res data: timestamp data * 512
        t_start = timestamps[trial_ID,0] * 512
        t_end = timestamps[trial_ID, 4] * 512
        # go through selected cells
        for cell_ID in cell_IDs:
            cell_spikes_trial = []
            # find all entries of the cell_ID in the clu list
            entries_cell = np.where(clu == cell_ID)
            # append entries from res file (data is shifted by -1 with respect to clu list)
            ind_res_file = entries_cell[0] - 1
            # only use spikes that correspond to time interval of the trial
            cell_spikes_trial = [x for x in res[ind_res_file] if t_start < x < t_end]
            # append data
            data["trial" + str(trial_ID)]["cell"+str(cell_ID)] = cell_spikes_trial

    return data
