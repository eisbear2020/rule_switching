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
#       - getData: returns dictionary with spike data for selected trials and selected cells
#
########################################################################################################################

import numpy as np

def get_cell_ID(data_dir, s_exp,cell_type_array):
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


def sel_trials(timestamps, trial_sel):
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

def linearize_location(whl_rot, timestamps):
# returns linearized location, skipping locations that have -1/1 (recording errors)
# - calculates distance from the center (101,116) using euclidean distance
# - invertes results for start arm (location center - location)
# - adds location of center to goal arm

    data = whl_rot.copy()

    for trial in range(len(timestamps)):
        ind = range(timestamps[trial, 0], timestamps[trial, 4] + 1)
        indx = np.where(data[ind, 0] > -1)[0] + ind[0]
        indy = np.where(data[ind, 1] > -1)[0] + ind[0]
        data[ind, 0] = np.interp(ind, np.where(data[ind, 0] > -1)[0] + ind[0], data[indx, 0])
        data[ind, 1] = np.interp(ind, np.where(data[ind, 1] > -1)[0] + ind[0], data[indy, 1])

    # location of center
    p = (101, 116)
    dis = np.zeros(len(data))
    for trial in range(len(timestamps)):
        ind = range(timestamps[trial, 0], timestamps[trial, 4] + 1)

        for i in ind:
            # euclidean distance to center
            dis[i] = np.sqrt((data[i, 0] - p[0]) ** 2 + (data[i, 1] - p[1]) ** 2)
    di = dis.copy()
    for trial in range(len(timestamps)):
        ind = range(timestamps[trial, 0], timestamps[trial, 4] + 1)
        switch = np.where(di[ind] == np.min(di[ind]))[0][0] + ind[0]  # index over entire whl
        for i in ind:
            di[i] = di[i] - np.min(di[ind])
        di[ind[0]:switch + 1] = max(di[ind[0]:switch + 1]) - di[ind[0]:switch + 1]
        di[switch + 1:ind[-1]] = di[switch] + di[switch + 1:ind[-1]]
        di[ind[0]:ind[-1]] = di[ind[0]:ind[-1]] - di[switch] + 100  # aligning to centre 100

    return di


def get_data(trial_IDs, cell_IDs, clu, res,timestamps):
# returns dictionary with spike data for selected trials and selected cells

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

def get_location(trial_IDs,whl_rot,timestamps):
# returns dictionary with location for selected trials
    #linearized location
    whl_lin = linearize_location(whl_rot,timestamps)

    # dictionary with locations
    loc = {}
    # go through selected trials
    for trial_ID in trial_IDs:
        #-----------------------------------------------
        #  timestamps: 20kHz/512 --> 25.6 ms per time bin
        #  whl: 20kHz/512 --> both have the same order of magnitude
        t_start = timestamps[trial_ID,0]
        t_end = timestamps[trial_ID, 4]
        # select locations that correspond to time interval of the trial
        loc_trial = whl_lin[t_start:t_end]
        # append data
        loc["trial"+str(trial_ID)] = loc_trial

    return loc
