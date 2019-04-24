########################################################################################################################
#
#   SELECT DATA
#
#   Description:
#
#       - getCellID: returns cell IDs of selected cell type
#       - setTrials: returns trial IDs of trials that meet the conditions in trial_sel
#       - getData: returns dictionary with spike data for selected trials and selected cells
#       - save_selected_data:
#
#           - imports data from .clu, .res, .timestamp and saves conditional data to "temp_data" directory
#
#           - criteria for selection: directory, experiment, cell type, environment, trial conditions (e.g
#             success/fail)
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


def get_cell_ID(data_dir, s_exp,cell_type_array):
    # returns cell IDs of selected cell type
    # ---------------------------------------
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
    rule_identifier = []
    trial_intervals = []
    trial_new_rule = 0
    temp = 0
    # go through all trials:
    for trial_ID, trial in enumerate(timestamps):
        # check if trial agrees with conditions
        # start,centrebegin, centreend, goalbegin, goalend, startarm, goalarm, control, lightarm, ruletype, errortrial
        if trial[9] in trial_sel["ruletype"] and trial[10] in trial_sel["errortrial"] and \
        trial[5] in trial_sel["startarm"] and trial[6] in trial_sel["goalarm"]:
            trial_intervals.append(trial_ID)
            # remember number of trials saved to identify rule switch case
            temp += 1
            # check if new rule --> if yes, append to rule identifier
            if not rule_identifier:
                rule_identifier.append(trial[9])
            elif trial[9] != rule_identifier[-1]:
                rule_identifier.append(trial[9])
                trial_new_rule = temp
    return trial_intervals, rule_identifier, trial_new_rule


def linearize_location(whl_rot, timestamps):
    # returns linearized location, skipping locations that have -1/1 (recording errors)
    # - calculates distance from the center (101,116) using euclidean distance
    # - inverts results for start arm (location center - location)
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
        # -----------------------------------------------
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


def get_location(trial_IDs, whl_rot, timestamps):
    # returns dictionary with location for selected trials
    # linearized location
    whl_lin = linearize_location(whl_rot, timestamps)

    # dictionary with locations
    loc = {}
    # go through selected trials
    for trial_ID in trial_IDs:
        # -----------------------------------------------
        #  timestamps: 20kHz/512 --> 25.6 ms per time bin
        #  whl: 20kHz/512 --> both have the same order of magnitude
        t_start = timestamps[trial_ID,0]
        t_end = timestamps[trial_ID, 4]
        # select locations that correspond to time interval of the trial
        loc_trial = whl_lin[t_start:t_end]
        # append data
        loc["trial"+str(trial_ID)] = loc_trial

    return loc


def save_selected_data(data_selection_dictionary):

    # data directory
    # ------------------------------------------------------------------------------------------------------------------
    data_dir = data_selection_dictionary["data_dir"]

    # select session
    # ------------------------------------------------------------------------------------------------------------------
    s_exp = data_selection_dictionary["session_name"]

    # select cell type
    # ------------------------------------------------------------------------------------------------------------------
    cell_type_array = data_selection_dictionary["cell_type_array"]
    cell_IDs = get_cell_ID(data_dir, s_exp, cell_type_array)

    # select trials
    # ------------------------------------------------------------------------------------------------------------------

    startarm = data_selection_dictionary["start_arm"]
    goalarm = data_selection_dictionary["goal_arm"]
    ruletype = data_selection_dictionary["rule_type"]
    errortrial = data_selection_dictionary["error_trial"]

    # go through all environments
    # ------------------------------------------------------------------------------------------------------------------
    # 2: first session
    # 4: session after sleep (with rule switch)
    # 6: last session

    # create dictionary
    data_dic = {
        "2": [],
        "4": [],
        "6": []
    }

    for env in data_dic:

        timestamps = np.loadtxt(data_dir+"/"+s_exp+"/"+s_exp+"_"+env+".timestamps").astype(int)

        trial_sel = {"startarm": startarm, "goalarm": goalarm, "ruletype": ruletype, "errortrial": errortrial}
        trial_IDs, rule_identifier, new_rule_trial = sel_trials(timestamps, trial_sel)

        # if no matching trials were found throw error
        if not trial_IDs:
            raise Exception("No matching trials found")

        # get location data
        # --------------------------------------------------------------------------------------------------------------
        whl_rot = np.loadtxt(data_dir + "/" + s_exp + "/" + s_exp + "_" + env + ".whl_rot").astype(int)

        loc = get_location(trial_IDs, whl_rot, timestamps)

        # get spike data
        # --------------------------------------------------------------------------------------------------------------

        # load cluster IDs and time of spikes
        clu = np.loadtxt(data_dir+"/"+s_exp+"/"+s_exp+"_"+env+".clu").astype(int)
        res = np.loadtxt(data_dir + "/" + s_exp + "/" + s_exp + "_" + env + ".res").astype(int)

        # extract data
        data = get_data(trial_IDs, cell_IDs, clu, res, timestamps)

        # save data dictionary as pickle
        data_dic[env] = {
            "whl_lin": loc,
            "res": data,
            "info": {"new_rule_trial": new_rule_trial, "rule_order": rule_identifier}
        }

    filename = "temp_data/"+s_exp+"_"+"ct_"+str(cell_type_array)+"_sa_"+str(startarm)+"_ga_"+\
               str(goalarm)+"_et_"+str(errortrial)
    outfile = open(filename, 'wb')
    pickle.dump(data_dic, outfile)
    outfile.close()
