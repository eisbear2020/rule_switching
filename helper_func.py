########################################################################################################################
#
#   Helper functions
#
#   Description:
#
#       - helper functions for importing & selecting data
#
#   Author: Lars Bollmann
#
#   Created: 08/03/2019
#
#   Structure:
#
#       Computing
#
#       - getCellID: returns cell IDs of selected cell type
#       - setTrials: returns trial IDs of trials that meet the conditions in trial_sel
#       - getData: returns dictionary with data for selected trials and selected cells
#       - getActivityMat: computes activity matrix (matrix with population vectors)
#
#       Plotting
#
#
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import matplotlib.colors as colors
from sklearn.metrics import jaccard_similarity_score
from sklearn.manifold import MDS

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


def getActivityMat(data,bin_interval,trial):
# computes activity matrix: bin_interval in seconds --> sums up the activity within one time interval
# rows: cells
# columns: time bins

    # find first and last firing for each trial
    first_firing = np.inf
    last_firing = 0

    for key,value in data[trial].items():
        if value:
            first_firing = int(np.amin([first_firing, np.amin(value)]))
            last_firing = int(np.amax([last_firing, np.amax(value)]))

    # duration of trial (one time bin: 0.05ms)
    dur_trial = (last_firing-first_firing)* 0.05*1e-3
    nr_intervals = int(dur_trial/bin_interval)
    size_intervals = int((last_firing-first_firing)/nr_intervals)

    # binary matrix
    act_mat = np.zeros([len(data[trial].keys()),nr_intervals])

    # go through all cells
    for cell_ID, [key,cell] in enumerate(data[trial].items()):
        # go through all time intervals
        for i in range(nr_intervals):
            start_intv = first_firing+i*size_intervals
            end_intv = first_firing+(i+1)*size_intervals
            cell_spikes_intv = [x for x in cell if start_intv <= x < end_intv]
            act_mat[cell_ID,i] = len(cell_spikes_intv)

    return act_mat

def calcPopVectorEntropy(act_mat):
# calculates shannon entropy for each population vector in act_mat
    pop_vec_entropy = np.zeros(act_mat.shape[1])
    # calculate entropy
    for i,pop_vec in enumerate(act_mat.T):
        # add small value because of log
        pop_vec_entropy[i] = sp.entropy(pop_vec+0.000001)
    return pop_vec_entropy

def multiDimScaling(act_mat,diff_meas,n_components):
# returns fitted multi scale model using defined difference measure
    if diff_meas == "jaccard":
        # calculate difference matrix: Jaccard
        D = np.zeros([act_mat.shape[1],act_mat.shape[1]])

        # Jaccard similarity
        for i,pop_vec_ref in enumerate(act_mat.T):
            for j,pop_vec_comp in enumerate(act_mat.T):
                D[i,j] = jaccard_similarity_score(pop_vec_ref,pop_vec_comp)

        # want difference --> diff_jaccard = 1 - sim_jaccard
        D = 1 - D
        # plt.imshow(D)
        # plt.colorbar()
        # plt.show()

    model = MDS(n_components=n_components, dissimilarity='precomputed', random_state=1)
    return model.fit_transform(D)















########################################################################################################################
#
# PLOTTING
#
########################################################################################################################


def plotActMat(act_mat,bin_interval):
# plot activation matrix (matrix of population vectors)
    plt.imshow(act_mat, vmin=0, vmax=act_mat.max(), cmap='jet', aspect='auto')
    plt.ylabel("CELL ID")
    plt.xlabel("TIME BINS / " + str(bin_interval) + " s")
    plt.title("CELL ACTIVATION / SPIKES PER TIME BIN")
    a = plt.colorbar()
    a.set_label("SPIKES")



#
# def linearize_whl(whl,tist):
#     lin_whl = np.zeros(len(whl)) - 1
#     speed = np.zeros(len(whl)) - 1
#     for t in tist:
#         lw, ls = l.linearize(whl[t[0]:t[4]])
#         lin_whl[t[0]:t[4]] = lw
#         speed[t[0]:t[4]] = ls
#     return lin_whl, speed