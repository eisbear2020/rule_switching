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
#       - calcPopVectorEntropy: calculates shannon entropy for each population vector in act_mat
#       - multiDimScaling: returns fitted multi scale model using defined difference measure
#
#       Summarizing
#
#
#
#
#
#       Plotting
#
#
########################################################################################################################

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as sp
import matplotlib.colors as colors
from sklearn.metrics import jaccard_similarity_score
from sklearn.manifold import MDS
import matplotlib.cm as cm
from collections import OrderedDict
from scipy.spatial import distance
from mpl_toolkits import mplot3d

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
    elif diff_meas == "cos":
        # calculate difference matrix: cosine
        D = np.zeros([act_mat.shape[1], act_mat.shape[1]])

        # cosine
        for i, pop_vec_ref in enumerate(act_mat.T):
            for j, pop_vec_comp in enumerate(act_mat.T):
                    D[i, j] = distance.cosine(pop_vec_ref, pop_vec_comp)
                    # if one of the vectors contains only zeros --> division by zero for cosine
                    if math.isnan(D[i,j]):
                        D[i, j] = 1



    model = MDS(n_components=n_components, dissimilarity='precomputed', random_state=1)
    return model.fit_transform(D)


def StateTransitionAnalysis(data_set,param_dic):
# looks at difference vectors between transitions

    # calculate transition vector between two subsequent population states

    # perform dimensionality reduction



########################################################################################################################
#
# SUMMARIZING
#
########################################################################################################################

def dimRed2D(data_set,param_dic):
# plot result of one condition over multiple trials

    nr_trials_to_compare = min(len(data_set.keys()), param_dic["nr_of_trials"])
    # number of columns for plotting
    c_p = 3

    # create figure instance
    fig, ax = plt.subplots(int(nr_trials_to_compare/c_p), c_p)
    # row for plot
    c_r = 0

    for plot_ID, key in enumerate(data_set):
        if not np.mod(plot_ID, c_p):
            c_r += 1
        # compute equal number of trials for all conditions (for plotting)
        if plot_ID > (nr_trials_to_compare-1):
            break
        act_mat = getActivityMat(data_set, param_dic["bin_interval"], key)
        if param_dic["dr_method"] == "MDS":
            mds = multiDimScaling(act_mat, param_dic["dr_method_p1"], param_dic["dr_method_p2"])
            plot2DscatterLines(ax[c_r-1,np.mod(plot_ID,c_p)], mds, param_dic["axis_lim"])


    fig.suptitle(param_dic["dr_method"]+" : "+param_dic["dr_method_p1"], fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()




def dimRed2DCompare(data_sets,param_dic):
# compares results of two different conditions using dimensionality reduction
    nr_trials_to_compare = param_dic["nr_of_trials"]
    # comparing same number of trials even if one condition has more trials
    for data_set in data_sets:
        nr_trials_to_compare = min(len(data_set.keys()), nr_trials_to_compare)
    # create figure instance
    fig, ax = plt.subplots(nr_trials_to_compare, len(data_sets))

    # go over all data sets
    for dat_ID, (data, data_set_desc) in enumerate(zip(data_sets,param_dic["data_descr"])):
        ax[0, dat_ID].set_title(data_set_desc)
        # go over several trials
        for plot_ID, key in enumerate(data):
            # compute equal number of trials for all conditions (for plotting)
            if plot_ID > (nr_trials_to_compare-1):
                break
            act_mat = getActivityMat(data, param_dic["bin_interval"], key)
            if param_dic["dr_method"] == "MDS":
                mds = multiDimScaling(act_mat, param_dic["dr_method_p1"], param_dic["dr_method_p2"])
                plot2DscatterLines(ax[plot_ID, dat_ID], mds, param_dic["axis_lim"])

    fig.suptitle(param_dic["dr_method"]+" : "+param_dic["dr_method_p1"], fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()


def dimRed3DCompare(data_sets,param_dic):
# compares results of two different conditions using dimensionality reduction
    nr_trials_to_compare = param_dic["nr_of_trials"]
    # comparing same number of trials even if one condition has more trials
    for data_set in data_sets:
        nr_trials_to_compare = min(len(data_set.keys()), nr_trials_to_compare)
    # create figure instance
    fig = plt.figure()


    # go over all data sets
    for dat_ID, (data, data_set_desc) in enumerate(zip(data_sets, param_dic["data_descr"])):
        #ax[0, dat_ID].set_title(data_set_desc)
        # go over several trials
        for plot_ID, key in enumerate(data):
            # compute equal number of trials for all conditions (for plotting)
            if plot_ID > (nr_trials_to_compare-1):
                break
            act_mat = getActivityMat(data, param_dic["bin_interval"], key)
            if param_dic["dr_method"] == "MDS":
                mds = multiDimScaling(act_mat, param_dic["dr_method_p1"], param_dic["dr_method_p2"])
                ax = fig.add_subplot(nr_trials_to_compare, len(data_sets), (dat_ID*nr_trials_to_compare+plot_ID+1), projection='3d')
                plot3DscatterLines(ax, mds, param_dic["axis_lim"])
                #plot3DscatterLines(ax[plot_ID, dat_ID], mds, param_dic["axis_lim"])
    fig.suptitle(param_dic["dr_method"]+" : "+param_dic["dr_method_p1"],fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()


def dimRedCombined(data_sets,param_dic):
# combines two data sets, reduces the dimension and plots sets in different colors in 2D

    # get trial ID from both sets
    trial_ID_set1 = list(data_sets[0].keys())[param_dic["sel_trial"]]
    trial_ID_set2 = list(data_sets[1].keys())[param_dic["sel_trial"]]

    # calculate matrix of population vectors
    act_mat_set1 = getActivityMat(data_sets[0], param_dic["bin_interval"], trial_ID_set1)
    act_mat_set2 = getActivityMat(data_sets[1], param_dic["bin_interval"], trial_ID_set2)

    # combine both matrices
    comb_mat = np.hstack((act_mat_set1, act_mat_set2))

    # multi dimensional scaling
    # -------------------------------------------------------------------------------------------------------------------
    mds = multiDimScaling(comb_mat, param_dic["dr_method_p1"],param_dic["dr_method_p2"])

    data_sep = act_mat_set1.shape[1]

    # 2D
    if param_dic["dr_method_p2"] == 2:
        fig, ax = plt.subplots()
        plot2D2SetScatter(ax, mds, data_sep, param_dic)
    # 3D
    elif param_dic["dr_method_p2"] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot3D2SetScatter(ax, mds, data_sep, param_dic)

    fig.suptitle(param_dic["dr_method"] + " : " + param_dic["dr_method_p1"], fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()








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

def plot2DscatterLines(ax,mds,axis_lim):
# plots result of multidimensional scaling as scatter plot
    colors = cm.rainbow(np.linspace(0, 1, mds.shape[0]-1))
    for i,c in zip(range(0,mds.shape[0]-1),colors):
        ax.plot(mds[i:i+2,0],mds[i:i+2,1],color=c)
    #plt.title(title)
    ax.scatter(mds[:, 0], mds[:, 1], color="grey")
    ax.scatter(mds [0, 0], mds [0, 1],color="black", marker="x",label="start",zorder=200)
    ax.scatter(mds[-1, 0], mds[-1, 1], color="black", label="end",zorder=200)
    #plt.legend()
    ax.set_xlim(axis_lim[0], axis_lim[1])
    ax.set_ylim(axis_lim[2], axis_lim[3])
    #plt.axis('equal')
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

def plot3DscatterLines(ax,mds,axis_lim):
# plots result of multidimensional scaling as scatter plot
    colors = cm.rainbow(np.linspace(0, 1, mds.shape[0]-1))
    for i,c in zip(range(0,mds.shape[0]-1),colors):
        ax.plot(mds[i:i+2,0],mds[i:i+2,1],mds[i:i+2,2],color=c)
    #plt.title(title)
    ax.scatter(mds[:, 0], mds[:, 1], mds[:, 2], color="grey")
    ax.scatter(mds [0, 0], mds [0, 1], mds[0, 2],color="black", marker="x",label="start",zorder=200)
    ax.scatter(mds[-1, 0], mds[-1, 1], mds[-1, 2], color="black", label="end",zorder=200)
    #plt.legend()
    ax.set_xlim(axis_lim[0], axis_lim[1])
    ax.set_ylim(axis_lim[2], axis_lim[3])
    ax.set_zlim(axis_lim[4], axis_lim[5])
    #plt.axis('equal')
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

def plot2D2SetScatter(ax,mds,data_sep,param_dic):
    # plot first rule
    for i,c in enumerate(mds):
        if i < data_sep-1:
            ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], color="lightblue")
        elif data_sep <= i:
            ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], color="lightcoral")
    ax.scatter(mds[:data_sep, 0], mds[:data_sep, 1], color="b",label=param_dic["data_descr"][0])
    ax.scatter(mds[data_sep:, 0], mds[data_sep:, 1], color="r",label=param_dic["data_descr"][1])
    ax.scatter(mds [0, 0], mds [0, 1],color="black", marker="x",label="start",zorder=200)
    ax.scatter(mds[data_sep-1, 0], mds[data_sep-1, 1], color="black", label="end",zorder=200)
    ax.scatter(mds [data_sep, 0], mds [data_sep, 1],color="black", marker="x",label="start",zorder=200)
    ax.scatter(mds[-1, 0], mds[-1, 1], color="black", label="end",zorder=200)

    ax.set_yticklabels([])
    ax.set_xticklabels([])


def plot3D2SetScatter(ax,mds,data_sep,param_dic):
    # plot first rule
    for i,c in enumerate(mds):
        if i < data_sep-1:
            ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1],mds[i:i + 2, 2], color="lightblue")
        elif data_sep <= i:
            ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1],mds[i:i + 2, 2], color="lightcoral")
    ax.scatter(mds[:data_sep, 0], mds[:data_sep, 1],mds[:data_sep, 2], color="b",label=param_dic["data_descr"][0])
    ax.scatter(mds[data_sep:, 0], mds[data_sep:, 1],mds[data_sep:, 2], color="r",label=param_dic["data_descr"][1])
    ax.scatter(mds [0, 0], mds [0, 1],mds [0, 2],color="black", marker="x",label="start",zorder=200)
    ax.scatter(mds[data_sep-1, 0], mds[data_sep-1, 1],mds[data_sep-1, 2], color="black", label="end",zorder=200)
    ax.scatter(mds [data_sep, 0], mds [data_sep, 1],mds [data_sep, 2],color="black", marker="x",label="start",zorder=200)
    ax.scatter(mds[-1, 0], mds[-1, 1],mds[-1, 2], color="black", label="end",zorder=200)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])