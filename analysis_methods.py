########################################################################################################################
#
#   Analysis methods
#
#   Description:
#
#       - analysis steps/methods
#
#   Author: Lars Bollmann
#
#   Created: 21/03/2019
#
#   Structure:
#
#       - manifoldTransition: evaluates the manifold change e.g during rule switch, transforming every trial separately
#
#       - manifoldTransitionConc:   change of the manifold e.g. during rule switching, using data from multiple trials
#                                   for transformation (dim. reduction) and separating data afterwards
#
#       - manifoldCompare:  compares results of two different conditions using dimensionality reduction and transforming
#                           each trial separately
#       - manifoldCompareConc:  compares results of two different conditions using dimensionality reduction.
#                               Uses concatenated data for transformation and separates data afterwards
#
#       - StateTransitionAnalysis: analysis the state transitions using difference vectors between two population states
#
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from comp_functions import get_activity_mat
from comp_functions import multi_dim_scaling
from comp_functions import pop_vec_diff
from comp_functions import perform_PCA

from plotting_functions import plot_2D_scatter
from plotting_functions import plot_3D_scatter

def manifold_transition(data_set,param_dic):
# analyses the transition of the manifold e.g. for the rule switch case. Each trial is transformed individually

    nr_trials_to_compare = param_dic["nr_of_trials"]

    nr_trials_to_compare = min(len(data_set.keys()), nr_trials_to_compare)


    # number of columns for plotting
    c_p = param_dic["c_p"]

    # 2D
    if param_dic["dr_method_p2"] == 2:
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
            act_mat, loc_vec = get_activity_mat(data_set[key], param_dic)
            if param_dic["dr_method"] == "MDS":
                mds = multi_dim_scaling(act_mat, param_dic)
                plot_2D_scatter(ax[c_r-1,np.mod(plot_ID,c_p)],mds,param_dic)

    # 3D
    elif param_dic["dr_method_p2"] == 3:
        # create figure instance
        fig = plt.figure()

        for plot_ID, key in enumerate(data_set):
            # compute equal number of trials for all conditions (for plotting)
            if plot_ID > (nr_trials_to_compare-1):
                break
            act_mat, loc_vec = get_activity_mat(data_set[key], param_dic)
            if param_dic["dr_method"] == "MDS":
                mds = multi_dim_scaling(act_mat, param_dic)
                ax = fig.add_subplot(int(nr_trials_to_compare/c_p), c_p, plot_ID+1, projection='3d')
                plot_3D_scatter(ax, mds, param_dic)

    fig.suptitle(param_dic["dr_method"]+" : "+param_dic["dr_method_p1"],fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()


def manifold_transition_conc(data_set,loc_set, param_dic):
# change of the manifold e.g. during rule switching, using data from multiple trials for transformation (dim. reduction)
# and separating data afterwards
# input: data_set --> firing times, loc_set --> locations

    nr_trials_to_compare = min(len(data_set.keys()), param_dic["nr_of_trials"])
    nr_cells = len(next(iter(data_set.values())))
    # matrix with concatenated spike data
    dat_mat = np.array([]).reshape(nr_cells,0)
    # vector with concatenated location data
    loc_vec = np.empty((0,1))
    # separator integer
    data_sep = np.zeros(nr_trials_to_compare+1)
    # nr of trials to compare
    trial_counter = 0
    # concatenate all trials and save separator for trials
    for i,key in enumerate(data_set):
        if trial_counter > (nr_trials_to_compare-1):
            break
        else:
            act_mat, loc_vec_part = get_activity_mat(data_set[key], param_dic,loc_set[key])
            # concatenate spike matrices
            dat_mat = np.hstack((dat_mat,act_mat))
            # concatenate location vectors
            loc_vec = np.vstack((loc_vec,np.expand_dims(loc_vec_part,1)))
            data_sep[trial_counter + 1] = data_sep[trial_counter] + act_mat.shape[1]
            trial_counter += 1

    # dimensionality reduction: multi dimensional scaling
    if param_dic["dr_method"] == "MDS":
        result_dr = multi_dim_scaling(dat_mat, param_dic)
    # dimensionality reduction: principal component analysis
    elif param_dic["dr_method"] == "PCA":
        result_dr = perform_PCA(dat_mat, param_dic)


    # number of columns for plotting
    c_p = 3
    # row for plot
    c_r = 0

    # 2D
    if param_dic["dr_method_p2"] == 2:
        # create figure instance
        fig, ax = plt.subplots(int(nr_trials_to_compare / c_p), c_p)

        for data_ID in range(nr_trials_to_compare):
            if not np.mod(data_ID, c_p):
                c_r += 1
            data = result_dr[int(data_sep[data_ID]):int(data_sep[data_ID+1]), :]
            plot_2D_scatter(ax[c_r-1, np.mod(data_ID,c_p)], data, param_dic,[],loc_vec)

    # 3D
    elif param_dic["dr_method_p2"] == 3:
        # create figure instance
        fig = plt.figure()

        for data_ID in range(nr_trials_to_compare):
            if not np.mod(data_ID, c_p):
                c_r += 1
            data = result_dr[int(data_sep[data_ID]):int(data_sep[data_ID+1]), :]
            ax = fig.add_subplot(int(nr_trials_to_compare / c_p), c_p, data_ID+1, projection='3d')
            plot_3D_scatter(ax, data, param_dic,[],loc_vec)


    fig.suptitle(param_dic["dr_method"]+" : "+param_dic["dr_method_p1"], fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()

def manifold_compare(data_sets,param_dic):
# compares results of two different conditions using dimensionality reduction. Transforms each trial individually
    nr_trials_to_compare = param_dic["nr_of_trials"]
    # comparing same number of trials even if one condition has more trials
    for data_set in data_sets:
        nr_trials_to_compare = min(len(data_set.keys()), nr_trials_to_compare)

    # 2D
    if param_dic["dr_method_p2"] == 2:
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
                act_mat, loc_vec = get_activity_mat(data[key], param_dic)
                if param_dic["dr_method"] == "MDS":
                    mds = multi_dim_scaling(act_mat, param_dic)
                    plot_2D_scatter(ax[plot_ID, dat_ID], mds,param_dic)

    # 3D
    elif param_dic["dr_method_p2"] == 3:
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
                act_mat, loc_vec = get_activity_mat(data[key], param_dic)
                if param_dic["dr_method"] == "MDS":
                    mds = multi_dim_scaling(act_mat, param_dic)
                    ax = fig.add_subplot(nr_trials_to_compare, len(data_sets),
                                         (dat_ID*nr_trials_to_compare+plot_ID+1), projection='3d')
                    plot_3D_scatter(ax, mds, param_dic)



    fig.suptitle(param_dic["dr_method"]+" : "+param_dic["dr_method_p1"], fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()


def manifold_compare_conc(data_sets,loc_sets, param_dic):
# combines two data sets, reduces the dimension and plots sets in different colors in 2D/3D for one data set of each
# condition
    # get trial ID from both sets
    trial_ID_set1 = list(data_sets[0].keys())[param_dic["sel_trial"]]
    trial_ID_set2 = list(data_sets[1].keys())[param_dic["sel_trial"]]

    # calculate matrix of population vectors
    act_mat_set1, loc_vec1 = get_activity_mat(data_sets[0][trial_ID_set1], param_dic, loc_sets[0][trial_ID_set1])
    act_mat_set2, loc_vec2 = get_activity_mat(data_sets[1][trial_ID_set2], param_dic, loc_sets[1][trial_ID_set2])

    # combine both matrices
    comb_mat = np.hstack((act_mat_set1, act_mat_set2))
    # comb_loc = np.vstack((np.expand_dims(loc_vec1,1),np.expand_dims(loc_vec2,1)))

    # multi dimensional scaling
    # -------------------------------------------------------------------------------------------------------------------
    mds = multi_dim_scaling(comb_mat, param_dic)

    data_sep = act_mat_set1.shape[1]

    # 2D
    if param_dic["dr_method_p2"] == 2:
        fig, ax = plt.subplots()
        plot_2D_scatter(ax, mds, param_dic, data_sep)
    # 3D
    elif param_dic["dr_method_p2"] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_3D_scatter(ax, mds, param_dic, data_sep)

    fig.suptitle(param_dic["dr_method"] + " : " + param_dic["dr_method_p1"], fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()


def state_transition_analysis(data_sets, param_dic):
# analysis the state transitions using difference vectors between two population states for a selected trial

    nr_cells = len(next(iter(data_sets[0].values())))
    dat_mat = np.array([]).reshape(nr_cells,0)
    data_sep = np.inf
    for data_set in data_sets:
        trial_ID = list(data_set.keys())[param_dic["sel_trial"]]
        act_mat, loc_vec = get_activity_mat(data_set[trial_ID], param_dic)
        diff_mat = pop_vec_diff(act_mat)
        if len(data_sets) > 1:
        # combine datasets
            dat_mat = np.hstack((dat_mat, diff_mat))
            data_sep = min(act_mat.shape[1], data_sep)
        else:
            dat_mat = diff_mat
            data_sep = []


    mds = multi_dim_scaling(dat_mat, param_dic)
    # 2D
    if param_dic["dr_method_p2"] == 2:
        fig, ax = plt.subplots()
        plot_2D_scatter(ax, mds, data_sep, param_dic)
    # 3D
    elif param_dic["dr_method_p2"] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_3D_scatter(ax, mds, data_sep, param_dic)
    fig.suptitle(param_dic["dr_method"]+" : "+param_dic["dr_method_p1"],fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()
