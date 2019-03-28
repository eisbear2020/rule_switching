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
#       - ManifoldTransition: evaluates the manifold change e.g during rule switch
#
#               - separate_data_time_bins: using time bins, transforming every trial separately
#               - concatenated_data_time_bins:  using data from multiple trials for transformation (dim. reduction)
#                                               and separating data afterwards
#
#
#       - ManifoldCompare:  compares results of two different conditions using dimensionality reduction and transforming
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
from comp_functions import get_activity_mat_time
from comp_functions import get_activity_mat_spatial
from comp_functions import multi_dim_scaling
from comp_functions import pop_vec_diff
from comp_functions import perform_PCA
from comp_functions import perform_TSNE

import matplotlib.cm as cm
from plotting_functions import plot_2D_scatter
from plotting_functions import plot_3D_scatter


class Manifold:
    ''' Base class for manifold methods'''

    def __init__(self, param_dic):
        # number of trials to compare
        self.nr_trials = param_dic["nr_of_trials"]

        # dimensionality reduction method
        self.dr_method = param_dic["dr_method"]

        # dimensionality reduction method: parameter 1
        #  - multidimensional scaling: difference measure (jaccard,cos,etc)
        self.dr_method_p1 = param_dic["dr_method_p1"]

        # dimensionality reduciton method: parameter 2
        #  - for all methods: number of components
        self.dr_method_p2 = param_dic["dr_method_p2"]

        # saving plot?
        self.save_plot = param_dic["save_plot"]

        # name for saving plot
        self.plot_file_name = param_dic["plot_file_name"]


class ManifoldTransition(Manifold):
    ''' Methods for analyzing manifold transition'''

    def __init__(self, data_set, loc_set, param_dic):

        # get attributes from parent class
        Manifold.__init__(self, param_dic)

        # dictionary with firing times for different cells and different trials
        self.data_set = data_set

        # dictionary with locations for different trials
        self.loc_set = loc_set

        # result of dimensionality reduction
        self.result_dr = []

        # array with location data
        self.loc_vec = []

        # parameter dictionary
        self.param_dic = param_dic

        # how many cells are in the data set
        self.nr_cells = len(next(iter(self.data_set.values())))

        # how many trials to compare
        self.nr_trials_to_compare = min(len(self.data_set.keys()), self.nr_trials)

        # array with separator indices for different trials
        self.data_sep = np.zeros(self.nr_trials_to_compare+1)

        # check how many columns should be used for plotting: 3 col, 2 col or 1 col
        if not np.mod(self.nr_trials_to_compare, 3):
            self.c_p = 3
        elif not np.mod(self.nr_trials_to_compare, 2):
            self.c_p = 2
        else:
            self.c_p = 1

    def concatenated_data_time_bins(self):
    # change of the manifold e.g. during rule switching, using data from multiple trials for transformation (dim. reduction)
    # and separating data afterwards

        dat_mat = np.array([]).reshape(self.nr_cells,0)
        # vector with concatenated location data
        self.loc_vec = np.empty((0,1))
        # nr of trials to compare
        trial_counter = 0
        # concatenate all trials and save separator for trials
        for i, key in enumerate(self.data_set):
            if trial_counter > (self.nr_trials_to_compare-1):
                break
            else:
                act_mat, loc_vec_part = get_activity_mat_time(self.data_set[key], self.param_dic,self.loc_set[key])
                # concatenate spike matrices
                dat_mat = np.hstack((dat_mat, act_mat))
                # concatenate location vectors
                self.loc_vec = np.vstack((self.loc_vec, np.expand_dims(loc_vec_part, 1)))
                self.data_sep[trial_counter + 1] = int(self.data_sep[trial_counter] + act_mat.shape[1])
                trial_counter += 1
        # apply dimensionality reduction to data
        self.reduce_dimension(dat_mat)
        # plot the results
        self.plot_results()


    def concatenated_data_spatial_bins(self):
    # change of the manifold e.g. during rule switching, using data from multiple trials for transformation (dim. reduction)
    # and separating data afterwards

        dat_mat = np.array([]).reshape(self.nr_cells,0)
        # vector with concatenated location data
        self.loc_vec = np.empty((0,1))
        # nr of trials to compare
        trial_counter = 0
        # concatenate all trials and save separator for trials
        for i, key in enumerate(self.data_set):
            if trial_counter > (self.nr_trials_to_compare-1):
                break
            else:
                act_mat, loc_vec_part = get_activity_mat_spatial(self.data_set[key], self.param_dic,self.loc_set[key])
                # concatenate spike matrices
                dat_mat = np.hstack((dat_mat, act_mat))
                # concatenate location vectors
                self.loc_vec = np.vstack((self.loc_vec, np.expand_dims(loc_vec_part, 1)))
                self.data_sep[trial_counter + 1] = int(self.data_sep[trial_counter] + act_mat.shape[1])
                trial_counter += 1
        # apply dimensionality reduction to data
        self.reduce_dimension(dat_mat)
        # plot the results
        # self.plot_results()
        self.plot_combined_mark_position()

    def separate_data_time_bins(self):
        # analyses the transition of the manifold e.g. for the rule switch case. Each trial is transformed individually

        # 2D
        if self.dr_method_p2 == 2:
            # create figure instance
            fig, ax = plt.subplots(int(self.nr_trials_to_compare / self.c_p), self.c_p)
            # row for plot
            c_r = 0

            for plot_ID, key in enumerate(self.data_set):
                if not np.mod(plot_ID, self.c_p):
                    c_r += 1
                # compute equal number of trials for all conditions (for plotting)
                if plot_ID > (self.nr_trials_to_compare - 1):
                    break
                act_mat, loc_vec = get_activity_mat_time(self.data_set[key], self.param_dic,self.loc_set[key])
                self.reduce_dimension(act_mat)
                plot_2D_scatter(ax[c_r - 1, np.mod(plot_ID, self.c_p)], self.result_dr, self.param_dic)

        # 3D
        elif self.dr_method_p2 == 3:
            # create figure instance
            fig = plt.figure()

            for plot_ID, key in enumerate(self.data_set):
                # compute equal number of trials for all conditions (for plotting)
                if plot_ID > (self.nr_trials_to_compare - 1):
                    break
                act_mat, loc_vec = get_activity_mat_time(self.data_set[key], self.param_dic,self.loc_set[key])
                self.reduce_dimension(act_mat)
                ax = fig.add_subplot(int(self.nr_trials_to_compare / self.c_p), self.c_p, plot_ID + 1, projection='3d')
                plot_3D_scatter(ax, self.result_dr, self.param_dic)

        fig.suptitle(self.dr_method + " : " + self.dr_method_p1, fontweight='bold')
        handles, labels = fig.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys())
        plt.show()



    def reduce_dimension(self,dat_mat):
    # reduces the dimension using one of the defined methods

        # clear in case there was a previous result
        self.result_dr = []
        # dimensionality reduction: multi dimensional scaling
        if self.dr_method == "MDS":
            self.result_dr = multi_dim_scaling(dat_mat, self.param_dic)
        # dimensionality reduction: principal component analysis
        elif self.dr_method == "PCA":
            self.result_dr = perform_PCA(dat_mat, self.param_dic)
        elif self.dr_method == "TSNE":
            self.result_dr = perform_TSNE(dat_mat, self.param_dic)


    def plot_results(self):
    # plots results as scatter plot

        # row for plot
        c_r = 0
        # 2D plot
        if self.dr_method_p2 == 2:
            # create figure instance
            fig, ax = plt.subplots(int(self.nr_trials_to_compare / self.c_p), self.c_p)
            # if one-dimensional (that means only one column for plotting) --> expand dims
            if ax.ndim == 1:
                ax = np.expand_dims(ax, axis=1)

            for data_ID in range(self.nr_trials_to_compare):
                if not np.mod(data_ID, self.c_p):
                    c_r += 1
                data_subset = self.result_dr[int(self.data_sep[data_ID]):int(self.data_sep[data_ID+1]), :]
                loc_vec_subset = self.loc_vec[int(self.data_sep[data_ID]):int(self.data_sep[data_ID+1])]
                plot_2D_scatter(ax[c_r-1, np.mod(data_ID,self.c_p)], data_subset, self.param_dic,[],loc_vec_subset)

        # 3D plot
        elif self.dr_method_p2 == 3:
            # create figure instance
            fig = plt.figure()

            for data_ID in range(self.nr_trials_to_compare):
                if not np.mod(data_ID, self.c_p):
                    c_r += 1
                data = self.result_dr[int(self.data_sep[data_ID]):int(self.data_sep[data_ID+1]), :]
                ax = fig.add_subplot(int(self.nr_trials_to_compare / self.c_p), self.c_p, data_ID+1, projection='3d')
                plot_3D_scatter(ax, data, self.param_dic,[],self.loc_vec)

        fig.suptitle(self.dr_method+" : "+self.dr_method_p1, fontweight='bold')
        handles, labels = fig.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys())
        plt.show()
        # save plot if option is set to true
        if self.save_plot:
            fig.savefig("plots/"+self.plot_file_name+".png")

    def plot_combined_mark_trials(self):
        # plots results as scatter plot

        # row for plot
        c_r = 0
        # 2D plot
        if self.dr_method_p2 == 2:
            # create figure instance
            fig, ax = plt.subplots()

            col_map = cm.rainbow(np.linspace(0, 1, len(self.data_sep)))

            for data_ID in range(self.nr_trials_to_compare):
                if not np.mod(data_ID, self.c_p):
                    c_r += 1
                data_subset = self.result_dr[int(self.data_sep[data_ID]):int(self.data_sep[data_ID + 1]), :]
                loc_vec_subset = self.loc_vec[int(self.data_sep[data_ID]):int(self.data_sep[data_ID + 1])]

                for i in range(0, data_subset.shape[0] - 1):
                    ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], color=col_map[data_ID, :],
                            label="trial " + str(data_ID))
                ax.scatter(data_subset[:, 0], data_subset[:, 1], color="grey")
                ax.scatter(data_subset[0, 0], data_subset[0, 1], color="black", marker="x", label="start", zorder=200)
                ax.scatter(data_subset[-1, 0], data_subset[-1, 1], color="black", label="end", zorder=200)





    def plot_combined_mark_position(self):
    # plots results as scatter plot

        # row for plot
        c_r = 0
        # 2D plot
        if self.dr_method_p2 == 2:
            # create figure instance
            fig, ax = plt.subplots()

            for data_ID in range(self.nr_trials_to_compare):
                if not np.mod(data_ID, self.c_p):
                    c_r += 1
                data_subset = self.result_dr[int(self.data_sep[data_ID]):int(self.data_sep[data_ID+1]), :]
                loc_vec_subset = self.loc_vec[int(self.data_sep[data_ID]):int(self.data_sep[data_ID+1])]
                plot_2D_scatter(ax, data_subset, self.param_dic,[],loc_vec_subset)

        # 3D plot
        elif self.dr_method_p2 == 3:
            # create figure instance
            fig = plt.figure()

            for data_ID in range(self.nr_trials_to_compare):
                if not np.mod(data_ID, self.c_p):
                    c_r += 1
                data = self.result_dr[int(self.data_sep[data_ID]):int(self.data_sep[data_ID+1]), :]
                ax = fig.add_subplot(int(self.nr_trials_to_compare / self.c_p), self.c_p, data_ID+1, projection='3d')
                plot_3D_scatter(ax, data, self.param_dic,[],self.loc_vec)

        fig.suptitle(self.dr_method+" : "+self.dr_method_p1, fontweight='bold')
        handles, labels = fig.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys())
        plt.show()
        print(self.save_plot)
        # save plot if option is set to true
        if self.save_plot:
            fig.savefig("plots/"+self.plot_file_name+".png")




class ManifoldCompare(Manifold):
    ''' Methods for analyzing manifold transition'''

    def __init__(self, data_sets, loc_sets, param_dic):

        # get attributes from parent class
        Manifold.__init__(self, param_dic)

        # dictionary with firing times for different cells and different trials
        self.data_sets = data_sets

        # dictionary with locations for different trials
        self.loc_sets = loc_sets

        # result of dimensionality reduction
        self.result_dr = []

        # array with location data
        self.loc_vec = []

        # parameter dictionary
        self.param_dic = param_dic

        # how many cells are in the data set
        self.nr_cells = len(next(iter(self.data_set.values())))

        # how many trials to compare
        self.nr_trials_to_compare = min(len(self.data_set.keys()), self.nr_trials)

        # array with separator indices for different trials
        self.data_sep = np.zeros(self.nr_trials_to_compare+1)

        # check how many columns should be used for plotting: 3 col, 2 col or 1 col
        if not np.mod(self.nr_trials_to_compare, 3):
            self.c_p = 3
        elif not np.mod(self.nr_trials_to_compare, 2):
            self.c_p = 2
        else:
            self.c_p = 1


    def manifold_compare_conc(self):
    # combines two data sets, reduces the dimension and plots sets in different colors in 2D/3D for one data set of each
    # condition
        # get trial ID from both sets
        trial_ID_set1 = list(data_sets[0].keys())[param_dic["sel_trial"]]
        trial_ID_set2 = list(data_sets[1].keys())[param_dic["sel_trial"]]

        # calculate matrix of population vectors
        act_mat_set1, loc_vec1 = get_activity_mat_time(data_sets[0][trial_ID_set1], param_dic, loc_sets[0][trial_ID_set1])
        act_mat_set2, loc_vec2 = get_activity_mat_time(data_sets[1][trial_ID_set2], param_dic, loc_sets[1][trial_ID_set2])

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

        def manifold_compare(data_sets, param_dic):
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
                for dat_ID, (data, data_set_desc) in enumerate(zip(data_sets, param_dic["data_descr"])):
                    ax[0, dat_ID].set_title(data_set_desc)
                    # go over several trials
                    for plot_ID, key in enumerate(data):
                        # compute equal number of trials for all conditions (for plotting)
                        if plot_ID > (nr_trials_to_compare - 1):
                            break
                        act_mat, loc_vec = get_activity_mat_time(data[key], param_dic)
                        if param_dic["dr_method"] == "MDS":
                            mds = multi_dim_scaling(act_mat, param_dic)
                            plot_2D_scatter(ax[plot_ID, dat_ID], mds, param_dic)

            # 3D
            elif param_dic["dr_method_p2"] == 3:
                # create figure instance
                fig = plt.figure()

                # go over all data sets
                for dat_ID, (data, data_set_desc) in enumerate(zip(data_sets, param_dic["data_descr"])):
                    # ax[0, dat_ID].set_title(data_set_desc)
                    # go over several trials
                    for plot_ID, key in enumerate(data):
                        # compute equal number of trials for all conditions (for plotting)
                        if plot_ID > (nr_trials_to_compare - 1):
                            break
                        act_mat, loc_vec = get_activity_mat_time(data[key], param_dic)
                        if param_dic["dr_method"] == "MDS":
                            mds = multi_dim_scaling(act_mat, param_dic)
                            ax = fig.add_subplot(nr_trials_to_compare, len(data_sets),
                                                 (dat_ID * nr_trials_to_compare + plot_ID + 1), projection='3d')
                            plot_3D_scatter(ax, mds, param_dic)

            fig.suptitle(param_dic["dr_method"] + " : " + param_dic["dr_method_p1"], fontweight='bold')
            handles, labels = fig.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys())
            plt.show()


def state_transition_analysis(data_sets, loc_set, param_dic):
# analysis the state transitions using difference vectors between two population states for a selected trial

    nr_cells = len(next(iter(data_sets[0].values())))
    dat_mat = np.array([]).reshape(nr_cells,0)
    # vector with concatenated location data
    loc_vec = np.empty((0, 1))
    data_sep = np.inf
    for data_set in data_sets:
        trial_ID = list(data_set.keys())[param_dic["sel_trial"]]
        act_mat, loc_vec_part = get_activity_mat_time(data_set[trial_ID], param_dic,loc_set[trial_ID])
        diff_mat = pop_vec_diff(act_mat)
        if len(data_sets) > 1:
        # combine datasets
            dat_mat = np.hstack((dat_mat, diff_mat))
            data_sep = min(act_mat.shape[1], data_sep)

        else:
            dat_mat = diff_mat
            loc_vec = np.expand_dims(loc_vec_part[:-1],1)
            data_sep = []

    mds = multi_dim_scaling(dat_mat, param_dic)
    # 2D
    if param_dic["dr_method_p2"] == 2:
        fig, ax = plt.subplots()
        plot_2D_scatter(ax, mds, param_dic, [], loc_vec)
    # 3D
    elif param_dic["dr_method_p2"] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_3D_scatter(ax, mds, param_dic, [], loc_vec)
    fig.suptitle(param_dic["dr_method"]+" : "+param_dic["dr_method_p1"],fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()

    # save plot if option is set to true
    if param_dic["save_plot"]:
        fig.savefig("plots/" + param_dic["plot_file_name"] + ".png")
