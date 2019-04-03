########################################################################################################################
#
#   MANIFOLD METHODS
#
#   Description: contains classes that are used to perform manifold analysis
#
#   Author: Lars Bollmann
#
#   Created: 21/03/2019
#
#   Structure:
#
#       - Manifold: Base class containing basic attributes and methods
#
#               - reduce_dimension: reduces dimension using defined method
#
#               - plot_in_one_fig: plots results as scatter plot separating either trials (default) or rules
#
#       - singleManifold: Methods/attributes to analyze manifold for one condition (e.g. RULE A)
#
#               - concatenated_data:  using data from multiple trials for transformation (dim. reduction)
#                                               and separating data afterwards
#
#               - state_transition: analyzes the state transitions using difference vectors between two population
#                                   states
#
#       - ManifoldTransition: evaluates the manifold change e.g during rule switch
#
#               - separate_data_time_bins: using time bins, transforming every trial separately
#
#       - ManifoldCompare:  compares results of two different conditions using dimensionality reduction and transforming
#                           each trial separately
#
#               - all_trials: compares two conditions (e.g. RULE A vs. RULE B) using all available trials from the data
#
#               - selected_trials: compares two conditions (e.g. RULE A vs. RULE B) using one trial for each condition
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
from plotting_functions import plot_compare

########################################################################################################################
#   MANIFOLD BASE CLASS
########################################################################################################################

class Manifold():
    ''' Base class for manifold analysis'''

    def __init__(self, param_dic):

        # number of trials to compare
        self.nr_trials = param_dic["nr_of_trials"]

        # binning method
        self.binning_method = param_dic["binning_method"]

        # saving plot?
        self.save_plot = param_dic["save_plot"]

        # name for saving plot
        self.plot_file_name = param_dic["plot_file_name"]

        # dimensionality reduction method
        self.dr_method = param_dic["dr_method"]

        # dimensionality reduction method: parameter 1
        #  - multidimensional scaling: difference measure (jaccard,cos,etc)
        if param_dic["dr_method"] == "MDS" and not param_dic["dr_method_p1"]:
            print("Difference measure for MDS not defined!")
            exit()
        else:
            self.dr_method_p1 = param_dic["dr_method_p1"]

        # dimensionality reduciton method: parameter 2
        #  - for all methods: number of components
        self.dr_method_p2 = param_dic["dr_method_p2"]

        # result of dimensionality reduction
        self.result_dr = []

        # vector with concatenated location data
        self.loc_vec = np.empty((0,1))

        # parameter dictionary
        self.param_dic = param_dic

        # array with separator indices for different trials
        self.data_sep = []


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


    def plot_in_one_fig(self,rule_sep = []):
        # plots results as scatter plot separating either trials (default) or rules
        fig = plot_compare(self.result_dr, self.param_dic, self.data_sep, rule_sep)
        # save plot if option is set to true
        if self.save_plot:
            fig.savefig("plots/" + self.plot_file_name + ".png")


########################################################################################################################
#   SINGLE MANIFOLD ANALYSIS
########################################################################################################################

class SingleManifold(Manifold):
    ''' Methods for analyzing one manifold'''

    def __init__(self, data_set, loc_set, param_dic):

        # get attributes from parent class
        Manifold.__init__(self, param_dic)

        # dictionary with firing times for different cells and different trials
        self.data_set = data_set

        # dictionary with locations for different trials
        self.loc_set = loc_set

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

    def state_analysis(self):
        # using population vectors from multiple trials for transformation (dim. reduction) and separating data
        # afterwards --> using either temporal or spatial bins

        dat_mat = np.array([]).reshape(self.nr_cells,0)
        # nr of trials to compare
        trial_counter = 0
        # concatenate all trials and save separator for trials
        for i, key in enumerate(self.data_set):
            if trial_counter > (self.nr_trials_to_compare-1):
                break
            else:
                if self.binning_method == "temporal":
                    act_mat, loc_vec_part = get_activity_mat_time(self.data_set[key], self.param_dic,self.loc_set[key])
                elif self.binning_method == "spatial":
                    act_mat, loc_vec_part = get_activity_mat_spatial(self.data_set[key], self.param_dic,
                                                                     self.loc_set[key])
                # concatenate spike matrices
                dat_mat = np.hstack((dat_mat, act_mat))
                # concatenate location vectors
                self.loc_vec = np.vstack((self.loc_vec, np.expand_dims(loc_vec_part, 1)))
                self.data_sep[trial_counter + 1] = int(self.data_sep[trial_counter] + act_mat.shape[1])
                trial_counter += 1
        # apply dimensionality reduction to data
        self.reduce_dimension(dat_mat)

    def state_transition_analysis(self):
        # analysis the state transitions using difference vectors between two population states for a selected trial
        # difference matrices (e.g. matrix of difference vectors) are transformed together and separated afterwards

        dat_mat = np.array([]).reshape(self.nr_cells,0)
        # nr of trials to compare
        trial_counter = 0
        # concatenate all trials and save separator for trials
        for i, key in enumerate(self.data_set):
            if trial_counter > (self.nr_trials_to_compare-1):
                break
            else:
                if self.binning_method == "temporal":
                    act_mat, loc_vec_part = get_activity_mat_time(self.data_set[key], self.param_dic,self.loc_set[key])
                elif self.binning_method == "spatial":
                    act_mat, loc_vec_part = get_activity_mat_spatial(self.data_set[key], self.param_dic,
                                                                     self.loc_set[key])
                # calculate differences between population vectors
                diff_mat = pop_vec_diff(act_mat)
                # concatenate spike matrices
                dat_mat = np.hstack((dat_mat, diff_mat))
                # concatenate location vectors
                self.loc_vec = np.vstack((self.loc_vec, np.expand_dims(loc_vec_part, 1)))
                self.data_sep[trial_counter + 1] = int(self.data_sep[trial_counter] + act_mat.shape[1])
                trial_counter += 1

        # for jaccard distance --> make difference matrix signed binary
        if self.dr_method_p1 == "jaccard":
            dat_mat = np.sign(dat_mat)

        # apply dimensionality reduction to data
        self.reduce_dimension(dat_mat)
        self.plot_in_one_fig_color_position()




    def plot_in_multi_figs(self):
    # plots results as scatter plot in different figures

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

        fig.suptitle(self.dr_method+" : "+self.dr_method_p1+" , binning: "+self.binning_method, fontweight='bold')
        handles, labels = fig.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys())
        plt.show()
        # save plot if option is set to true
        if self.save_plot:
            fig.savefig("plots/"+self.plot_file_name+".png")


    def plot_in_one_fig_color_position(self):
    # plots results as scatter plot and colors lines according to location

        # 2D plot
        if self.dr_method_p2 == 2:
            # create figure instance
            fig, ax = plt.subplots()

            for data_ID in range(self.nr_trials_to_compare):
                data_subset = self.result_dr[int(self.data_sep[data_ID]):int(self.data_sep[data_ID+1]), :]
                loc_vec_subset = self.loc_vec[int(self.data_sep[data_ID]):int(self.data_sep[data_ID+1])]
                plot_2D_scatter(ax, data_subset, self.param_dic,[],loc_vec_subset)

        # 3D plot
        elif self.dr_method_p2 == 3:
            # create figure instance
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            for data_ID in range(self.nr_trials_to_compare):
                data = self.result_dr[int(self.data_sep[data_ID]):int(self.data_sep[data_ID+1]), :]
                plot_3D_scatter(ax, data, self.param_dic,[],self.loc_vec)

        fig.suptitle(self.dr_method+" : "+self.dr_method_p1, fontweight='bold')
        handles, labels = fig.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys())
        plt.show()
        # save plot if option is set to true
        if self.save_plot:
            fig.savefig("plots/"+self.plot_file_name+".png")

########################################################################################################################
#   MANIFOLD TRANSITION ANALYSIS
########################################################################################################################

class ManifoldTransition(SingleManifold):
    ''' Methods for analyzing manifold transition'''

    def __init__(self, data_set, loc_set, param_dic):

        # get attributes from parent class
        SingleManifold.__init__(self, data_set, loc_set, param_dic)

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


class ManifoldCompare(Manifold):
    ''' Methods for comparing manifolds with different conditions (rule) '''

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
        self.loc_vec = np.empty((0,1))

        # parameter dictionary
        self.param_dic = param_dic

        # data separator
        self.data_sep = np.empty((0, 1))

        # rule separator
        self.rule_sep = np.zeros(len(data_sets), dtype=int)

        # how many cells are in the data set
        self.nr_cells = len(next(iter(self.data_sets[0].values())))

    def state_analysis(self):
    # combines multiple data sets, reduces the dimension and plots sets in different colors in 2D/3D for one data set of each
    # condition

        # initialize data matrix
        dat_mat = np.array([]).reshape(self.nr_cells, 0)

        trial_counter = 0
        # go through all data sets
        for data_set_ID, (data_set,loc_set) in enumerate(zip(self.data_sets,self.loc_sets)):
            # concatenate all trials and save separator for trials
            data_set_data_sep = np.zeros(len(data_set.keys())+1)
            self.rule_sep[data_set_ID] = int(len(data_set.keys()))
            # add zero array to existing separator array
            self.data_sep = np.vstack((self.data_sep, np.expand_dims(data_set_data_sep, 1)))
            for i, key in enumerate(data_set):
                if self.binning_method == "temporal":
                    act_mat, loc_vec_part = get_activity_mat_time(data_set[key], self.param_dic,loc_set[key])
                elif self.binning_method == "spatial":
                    act_mat, loc_vec_part = get_activity_mat_spatial(data_set[key], self.param_dic,
                                                                     loc_set[key])
                # concatenate spike matrices
                dat_mat = np.hstack((dat_mat, act_mat))
                # concatenate location vectors
                self.loc_vec = np.vstack((self.loc_vec, np.expand_dims(loc_vec_part, 1)))
                self.data_sep[trial_counter + 1] = int(self.data_sep[trial_counter] + act_mat.shape[1])
                trial_counter += 1

        # remove last elements --> only contain zero
        self.data_sep = self.data_sep[:-1]
        self.rule_sep = self.rule_sep[:-1]
        # apply dimensionality reduction to data
        self.reduce_dimension(dat_mat)
        self.plot_in_one_fig(self.rule_sep[0])


    def state_analysis_selected_trial(self, trial_nr):
    # combines two data sets, reduces the dimension and plots sets in different colors in 2D/3D for one data set of each
    # condition
        # get trial ID from both sets
        trial_ID_set1 = list(self.data_sets[0].keys())[trial_nr]
        trial_ID_set2 = list(self.data_sets[1].keys())[trial_nr]

        # calculate matrix of population vectors
        act_mat_set1, loc_vec1 = get_activity_mat_time(self.data_sets[0][trial_ID_set1], self.param_dic, self.loc_sets[0][trial_ID_set1])
        act_mat_set2, loc_vec2 = get_activity_mat_time(self.data_sets[1][trial_ID_set2], self.param_dic, self.loc_sets[1][trial_ID_set2])

        # combine both matrices
        comb_mat = np.hstack((act_mat_set1, act_mat_set2))
        # comb_loc = np.vstack((np.expand_dims(loc_vec1,1),np.expand_dims(loc_vec2,1)))

        # multi dimensional scaling
        # -------------------------------------------------------------------------------------------------------------------
        mds = multi_dim_scaling(comb_mat, self.param_dic)

        data_sep = act_mat_set1.shape[1]

        # 2D
        if self.param_dic["dr_method_p2"] == 2:
            fig, ax = plt.subplots()
            plot_2D_scatter(ax, mds, self.param_dic, data_sep)
        # 3D
        elif self.param_dic["dr_method_p2"] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plot_3D_scatter(ax, mds, self.param_dic, data_sep)

        fig.suptitle(self.param_dic["dr_method"] + " : " + self.param_dic["dr_method_p1"], fontweight='bold')
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

    def state_transition_analysis(self):
        # analysis the state transitions using difference vectors between two population states for a selected trial
        # difference matrices (e.g. matrix of difference vectors) are transformed together and separated afterwards

        dat_mat = np.array([]).reshape(self.nr_cells, 0)
        # nr of trials to compare
        trial_counter = 0


        # go through all data sets
        for data_set_ID, (data_set,loc_set) in enumerate(zip(self.data_sets,self.loc_sets)):
            # concatenate all trials and save separator for trials
            data_set_data_sep = np.zeros(len(data_set.keys())+1)
            self.rule_sep[data_set_ID] = int(len(data_set.keys()))
            # add zero array to existing separator array
            self.data_sep = np.vstack((self.data_sep, np.expand_dims(data_set_data_sep, 1)))
            for i, key in enumerate(data_set):
                if self.binning_method == "temporal":
                    act_mat, loc_vec_part = get_activity_mat_time(data_set[key], self.param_dic,loc_set[key])
                elif self.binning_method == "spatial":
                    act_mat, loc_vec_part = get_activity_mat_spatial(data_set[key], self.param_dic,
                                                                     loc_set[key])
                # calculate differences between population vectors
                diff_mat = pop_vec_diff(act_mat)
                # concatenate spike matrices
                dat_mat = np.hstack((dat_mat, diff_mat))
                # concatenate location vectors
                self.loc_vec = np.vstack((self.loc_vec, np.expand_dims(loc_vec_part, 1)))
                self.data_sep[trial_counter + 1] = int(self.data_sep[trial_counter] + act_mat.shape[1])
                trial_counter += 1

        # remove last elements --> only contain zero
        self.data_sep = self.data_sep[:-1]
        self.rule_sep = self.rule_sep[:-1]

        # for jaccard distance --> make difference matrix signed binary
        if self.dr_method_p1 == "jaccard":
            dat_mat = np.sign(dat_mat)

        # apply dimensionality reduction to data
        self.reduce_dimension(dat_mat)
        self.plot_in_one_fig_color_position()
        self.plot_in_one_fig(self.rule_sep[0])

    def plot_in_one_fig_color_position(self):
    # plots results as scatter plot and colors lines according to location

        # 2D plot
        if self.dr_method_p2 == 2:
            # create figure instance
            fig, ax = plt.subplots()
            data_sep = self.data_sep[self.rule_sep[0]+1]+1
            plot_2D_scatter(ax, self.result_dr, self.param_dic,data_sep,self.loc_vec)


        # 3D plot
        elif self.dr_method_p2 == 3:
            # create figure instance
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            data_sep = self.data_sep[self.rule_sep[0]+1]+1
            plot_3D_scatter(ax, self.result_dr, self.param_dic,data_sep,self.loc_vec)

        fig.suptitle(self.dr_method+" : "+self.dr_method_p1, fontweight='bold')
        handles, labels = fig.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys())
        plt.show()
        # save plot if option is set to true
        if self.save_plot:
            fig.savefig("plots/"+self.plot_file_name+".png")