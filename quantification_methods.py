########################################################################################################################
#
#   QUANTIFICATION METHODS
#
#   Description: contains classes that are used to quantify results from manifold analysis
#
#   Author: Lars Bollmann
#
#   Created: 04/04/2019
#
#   Structure:
#
#
########################################################################################################################

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from collections import OrderedDict
from comp_functions import get_activity_mat_time
from comp_functions import get_activity_mat_spatial
from comp_functions import calc_diff
from comp_functions import multi_dim_scaling
from comp_functions import pop_vec_diff
from comp_functions import perform_PCA
from comp_functions import perform_TSNE
from comp_functions import calc_cohens_d

import matplotlib.cm as cm
from plotting_functions import plot_2D_scatter
from plotting_functions import plot_3D_scatter
from plotting_functions import plot_compare
import seaborn as sns
import matplotlib as mpl
import os
from collections import OrderedDict

# set saving directory to current directory
mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))

########################################################################################################################
#   ANALYSIS BASE CLASS
########################################################################################################################

class Analysis():
    ''' Base class for quantitative analysis'''

    def __init__(self, param_dic):

        # binning method
        self.binning_method = param_dic["binning_method"]

        # saving plot?
        self.save_plot = param_dic["save_plot"]

        # name for saving plot
        self.plot_file_name = param_dic["plot_file_name"]

        # vector with concatenated location data
        self.loc_vec = np.empty((0, 1))

        # parameter dictionary
        self.param_dic = param_dic

        # array with separator indices for different trials
        self.data_sep = []


class TransitionAnalysis(Analysis):
    ''' Base class for quantitative analysis'''

    def __init__(self, data_set, loc_set, param_dic, new_rule_trial):

        # get attributes from parent class
        Analysis.__init__(self, param_dic)

        # dictionary with firing times for different cells and different trials
        self.data_set = data_set

        # dictionary with locations for different trials
        self.loc_set = loc_set

        # first trial with new rule
        self.new_rule_trial = new_rule_trial

        # how many cells are in the data set
        self.nr_cells = len(next(iter(self.data_set.values())))

        # how many trials to compare
        self.nr_trials_to_compare = len(self.data_set.keys())

        # array with separator indices for different trials
        self.data_sep = np.zeros(self.nr_trials_to_compare+1)

    def create_save_spatial_bin_dictionary(self):
        # create "activity matrices" consisting of population vectors for each rule

        act_mat, _ = get_activity_mat_spatial(next(iter(self.data_set.values())),
                                              self.param_dic, next(iter(self.loc_set.values())))
        # how many intervals
        nr_intervals = act_mat.shape[1]
        act_mat = []

        # initialize dictionaries --> each entry in dictionary is a spatial bin --> contains all population vectors
        # (from different trials) for this spatial interval
        dic_spat_int_1 = {}
        dic_spat_int_2 = {}

        for spat_int in range(nr_intervals):
            dic_spat_int_1["INT"+str(spat_int)] = dic_spat_int_2["INT"+str(spat_int)] = \
                np.array([]).reshape(self.nr_cells,0)

        for trial, key in enumerate(self.data_set):
            act_mat, loc_vec_part = get_activity_mat_spatial(self.data_set[key], self.param_dic,
                                                             self.loc_set[key])
            # for spatial binning we can get the number of intervals from the shape of act_mat
            nr_intervals = act_mat.shape[1]

            # separate both rules
            if trial < (self.new_rule_trial-1):
                # rule 1
                # write elements in dictionary
                for int_count, key in enumerate(dic_spat_int_1):
                    dic_spat_int_1[key] = np.hstack((dic_spat_int_1[key], np.expand_dims(act_mat[:,int_count],1)))
            else:
                # write elements in dictionary
                for int_count, key in enumerate(dic_spat_int_2):
                    dic_spat_int_2[key] = np.hstack((dic_spat_int_2[key], np.expand_dims(act_mat[:,int_count],1)))

        # save first dictionary as pickle
        filename = "temp_data/quant_analysis/" +"SWITCH_"+ self.param_dic["data_descr"][0] + "_spatial_"\
                   + str(self.param_dic["spatial_bin_size"])
        outfile = open(filename, 'wb')
        pickle.dump(dic_spat_int_1, outfile)
        outfile.close()

        # save second dictionary as pickle
        filename = "temp_data/quant_analysis/" +"SWITCH_"+ self.param_dic["data_descr"][1] + "_spatial_" + \
                   str(self.param_dic["spatial_bin_size"])
        outfile = open(filename, 'wb')
        pickle.dump(dic_spat_int_2, outfile)
        outfile.close()

    def cross_cos_diff_spat_trials(self, spat_bin_dic_1, spat_bin_dic_2):

        if len(spat_bin_dic_1.keys()) != len(spat_bin_dic_2.keys()):
            print("Number of spatial bins in both dictionaries don't match")
            exit()

        nr_trials_after_switch = next(iter(spat_bin_dic_2.values())).shape[1]
        nr_trials_before_switch = next(iter(spat_bin_dic_1.values())).shape[1]
        nr_intervals = len(spat_bin_dic_1.keys())
        x_axis = np.arange(0,200,self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        col_map = cm.rainbow(np.linspace(0, 1, nr_trials_after_switch))

        # go through all trials after rule switch and compare each one with rule before rule switch
        # --> each entry in dictionary is a spatial bin --> contains all population vectors

        for trial_after_switch in range(nr_trials_after_switch):
            cross_diff = np.zeros((nr_intervals,nr_trials_before_switch))

            # go through all spatial bins and calculate difference --> rows: spatial bins, columns: cos distances
            for i, key in enumerate(spat_bin_dic_2):
                cross_diff[i, :] = calc_diff(spat_bin_dic_1[key], np.expand_dims(spat_bin_dic_2[key][:,trial_after_switch],1),"cos").flatten()

            plt.subplot(2,1,1)
            plt.plot(x_axis, np.average(cross_diff,1), color= col_map[trial_after_switch,:], marker= "o", label="TRIAL "+str(trial_after_switch))
            plt.legend()
            plt.grid()
            plt.xlim([min(x_axis), max(x_axis) + 20])
            plt.title("DIFFERENCE BETWEEN SINGLE TRIALS AFTER RULE SWITCH AND ALL TRIALS BEFORE RULE SWITCH")
            plt.xlabel("MAZE POSITION")
            plt.ylabel("AVERAGE COS DIFFERENCE")

        # go through all spatial bins and see how the difference changes with trials after rule switch
        # --> each entry in dictionary is a spatial bin --> contains all population vectors

        col_map = cm.tab20(np.linspace(0, 1, len(x_axis)))
        spat_position = x_axis
        x_axis = np.arange(nr_trials_after_switch)

        for i, key in enumerate(spat_bin_dic_2):
            cross_diff = np.zeros((nr_trials_after_switch, nr_trials_before_switch))
            for trial_after_switch in range(nr_trials_after_switch):
                cross_diff[trial_after_switch, :] = calc_diff(spat_bin_dic_1[key], np.expand_dims(spat_bin_dic_2[key][:,trial_after_switch],1),"cos").flatten()

            plt.subplot(2, 1, 2)
            plt.plot(x_axis, np.average(cross_diff,1), color= col_map[i,:],marker= "o", label=str(spat_position[i])+" cm")
            plt.legend()
            plt.title("DIFFERENCE BETWEEN SINGLE TRIALS AFTER RULE SWITCH AND ALL TRIALS BEFORE RULE SWITCH")
            plt.xlabel("TRIAL AFTER RULE SWITCH")
            plt.xlim([0, nr_trials_after_switch + 1])
            plt.ylabel("AVERAGE COS DIFFERENCE")
            plt.grid()

        plt.show()

    def cross_cos_diff_spat(self, spat_bin_dic_1, spat_bin_dic_2):

        if len(spat_bin_dic_1.keys()) != len(spat_bin_dic_2.keys()):
            print("Number of spatial bins in both dictionaries don't match")
            exit()

        nr_intervals = len(spat_bin_dic_1.keys())
        nr_comparisons = next(iter(spat_bin_dic_1.values())).shape[1] * next(iter(spat_bin_dic_2.values())).shape[1]

        cross_diff = np.zeros((nr_intervals, nr_comparisons))
        n_1 = next(iter(spat_bin_dic_1.values())).shape[1]
        n_2 = next(iter(spat_bin_dic_2.values())).shape[1]
        within_diff_1 = np.zeros((nr_intervals, int(n_1*(n_1 - 1)/2)))
        within_diff_2 = np.zeros((nr_intervals, int(n_2*(n_2 - 1)/2)))
        stats_array = np.zeros((nr_intervals, 2))

        # go through all spatial bins and calculate difference --> rows: spatial bins, columns: cos distances
        for i, key in enumerate(spat_bin_dic_1):
            cross_diff[i, :] = calc_diff(spat_bin_dic_1[key], spat_bin_dic_2[key], "cos").flatten()
            temp1 = calc_diff(spat_bin_dic_1[key], spat_bin_dic_1[key], "cos")
            within_diff_1[i,:] = temp1[np.triu_indices(temp1.shape[0],1)]
            temp2 = calc_diff(spat_bin_dic_2[key], spat_bin_dic_2[key], "cos")
            within_diff_2[i,:] = temp2[np.triu_indices(temp2.shape[0],1)]

        # for each spatial bin compare union of within_diff_1/within_diff_2 and cross_diff
        for i, (w_d_1, w_d_2, c_d) in enumerate(zip(within_diff_1, within_diff_2,cross_diff)):
            w_d = np.hstack((w_d_1, w_d_2))
            stats_array[i,0],stats_array[i,1] = stats.kruskal(w_d, c_d)
            # compare

        x_axis = np.arange(0,200,self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        avg_1 = np.average(within_diff_1, axis=1)
        err_1 = stats.sem(within_diff_1, axis=1)

        avg_2 = np.average(within_diff_2, axis=1)
        err_2 = stats.sem(within_diff_2, axis=1)

        avg = np.average(cross_diff, axis=1)
        err = stats.sem(cross_diff, axis=1)

        plt.subplot(2,2,1)
        plt.errorbar(x_axis,avg,yerr = err,fmt="o")
        plt.grid()
        plt.xlabel("MAZE LOCATION / cm")
        plt.ylabel("AVERAGE DISTANCE (COS) - AVG & SEM")
        plt.title("ABSOLUTE")

        # add significance marker
        for i,p_v in enumerate(stats_array[:,1]):
            print(p_v)
            if p_v < 0.05:
                plt.scatter(x_axis[i]+2,avg[i]+0.02, marker="*", edgecolors="Red",label="K-W, 0.05")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.subplot(2, 2, 2)
        plt.scatter(x_axis, avg/avg_1)
        plt.xlabel("MAZE LOCATION / cm")
        plt.ylabel("AVERAGE DISTANCE (COS)")
        plt.title("NORMALIZED BY RULE 1")
        plt.grid()

        # add significance marker
        for i,p_v in enumerate(stats_array[:,1]):
            print(p_v)
            if p_v < 0.05:
                plt.scatter(x_axis[i]+2,avg[i]/avg_1[i]+0.05, marker="*", edgecolors="Red",label="K-W, 0.05")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.subplot(2, 2, 3)
        plt.scatter(x_axis, avg/avg_2)
        plt.xlabel("MAZE LOCATION / cm")
        plt.ylabel("AVERAGE DISTANCE (COS)")
        plt.title("NORMALIZED BY RULE 2")
        plt.grid()

        # add significance marker
        for i,p_v in enumerate(stats_array[:,1]):
            print(p_v)
            if p_v < 0.05:
                plt.scatter(x_axis[i]+2,avg[i]/avg_2[i]+0.05, marker="*", edgecolors="Red",label="K-W, 0.05")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())


        plt.subplot(2, 2, 4)
        plt.scatter(x_axis, stats_array[:,1])
        plt.hlines(0.05,min(x_axis),max(x_axis),colors="Red",label="0.05")
        plt.xlabel("MAZE LOCATION / cm")
        plt.ylabel("P-VALUE")
        plt.title("KRUSKAL: WITHIN-RULE vs. ACROSS-RULES")
        plt.legend()
        plt.grid()

        plt.show()


class ComparisonAnalysis(Analysis):
    ''' Base class for quantitative analysis'''

    def __init__(self, data_sets, loc_sets, param_dic):

        # get attributes from parent class
        Analysis.__init__(self, param_dic)

        # dictionary with firing times for different cells and different trials
        self.data_sets = data_sets

        # dictionary with locations for different trials
        self.loc_sets = loc_sets

        # how many cells are in the data set
        self.nr_cells = len(next(iter(self.data_sets[0].values())))

    def create_save_spatial_bin_dictionary(self):
        # create "activity matrices" consisting of population vectors for each rule

        act_mat, _ = get_activity_mat_spatial(next(iter(self.data_sets[0].values())),
                                              self.param_dic, next(iter(self.loc_sets[0].values())))
        # how many intervals
        nr_intervals = act_mat.shape[1]
        act_mat = []

        # go through both data sets
        for i, (data_set, loc_set) in enumerate(zip(self.data_sets,self.loc_sets)):

            # initialize dictionary --> each entry in dictionary is a spatial bin --> contains all population vectors
            # (from different trials) for this spatial interval
            dic_spat_int = {}

            for spat_int in range(nr_intervals):
                dic_spat_int["INT" + str(spat_int)] = np.array([]).reshape(self.nr_cells, 0)

            # go through all trials
            for trial, key in enumerate(data_set):
                act_mat, loc_vec_part = get_activity_mat_spatial(data_set[key], self.param_dic,
                                                                 loc_set[key])
                # write elements in dictionary
                for int_count, key in enumerate(dic_spat_int):
                    dic_spat_int[key] = np.hstack((dic_spat_int[key], np.expand_dims(act_mat[:,int_count],1)))

            # save first dictionary as pickle
            filename = "temp_data/quant_analysis/"+self.param_dic["data_descr"][i] +"_spatial_"+ \
                       str(self.param_dic["spatial_bin_size"])
            outfile = open(filename, 'wb')
            pickle.dump(dic_spat_int, outfile)
            outfile.close()
