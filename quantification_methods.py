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
#   TODO: change structure: only need .res and .whl for dictionary cration --> dont use for initialization of class
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
from plotting_functions import plot_remapping_summary
from plotting_functions import plot_cell_charact
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

        # storing cross-diff
        self.cross_diff = []

        self.within_diff_1 = []

        self.within_diff_2 = []
        # results of test statistic
        self.stats_array = []

    def cross_cos_diff(self, bin_dic_1, bin_dic_2):
        # calculates the pair-wise cos difference within each set and across both sets

        if len(bin_dic_1.keys()) != len(bin_dic_2.keys()):
            print("Number of bins in both dictionaries don't match")
            exit()

        nr_intervals = len(bin_dic_1.keys())
        nr_comparisons = next(iter(bin_dic_1.values())).shape[1] * next(iter(bin_dic_2.values())).shape[1]

        cross_diff = np.zeros((nr_intervals, nr_comparisons))
        n_1 = next(iter(bin_dic_1.values())).shape[1]
        n_2 = next(iter(bin_dic_2.values())).shape[1]
        within_diff_1 = np.zeros((nr_intervals, int(n_1*(n_1 - 1)/2)))
        within_diff_2 = np.zeros((nr_intervals, int(n_2*(n_2 - 1)/2)))
        stats_array = np.zeros((nr_intervals, 2))

        # go through all spatial bins and calculate difference --> rows: spatial bins, columns: cos distances
        for i, key in enumerate(bin_dic_1):
            cross_diff[i, :] = calc_diff(bin_dic_1[key], bin_dic_2[key], "cos").flatten()
            temp1 = calc_diff(bin_dic_1[key], bin_dic_1[key], "cos")
            within_diff_1[i,:] = temp1[np.triu_indices(temp1.shape[0],1)]
            temp2 = calc_diff(bin_dic_2[key], bin_dic_2[key], "cos")
            within_diff_2[i,:] = temp2[np.triu_indices(temp2.shape[0],1)]


        #plot distributions for each bin

        # row for plot
        c_r = -1
        c_p = 0
        fig, ax = plt.subplots(6, 3)
        for i, (w_d_1, w_d_2, c_d) in enumerate(zip(within_diff_1, within_diff_2, cross_diff)):
            w_d = np.hstack((w_d_1, w_d_2))
            if not np.mod(i, 3):
                c_r += 1
                c_p = 0
            ax[c_r,c_p].hist(w_d, label= "WITHIN", bins=40)
            ax[c_r,c_p].vlines(np.median(w_d),0,10, label="MEDIAN WITHIN",colors="blue")
            ax[c_r,c_p].hist(c_d,label="CROSS DIFF",bins=40)
            ax[c_r, c_p].vlines(np.median(c_d),0,10, label="MEDIAN CROSS",colors="red")
            ax[c_r,c_p].legend()

            c_p += 1
        plt.show()

        # for each spatial bin compare union of within_diff_1/within_diff_2 and cross_diff
        for i, (w_d_1, w_d_2, c_d) in enumerate(zip(within_diff_1, within_diff_2, cross_diff)):
            w_d = np.hstack((w_d_1, w_d_2))
            stats_array[i, 0], stats_array[i, 1] = stats.kruskal(w_d, c_d)


        # safe results
        self.cross_diff = cross_diff
        self.within_diff_1 = within_diff_1
        self.within_diff_2 = within_diff_2
        self.stats_array = stats_array

    def plot_remap_results(self):

        plot_remapping_summary(self.cross_diff, self.within_diff_1, self.within_diff_2, self.stats_array, self.param_dic)

    def combine_bin_dictionaries(self, dic_1, dic_2, comb_dic_name):
        # combines entries of two dictionaries
        if len(dic_1.keys()) != len(dic_2.keys()):
            print("Number of spatial bins in both dictionaries don't match")
            exit()

        # initialize dictionary --> each entry in dictionary is a spatial bin --> contains all population vectors
        # (from different trials) for this spatial interval
        combined_dic = {}

        for key in dic_1:
            combined_dic[key] = np.hstack((dic_1[key],dic_2[key]))

        # save first dictionary as pickle
        filename = "temp_data/quant_analysis/"+comb_dic_name
        outfile = open(filename, 'wb')
        pickle.dump(combined_dic, outfile)
        outfile.close()

    def leave_one_out(self, dic_1, dic_2):
        # calculate cross difference with all cells and drop cells that do not contribute to difference

        nr_cells = next(iter(dic_1.values())).shape[0]
        nr_bins = len(dic_1.keys())
        cell_to_p_value_contribution = np.zeros((nr_cells,nr_bins))
        cell_to_diff_contribution = np.zeros((nr_cells,nr_bins))

        # first cross diff using all cells
        self.cross_cos_diff(dic_1, dic_2)
        init_stats = self.stats_array
        init_cross = self.cross_diff

        # go through all cells
        for cell_ID in range(nr_cells):
            # make copies of both dictionaries
            dic_1_c = dic_1.copy()
            dic_2_c = dic_2.copy()

            for key in dic_1_c:
                # delete cell from copies of both dictionaries
                dic_1_c[key] = np.delete(dic_1_c[key],cell_ID, axis=0)
                dic_2_c[key] = np.delete(dic_2_c[key], cell_ID, axis=0)

            self.cross_cos_diff(dic_1_c, dic_2_c)
            cell_to_p_value_contribution[cell_ID,:] = init_stats[:,1]-self.stats_array[:,1]
            cell_to_diff_contribution[cell_ID,:] = np.average(init_cross, axis = 1) - np.average(self.cross_diff,axis=1)

        return cell_to_diff_contribution, cell_to_p_value_contribution

    def cell_rule_diff(self, dic_1, dic_2):
        # calculate change in average firing rate and standard error of the mean for each cell and bin between rules
        # using cohens d: (avg1-avg2)/pooled std

        nr_cells = next(iter(dic_1.values())).shape[0]
        nr_bins = len(dic_1.keys())
        cohens_d = np.zeros((nr_cells,nr_bins))

        # go through all spatial bins
        for i, key in enumerate(dic_1):
            cohens_d[:,i] = calc_cohens_d(dic_2[key],dic_1[key])
        return cohens_d

    def cell_avg_rate_map(self, dic_1, dic_2):
        # returns average rate map combining data from both dictionaries for each cell and spatial bin

        nr_cells = next(iter(dic_1.values())).shape[0]
        nr_bins = len(dic_1.keys())
        avg_rate_map = np.zeros((nr_cells, nr_bins))

        # go through all spatial bins
        for i, key in enumerate(dic_1):
            avg_rate_map[:, i] = np.average(np.hstack((dic_1[key],dic_2[key])),axis=1)
        return avg_rate_map


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

        # calculate within diff to plot
        med_within_diff = np.zeros(nr_intervals)

        # go through each bin
        for i, key in enumerate(spat_bin_dic_2):
            temp1 = calc_diff(spat_bin_dic_1[key], spat_bin_dic_1[key], "cos")
            within_diff_1 = temp1[np.triu_indices(temp1.shape[0], 1)]
            temp2 = calc_diff(spat_bin_dic_2[key], spat_bin_dic_2[key], "cos")
            within_diff_2 = temp2[np.triu_indices(temp2.shape[0], 1)]
            med_within_diff[i] = np.median(np.hstack((within_diff_1, within_diff_2)))


        # go through all trials after rule switch and compare each one with rule before rule switch
        # --> each entry in dictionary is a spatial bin --> contains all population vectors

        for trial_after_switch in range(nr_trials_after_switch):
            cross_diff = np.zeros((nr_intervals,nr_trials_before_switch))

            # go through all spatial bins and calculate difference --> rows: spatial bins, columns: cos distances
            for i, key in enumerate(spat_bin_dic_2):
                cross_diff[i, :] = calc_diff(spat_bin_dic_1[key], np.expand_dims(spat_bin_dic_2[key][:,trial_after_switch],1),"cos").flatten()

            plt.subplot(2,1,1)
            plt.plot(x_axis, np.median(cross_diff,1), color= col_map[trial_after_switch,:], marker= "o", label="TRIAL "+str(trial_after_switch))
            plt.grid()
            plt.xlim([min(x_axis), max(x_axis) + 20])
            plt.title("DIFFERENCE BETWEEN EACH TRIAL OF RULE 2 (_6) AND ALL TRIALS BEFORE RULE SWITCH (INCL. PREV. DAY)")
            plt.xlabel("MAZE POSITION")
            plt.ylabel("MEDIAN COS DIFFERENCE")

        plt.plot(x_axis, med_within_diff, color= "black", marker= "o", label="WITHIN")
        plt.legend()
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
            plt.plot(x_axis, np.median(cross_diff,1), color= col_map[i,:],marker= "o", label=str(spat_position[i])+" cm")
            plt.legend()
            plt.title("DIFFERENCE BETWEEN EACH TRIAL OF RULE 2 (_6) AND ALL TRIALS BEFORE RULE SWITCH (INCL. PREV. DAY)")
            plt.xlabel("TRIAL AFTER RULE SWITCH")
            plt.xlim([0, nr_trials_after_switch + 1])
            plt.ylabel("MEDIAN COS DIFFERENCE")
            plt.grid()

        plt.show()

    def characterize_cells(self, bin_dic_1, bin_dic_2):

        x_axis = np.arange(0,200,self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        cell_avg_rate_map = self.cell_avg_rate_map(bin_dic_1, bin_dic_2)
        cohens_d = self.cell_rule_diff(bin_dic_1, bin_dic_2)
        cell_to_diff_contribution, cell_to_p_value_contribution = self.leave_one_out(bin_dic_1, bin_dic_2)

        plot_cell_charact(cell_avg_rate_map, cohens_d, cell_to_diff_contribution, cell_to_p_value_contribution,x_axis)

        bin_dic_1_c = bin_dic_1.copy()
        bin_dic_2_c = bin_dic_2.copy()

        for key in bin_dic_1_c:
            # delete cell from copies of both dictionaries
            bin_dic_1_c[key] = np.delete(bin_dic_1_c[key], [46,69], axis=0)
            bin_dic_2_c[key] = np.delete(bin_dic_2_c[key], [46,69], axis=0)

        self.cross_cos_diff_spat_trials(bin_dic_1_c,bin_dic_2_c)

        # plt.hist(cell_to_diff_contribution, bins=20)
        # plt.show()
        #
        # # create new data set with only cells that decrease p-value if present
        #
        # dic_1_c = dic_1.copy()
        # dic_2_c = dic_2.copy()
        #
        # for key in dic_1_c:
        #     # delete cell from copies of both dictionaries
        #     dic_1_c[key] = dic_1_c[key][np.where(np.average(cell_to_diff_contribution, axis = 1) < 0),:]
        #     dic_2_c[key] = dic_2_c[key][np.where(np.average(cell_to_diff_contribution, axis = 1) < 0),:]
        #
        # self.cross_cos_diff(dic_1_c, dic_2_c)
        # self.plot_remap_results()

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

    def cross_cos_diff_spat_trials(self, spat_bin_dic_1, spat_bin_dic_2):

        if len(spat_bin_dic_1.keys()) != len(spat_bin_dic_2.keys()):
            print("Number of spatial bins in both dictionaries don't match")
            exit()

        nr_trials_rule_1 = next(iter(spat_bin_dic_1.values())).shape[1]
        nr_trials_rule_2 = next(iter(spat_bin_dic_2.values())).shape[1]
        nr_intervals = len(spat_bin_dic_1.keys())
        x_axis = np.arange(0,200,self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        col_map = cm.rainbow(np.linspace(0, 1, nr_trials_rule_2))

        # calculate within diff to plot
        med_within_diff = np.zeros(nr_intervals)

        # go through each bin
        for i, key in enumerate(spat_bin_dic_2):
            temp1 = calc_diff(spat_bin_dic_1[key], spat_bin_dic_1[key], "cos")
            within_diff_1 = temp1[np.triu_indices(temp1.shape[0], 1)]
            temp2 = calc_diff(spat_bin_dic_2[key], spat_bin_dic_2[key], "cos")
            within_diff_2 = temp2[np.triu_indices(temp2.shape[0], 1)]
            med_within_diff[i] = np.median(np.hstack((within_diff_1, within_diff_2)))

        # go through all trials for rule 1 and compare each trial with all trials for rule 2
        # --> each entry in dictionary is a spatial bin --> contains all population vectors

        for trial_rule_2 in range(nr_trials_rule_2):
            cross_diff = np.zeros((nr_intervals,nr_trials_rule_1))

            # go through all spatial bins and calculate difference --> rows: spatial bins, columns: cos distances
            for i, key in enumerate(spat_bin_dic_2):
                cross_diff[i, :] = calc_diff(spat_bin_dic_1[key], np.expand_dims(spat_bin_dic_2[key][:,trial_rule_2],1),"cos").flatten()

            plt.subplot(2,1,1)
            plt.plot(x_axis, np.median(cross_diff,1), color= col_map[trial_rule_2,:], marker= "o", label="TRIAL "+str(trial_rule_2))
            plt.grid()
            plt.xlim([min(x_axis), max(x_axis) + 20])
            plt.title("DIFFERENCE BETWEEN EACH TRIAL OF RULE 1 AFTER SLEEP AND ALL TRIALS OF RULE 1 BEFORE SLEEP")
            plt.xlabel("MAZE POSITION")
            plt.ylabel("MEDIAN COS DIFFERENCE")

        plt.plot(x_axis, med_within_diff, color= "black", marker= "o", label="WITHIN")
        plt.legend()

        # go through all spatial bins and see how the difference changes with trials after rule switch
        # --> each entry in dictionary is a spatial bin --> contains all population vectors

        col_map = cm.tab20(np.linspace(0, 1, len(x_axis)))
        spat_position = x_axis
        x_axis = np.arange(nr_trials_rule_2)

        for i, key in enumerate(spat_bin_dic_2):
            cross_diff = np.zeros((nr_trials_rule_2, nr_trials_rule_1))
            for trial_after_switch in range(nr_trials_rule_2):
                cross_diff[trial_after_switch, :] = calc_diff(spat_bin_dic_1[key], np.expand_dims(spat_bin_dic_2[key][:,trial_after_switch],1),"cos").flatten()

            plt.subplot(2, 1, 2)
            plt.plot(x_axis, np.median(cross_diff,1), color= col_map[i,:],marker= "o", label=str(spat_position[i])+" cm")
            plt.legend()
            plt.title("DIFFERENCE BETWEEN EACH TRIAL OF RULE 1 AFTER SLEEP AND ALL TRIALS OF RULE 1 BEFORE SLEEP")
            plt.xlabel("TRIAL ID RULE 2")
            plt.xlim([0, nr_trials_rule_2 + 1])
            plt.ylabel("MEDIAN COS DIFFERENCE")
            plt.grid()

        plt.show()