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
#       - BinDictionary:    class for creation & modification of binned dictionaries. One dictionary contains one entry
#                           per bin. One bin contains all population vectors as column vectors of different trials
#
#       - Analysis: class that analysis the data contained in binned dictionaries
#
#                   - cross_cos_diff: pooled within cos difference for both rules vs. across rule cos difference
#                     using all trials
#
#                   - cross_cos_diff_trials: cross difference between each trial of data set 2 and all trials of data
#                     set 2
#
#                   - characterize_cells: analysis of contribution of single cells to the difference between both data
#                     sets
#
#                   - remove_cells: cross_cos_diff and cross_cos_diff_trials with modified dictionaries leaving out
#                     defined cells
#
#
########################################################################################################################

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from comp_functions import get_activity_mat_spatial
from comp_functions import calc_diff
from comp_functions import calc_cohens_d
import matplotlib.cm as cm
from plotting_functions import plot_remapping_summary
from plotting_functions import plot_cell_charact
import matplotlib as mpl
import os
from statsmodels import robust

# set saving directory to current directory
mpl.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))

########################################################################################################################
#   BIN DICTIONARY CLASS
########################################################################################################################


class BinDictionary:
    """ Class for binned dictionaries --> bins can be spatial, temporal"""

    def __init__(self, param_dic):

        # binning method
        self.binning_method = param_dic["binning_method"]

        # vector with concatenated location data
        self.loc_vec = np.empty((0, 1))

        # parameter dictionary
        self.param_dic = param_dic

        # array with separator indices for different trials
        self.data_sep = []

        # set default saving directory
        self.saving_dir = param_dic["saving_dir_bin_dic"]

    def create_spatial_bin_dictionaries_transition(self, data_set, loc_set, new_rule_trial, dic_1_name, dic_2_name):
        # separates one data set into two binned dictionaries depending on the new rule trial
        # create "activity matrices" consisting of population vectors for each rule

        # how many cells are in the data set
        nr_cells = len(next(iter(data_set.values())))

        act_mat, _ = get_activity_mat_spatial(next(iter(data_set.values())),
                                              self.param_dic, next(iter(loc_set.values())))
        # how many intervals
        nr_intervals = act_mat.shape[1]
        act_mat = []

        # initialize dictionaries --> each entry in dictionary is a spatial bin --> contains all population vectors
        # (from different trials) for this spatial interval
        dic_spat_int_1 = {}
        dic_spat_int_2 = {}

        for spat_int in range(nr_intervals):
            dic_spat_int_1["INT"+str(spat_int)] = dic_spat_int_2["INT"+str(spat_int)] = \
                np.array([]).reshape(nr_cells,0)

        for trial, key_data_set in enumerate(data_set):
            act_mat, loc_vec_part = get_activity_mat_spatial(data_set[key_data_set], self.param_dic,
                                                             loc_set[key_data_set])

            # separate both rules
            if trial < (new_rule_trial-1):
                # rule 1
                # write elements in dictionary
                for int_count, key in enumerate(dic_spat_int_1):
                    dic_spat_int_1[key] = np.hstack((dic_spat_int_1[key], np.expand_dims(act_mat[:, int_count], 1)))
            else:
                # write elements in dictionary
                for int_count, key in enumerate(dic_spat_int_2):
                    dic_spat_int_2[key] = np.hstack((dic_spat_int_2[key], np.expand_dims(act_mat[:, int_count], 1)))

        # save first dictionary as pickle
        filename = self.saving_dir + "SWITCH_" + dic_1_name + "_spatial"
        outfile = open(filename, 'wb')
        pickle.dump(dic_spat_int_1, outfile)
        outfile.close()

        # save second dictionary as pickle
        filename = self.saving_dir + "SWITCH_" + dic_2_name + "_spatial"
        outfile = open(filename, 'wb')
        pickle.dump(dic_spat_int_2, outfile)
        outfile.close()

    def create_spatial_bin_dictionary(self, data_set, loc_set, dic_name):
        # create "activity matrices" consisting of population vectors for each rule

        # how many cells are in the data set
        nr_cells = len(next(iter(data_set.values())))

        act_mat, _ = get_activity_mat_spatial(next(iter(data_set.values())),
                                              self.param_dic, next(iter(loc_set.values())))
        # how many intervals
        nr_intervals = act_mat.shape[1]
        act_mat = []

        # initialize dictionary --> each entry in dictionary is a spatial bin --> contains all population vectors
        # (from different trials) for this spatial interval
        dic_spat_int = {}

        for spat_int in range(nr_intervals):
            dic_spat_int["INT" + str(spat_int)] = np.array([]).reshape(nr_cells, 0)

        # go through all trials
        for trial, key in enumerate(data_set):
            act_mat, loc_vec_part = get_activity_mat_spatial(data_set[key], self.param_dic,
                                                             loc_set[key])
            # write elements in dictionary
            for int_count, key in enumerate(dic_spat_int):
                dic_spat_int[key] = np.hstack((dic_spat_int[key], np.expand_dims(act_mat[:, int_count], 1)))

        # save first dictionary as pickle
        filename = self.saving_dir + dic_name + "_spatial"
        outfile = open(filename, 'wb')
        pickle.dump(dic_spat_int, outfile)
        outfile.close()

    def combine_bin_dictionaries(self, dic_1_name, dic_2_name, comb_dic_name):
        # takes two dictionaries and combines them in one

        dic_1 = pickle.load(open(self.saving_dir+dic_1_name, "rb"))
        dic_2 = pickle.load(open(self.saving_dir+dic_2_name, "rb"))

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
        filename = self.saving_dir+comb_dic_name
        outfile = open(filename, 'wb')
        pickle.dump(combined_dic, outfile)
        outfile.close()


########################################################################################################################
#   ANALYSIS BASE CLASS
########################################################################################################################

class Analysis:
    """Base class for quantitative analysis """

    def __init__(self, dic_1_name, dic_2_name, param_dic):

        self.saving_dir = param_dic["saving_dir_bin_dic"]

        # bin dictionaries
        self.bin_dic_1 = pickle.load(open(self.saving_dir + dic_1_name, "rb"))
        self.bin_dic_2 = pickle.load(open(self.saving_dir + dic_2_name, "rb"))

        # binning method
        self.binning_method = param_dic["binning_method"]

        # saving plot?
        self.save_plot = param_dic["save_plot"]

        # name for saving plot
        self.plot_file_name = param_dic["plot_file_name"]

        # stats method for comparison
        self.stats_method = param_dic["stats_method"]

        # vector with concatenated location data
        self.loc_vec = np.empty((0, 1))

        # parameter dictionary
        self.param_dic = param_dic

        # array with separator indices for different trials
        self.data_sep = []

        # storing cross-diff
        self.cross_diff = []

        # storing within diff for dic 1
        self.within_diff_1 = []

        # storing within diff for dic 2

        self.within_diff_2 = []
        # results of test statistic
        self.stats_array = []

    def cross_cos_diff(self, plot_results=True,  bin_dic_1_ext=None, bin_dic_2_ext=None):
        # calculates the pair-wise cos difference within each set and across both sets

        # if no dictionaries are provided use dictionaries of this instance
        if not (bin_dic_1_ext and bin_dic_2_ext):
            bin_dic_1 = self.bin_dic_1
            bin_dic_2 = self.bin_dic_2
        else:
            bin_dic_1 = bin_dic_1_ext
            bin_dic_2 = bin_dic_2_ext

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

        # for each spatial bin compare union of within_diff_1/within_diff_2 and cross_diff
        for i, (w_d_1, w_d_2, c_d) in enumerate(zip(within_diff_1, within_diff_2, cross_diff)):
            w_d = np.hstack((w_d_1, w_d_2))

            if self.stats_method == "KW":
                stats_array[i, 0], stats_array[i, 1] = stats.kruskal(w_d, c_d)
            elif self.stats_method == "MWU":
                stats_array[i, 0], stats_array[i, 1] = stats.mannwhitneyu(w_d, c_d)

        if plot_results:
            plot_remapping_summary(cross_diff, within_diff_1, within_diff_2, stats_array, self.param_dic)

            # plot distributions for each bin
            # row for plot
            c_r = -1
            c_p = 0
            fig, ax = plt.subplots(6, 3)
            for i, (w_d_1, w_d_2, c_d) in enumerate(zip(within_diff_1, within_diff_2, cross_diff)):
                w_d = np.hstack((w_d_1, w_d_2))
                if not np.mod(i, 3):
                    c_r += 1
                    c_p = 0
                ax[c_r, c_p].hist(w_d, label="WITHIN", bins=40)
                ax[c_r, c_p].vlines(np.median(w_d), 0, 10, label="MEDIAN WITHIN", colors="blue")
                ax[c_r, c_p].hist(c_d, label="CROSS DIFF", bins=40)
                ax[c_r, c_p].vlines(np.median(c_d), 0, 10, label="MEDIAN CROSS", colors="red")
                ax[c_r, c_p].legend()
                ax[c_r, c_p].set_title("INT "+str(i))

                c_p += 1
            plt.show()

        # if no dictionaries are provided, dictionaries of this instance are used --> results can be saved to
        # class attributes
        if not (bin_dic_1_ext and bin_dic_2_ext):
            # safe results
            self.cross_diff = cross_diff
            self.within_diff_1 = within_diff_1
            self.within_diff_2 = within_diff_2
            self.stats_array = stats_array

        # if separate dictionaries are used --> returned values
        else:
            return cross_diff, within_diff_1, within_diff_2, stats_array

    def leave_one_out(self):
        # calculate cross difference with all cells and drop cells that do not contribute to difference

        nr_cells = next(iter(self.bin_dic_1.values())).shape[0]
        nr_bins = len(self.bin_dic_1.keys())
        cell_to_p_value_contribution = np.zeros((nr_cells, nr_bins))
        cell_to_diff_contribution = np.zeros((nr_cells, nr_bins))
        # relative contribution
        rel_cell_to_diff_contribution = np.zeros((nr_cells, nr_bins))

        # first cross diff using all cells
        self.cross_cos_diff(False)

        # go through all cells
        for cell_ID in range(nr_cells):
            # make copies of both dictionaries
            dic_1_c = self.bin_dic_1.copy()
            dic_2_c = self.bin_dic_2.copy()

            for key in dic_1_c:
                # delete cell from copies of both dictionaries
                dic_1_c[key] = np.delete(dic_1_c[key], cell_ID, axis=0)
                dic_2_c[key] = np.delete(dic_2_c[key], cell_ID, axis=0)

            # calculate cross diff for modified dic with deleted cell
            cross_diff_mod, _, _, stats_array_mod = self.cross_cos_diff(False, dic_1_c, dic_2_c)
            cell_to_p_value_contribution[cell_ID, :] = self.stats_array[:, 1] - stats_array_mod[:, 1]

            cell_to_diff_contribution[cell_ID, :] = np.median(self.cross_diff, axis=1)-np.median(cross_diff_mod, axis=1)

            rel_cell_to_diff_contribution[cell_ID, :] = np.median(cross_diff_mod, axis=1) / \
                                                        np.median(self.cross_diff, axis=1)

        return cell_to_diff_contribution, rel_cell_to_diff_contribution, cell_to_p_value_contribution

    def cell_rule_diff(self):
        # calculate change in average firing rate and standard error of the mean for each cell and bin between rules
        # using cohens d: (avg1-avg2)/pooled std

        nr_cells = next(iter(self.bin_dic_1.values())).shape[0]
        nr_bins = len(self.bin_dic_1.keys())
        cohens_d = np.zeros((nr_cells, nr_bins))

        # go through all spatial bins
        for i, key in enumerate(self.bin_dic_1):
            cohens_d[:,i] = calc_cohens_d(self.bin_dic_2[key], self.bin_dic_1[key])
        return cohens_d

    def cell_avg_rate_map(self):
        # returns average rate map combining data from both dictionaries for each cell and spatial bin

        nr_cells = next(iter(self.bin_dic_1.values())).shape[0]
        nr_bins = len(self.bin_dic_1.keys())
        avg_rate_map = np.zeros((nr_cells, nr_bins))

        # go through all spatial bins
        for i, key in enumerate(self.bin_dic_1):
            avg_rate_map[:, i] = np.average(np.hstack((self.bin_dic_1[key], self.bin_dic_2[key])), axis=1)
        return avg_rate_map

    def cross_cos_diff_spat_trials(self, spat_bin_dic_1=None, spat_bin_dic_2=None):
        # calculates within vs. across using all trials from both dictionaries

        # if no dictionaries are provided use dictionaries of instance
        if not (spat_bin_dic_1 and spat_bin_dic_2):
            spat_bin_dic_1 = self.bin_dic_1
            spat_bin_dic_2 = self.bin_dic_2

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
            plt.title("DIFFERENCE BETWEEN EACH TRIAL OF DATA SET 1 AND ALL TRIALS OF DATA SET 2")
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
            plt.title("DIFFERENCE BETWEEN EACH TRIAL OF DATA SET 1 AND ALL TRIALS OF DATA SET 2")
            plt.xlabel("TRIAL ID RULE 2")
            plt.xlim([0, nr_trials_rule_2 + 1])
            plt.ylabel("MEDIAN COS DIFFERENCE")
            plt.grid()

        plt.show()

    def characterize_cells(self):
        # performs different analysis steps to identify contribution/fingerprint of single cells

        x_axis = np.arange(0,200,self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        cell_avg_rate_map = self.cell_avg_rate_map()
        cohens_d = self.cell_rule_diff()
        _, rel_cell_to_diff_contribution, cell_to_p_value_contribution = self.leave_one_out()

        # invert relative contribution for log --> if cell increases diff --> positive value
        rel_cell_to_diff_contribution = 1/rel_cell_to_diff_contribution

        plot_cell_charact(cell_avg_rate_map, cohens_d, rel_cell_to_diff_contribution,
                          cell_to_p_value_contribution, x_axis)

    def remove_cells(self, cell_ID_list):
        # performs different analysis steps with a modified dictionary (with removed cells)
        bin_dic_1_c = self.bin_dic_1.copy()
        bin_dic_2_c = self.bin_dic_2.copy()

        for key in bin_dic_1_c:
            # delete cell from copies of both dictionaries
            bin_dic_1_c[key] = np.delete(bin_dic_1_c[key], cell_ID_list, axis=0)
            bin_dic_2_c[key] = np.delete(bin_dic_2_c[key], cell_ID_list, axis=0)

        self.cross_cos_diff_spat_trials(bin_dic_1_c, bin_dic_2_c)
        self.cross_cos_diff(True, bin_dic_1_c, bin_dic_2_c)

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

    def cell_contribution(self):
        # check how many cells contribute how much to the difference between two conditions (e.g. RULES)
        x_axis = np.arange(0, 200, self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        cell_to_diff_contribution, rel_cell_to_diff_contribution, _ = self.leave_one_out()

        # are interested in contributions that make difference larger --> 1 - rel.cell.contr.
        rel_cell_to_diff_contribution = 1-rel_cell_to_diff_contribution

        # cells to consider
        nr_cells = 30

        # check contribution of first n cells after sorting
        rel_contribution_array = np.full((nr_cells+1, rel_cell_to_diff_contribution.shape[1]), np.nan)
        contribution_array = np.full((nr_cells + 1, cell_to_diff_contribution.shape[1]), np.nan)

        # make first column all zero
        rel_contribution_array[0, :] = 0
        contribution_array[0, :] = 0

        # go through spatial bins
        for i, spat_bin in enumerate(rel_cell_to_diff_contribution.T):
            # select all cells that contribute to diff
            temp = spat_bin[spat_bin > 0]
            # sort cells with positive contribution according to magnitude of contribution
            temp = np.cumsum(np.flip(np.sort(temp), axis=0))
            # copy to contribution array
            rel_contribution_array[1:min(nr_cells, temp.shape[0])+1, i] = temp[:min(nr_cells, temp.shape[0])]

        fig, axes = plt.subplots(2, 2)

        col_map = cm.rainbow(np.linspace(0, 1, x_axis.shape[0]))

        ax1 = axes[0,0]

        for i, contribution in enumerate(rel_contribution_array.T):
            ax1.plot(np.arange(0, nr_cells+1), contribution, color=col_map[i, :], label=str(x_axis[i])+" cm",
                     marker="o")

        ax1.set_title("RELATIVE CELL CONTRIBUTION TO DIFFERENCE")
        ax1.set_ylabel("CUM. REL. CONTRIBUTION TO DIFFERENCE")
        ax1.set_xlabel("NR. CELLS")
        ax1.legend()

        # go through spatial bins
        for i, spat_bin in enumerate(cell_to_diff_contribution.T):
            # select all cells that contribute to diff
            temp = spat_bin[spat_bin > 0]
            # sort cells with positive contribution according to magnitude of contribution
            temp = np.cumsum(np.flip(np.sort(temp), axis=0))
            # copy to contribution array
            contribution_array[1:min(nr_cells, temp.shape[0])+1, i] = temp[:min(nr_cells, temp.shape[0])]

        col_map = cm.rainbow(np.linspace(0, 1, x_axis.shape[0]))

        ax2 = axes[0, 1]

        for i, contribution in enumerate(contribution_array.T):
            ax2.plot(np.arange(0, nr_cells+1), contribution, color=col_map[i, :], label=str(x_axis[i])+" cm",
                     marker="o")

        ax2.set_title("ABS. CELL CONTRIBUTION TO DIFFERENCE")
        ax2.set_ylabel("CUM. CONTRIBUTION TO DIFFERENCE")
        ax2.set_xlabel("NR. CELLS")
        ax2.legend()

        norm_contribution = np.zeros(contribution_array.shape[1])
        norm_rel_contribution = np.zeros(contribution_array.shape[1])

        # go through contribution vector and and
        for i, (contribution, rel_contribution) in enumerate(zip(contribution_array.T,rel_contribution_array.T)):
            # go trough cell contribution
            for cell_ID in range(1, contribution.shape[0]):
                if abs(rel_contribution[cell_ID] - rel_contribution[cell_ID -1]) > 0.005:
                    norm_contribution[i] = contribution[cell_ID]/cell_ID
                    norm_rel_contribution[i] = rel_contribution[cell_ID] / cell_ID
                else:
                    break

        # plot for each spatial bin: magnitude of difference & contribution normalized by nr. of cells
        width = 8

        ax3 = axes[1, 0]
        ax3.bar(x_axis, norm_rel_contribution, width,color="orange")
        ax3.set_title("REMAPPING CHARACTERISTICS (RELATIVE)")
        ax3.set_ylabel("REL. CONTRIBUTION / NR. CELLS")
        ax3.set_xlabel("MAZE POSITION")

        overal_diff = np.median(self.cross_diff, axis=1)
        mad = robust.mad(self.cross_diff, c = 1, axis=1)

        ax4 = axes[1, 1]
        ax4.bar(x_axis, overal_diff,width,yerr=mad, label="CROSS DIFF")
        ax4.bar(x_axis, norm_contribution, width, label="NORMALIZED CONTRIBUTION",color="orange")
        ax4.set_title("REMAPPING CHARACTERISTICS")
        ax4.set_ylabel("CROSS DIFF (AVG & MAD) \n ABS. CONTRIBUTION / NR. CELLS")
        ax4.set_xlabel("MAZE POSITION")
        ax4.legend()
        fig.suptitle("LEAVE ONE OUT ANALYSIS")

        plt.show()