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
#       TODO: implement non spatial methods
#
########################################################################################################################

import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import stats
from scipy.spatial import distance
from comp_functions import get_activity_mat_spatial
from comp_functions import get_activity_mat_time
from comp_functions import pop_vec_diff
from comp_functions import calc_diff
from comp_functions import pop_vec_dist
from comp_functions import calc_cohens_d
from comp_functions import angle_between_col_vectors
import matplotlib.cm as cm
from plotting_functions import plot_remapping_summary
from plotting_functions import plot_cell_charact
from plotting_functions import plot_transition_comparison
from plotting_functions import plot_operations_comparison
import matplotlib as mpl
import os
from statsmodels import robust
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

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

        # file name for saving
        self.file_name = param_dic["file_name"]

    def check_and_create_dic(self):
        # if dictionary does not exist yet --> create a new one
        if not os.path.isfile(self.saving_dir + self.file_name):
            dic = {
                "spatial": {},
                "temporal": {}
            }

        # if dictionary exists --> return
        else:
            dic = pickle.load(open(self.saving_dir + self.file_name, "rb"))
        return dic

    def create_spatial_bin_dictionaries_transition(self, data_set, loc_set, new_rule_trial, dic_1_name, dic_2_name):
        # separates one data set into two binned dictionaries depending on the new rule trial
        # create "activity matrices" consisting of population vectors for each rule

        # check if dictionary exists already
        dic = self.check_and_create_dic()

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

        # save to dictionary
        dic["spatial"]["SWITCH_" + dic_1_name] = dic_spat_int_1
        dic["spatial"]["SWITCH_" + dic_2_name] = dic_spat_int_2

        # save first dictionary as pickle
        filename = self.saving_dir+self.file_name
        outfile = open(filename, 'wb')
        pickle.dump(dic, outfile)
        outfile.close()

    def create_spatial_bin_dictionary(self, data_set, loc_set, dic_name):
        # create "activity matrices" consisting of population vectors for each rule

        # check if dictionary exists already
        dic = self.check_and_create_dic()

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

        dic["spatial"][dic_name] = dic_spat_int

        # save first dictionary as pickle
        filename = self.saving_dir+self.file_name
        outfile = open(filename, 'wb')
        pickle.dump(dic, outfile)
        outfile.close()

    def combine_bin_dictionaries(self, dic_1_name, dic_2_name, comb_dic_name):
        # takes two dictionaries and combines them in one

        # check if dictionary exists already
        dic = self.check_and_create_dic()

        dic_1 = dic["spatial"][dic_1_name]
        dic_2 = dic["spatial"][dic_2_name]

        # combines entries of two dictionaries
        if len(dic_1.keys()) != len(dic_2.keys()):
            print("Number of spatial bins in both dictionaries don't match")
            exit()

        # initialize dictionary --> each entry in dictionary is a spatial bin --> contains all population vectors
        # (from different trials) for this spatial interval
        combined_dic = {}

        for key in dic_1:
            combined_dic[key] = np.hstack((dic_1[key],dic_2[key]))

        dic["spatial"][comb_dic_name] = combined_dic

        # save first dictionary as pickle
        filename = self.saving_dir+self.file_name
        outfile = open(filename, 'wb')
        pickle.dump(dic, outfile)
        outfile.close()

########################################################################################################################
#   ANALYSIS BASE CLASS
########################################################################################################################


class Analysis:
    """Base class for quantitative analysis """

    def __init__(self, dic_1_name, dic_2_name, param_dic):

        self.saving_dir = param_dic["saving_dir_bin_dic"]
        self.file_name = param_dic["file_name"]

        # bin dictionaries
        self.bin_dic_1 = pickle.load(open(self.saving_dir + self.file_name, "rb"))["spatial"][dic_1_name]
        self.bin_dic_2 = pickle.load(open(self.saving_dir + self.file_name, "rb"))["spatial"][dic_2_name]

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
                stats_array[i, 0], stats_array[i, 1] = stats.mannwhitneyu(w_d, c_d, alternative="less")

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

    def leave_one_out_average_over_trials(self, distance_measure):

        nr_cells = next(iter(self.bin_dic_1.values())).shape[0]
        nr_bins = len(self.bin_dic_1.keys())
        cell_to_diff_contribution = np.zeros((nr_cells, nr_bins))
        # relative contribution
        rel_cell_to_diff_contribution = np.zeros((nr_cells, nr_bins))
        init_cos_distance = np.zeros(nr_bins)

        # first cos distance using all cells

        # go through all intervals
        for i, key in enumerate(self.bin_dic_1):

            avg_1 = np.expand_dims(np.average(self.bin_dic_1[key], axis=1),axis=1)
            avg_2 = np.expand_dims(np.average(self.bin_dic_2[key], axis=1), axis = 1)

            init_cos_distance[i] = calc_diff(avg_1,avg_2,distance_measure)


        # go through all cells
        for cell_ID in range(nr_cells):
            # make copies of both dictionaries
            dic_1_c = self.bin_dic_1.copy()
            dic_2_c = self.bin_dic_2.copy()

            for key in dic_1_c:
                # delete cell from copies of both dictionaries
                dic_1_c[key] = np.delete(dic_1_c[key], cell_ID, axis=0)
                dic_2_c[key] = np.delete(dic_2_c[key], cell_ID, axis=0)

            cos_distance = np.zeros(nr_bins)

            # go through all intervals
            for i, key in enumerate(dic_1_c):
                avg_1 = np.expand_dims(np.average(dic_1_c[key], axis=1), axis=1)
                avg_2 = np.expand_dims(np.average(dic_2_c[key], axis=1), axis=1)

                cos_distance[i] = calc_diff(avg_1, avg_2, distance_measure)

            cell_to_diff_contribution[cell_ID, :] = init_cos_distance - cos_distance #init_cos_distance - cos_distance
            rel_cell_to_diff_contribution[cell_ID, :] = cos_distance / init_cos_distance

        # for i,bin in enumerate(cell_to_diff_contribution.T):
        #     print(np.sum(bin,axis=0),init_cos_distance[i])
            #print(abs(np.sum([x for x in bin if x < 0], axis=0)), init_cos_distance[i])


        return cell_to_diff_contribution, rel_cell_to_diff_contribution, init_cos_distance

    def leave_n_out_average_over_trials(self, distance_measure):

        nr_cells = next(iter(self.bin_dic_1.values())).shape[0]
        nr_bins = len(self.bin_dic_1.keys())
        cell_to_diff_contribution = np.zeros((nr_cells, nr_bins))
        # relative contribution
        rel_cell_to_diff_contribution = np.zeros((nr_cells, nr_bins))
        init_cos_distance = np.zeros(nr_bins)

        # first cos distance using all cells

        # go through all intervals
        for i, key in enumerate(self.bin_dic_1):

            avg_1 = np.expand_dims(np.average(self.bin_dic_1[key], axis=1),axis=1)
            avg_2 = np.expand_dims(np.average(self.bin_dic_2[key], axis=1), axis = 1)

            init_cos_distance[i] = calc_diff(avg_1,avg_2,distance_measure)


        # go through all cells
        for cell_ID in range(nr_cells):
            # make copies of both dictionaries
            dic_1_c = self.bin_dic_1.copy()
            dic_2_c = self.bin_dic_2.copy()

            for key in dic_1_c:
                # delete cell from copies of both dictionaries
                dic_1_c[key] = np.delete(dic_1_c[key], cell_ID, axis=0)
                dic_2_c[key] = np.delete(dic_2_c[key], cell_ID, axis=0)

            cos_distance = np.zeros(nr_bins)

            # go through all intervals
            for i, key in enumerate(dic_1_c):
                avg_1 = np.expand_dims(np.average(dic_1_c[key], axis=1), axis=1)
                avg_2 = np.expand_dims(np.average(dic_2_c[key], axis=1), axis=1)

                cos_distance[i] = calc_diff(avg_1, avg_2, distance_measure)

            cell_to_diff_contribution[cell_ID, :] = init_cos_distance - cos_distance
            rel_cell_to_diff_contribution[cell_ID, :] = cos_distance / init_cos_distance

        plt.plot(cell_to_diff_contribution[:,-4])
        plt.show()
        exit()

        # for i,bin in enumerate(cell_to_diff_contribution.T):
        #     print(np.sum(bin,axis=0),init_cos_distance[i])
            #print(abs(np.sum([x for x in bin if x < 0], axis=0)), init_cos_distance[i])


        return cell_to_diff_contribution, rel_cell_to_diff_contribution, init_cos_distance

    def leave_n_out_random_average_over_trials(self, distance_measure, nr_shuffles):

        nr_cells = next(iter(self.bin_dic_1.values())).shape[0]
        nr_bins = len(self.bin_dic_1.keys())
        cell_to_diff_contribution = np.zeros((nr_cells, nr_bins))
        # relative contribution
        rel_cell_to_diff_contribution = np.zeros((nr_cells, nr_bins))
        init_cos_distance = np.zeros(nr_bins)

        # first cos distance using all cells

        # go through all intervals
        for i, key in enumerate(self.bin_dic_1):

            avg_1 = np.expand_dims(np.average(self.bin_dic_1[key], axis=1),axis=1)
            avg_2 = np.expand_dims(np.average(self.bin_dic_2[key], axis=1), axis = 1)

            init_cos_distance[i] = calc_diff(avg_1,avg_2,distance_measure)

        # different subsets of cells
        for nr_cells_subset in range(nr_cells):

            # do random selection n times
            n_r_s = nr_shuffles

            cos_distance = np.zeros((n_r_s,nr_bins))

            for random_selection in range(n_r_s):

                # make copies of both dictionaries
                dic_1_c = self.bin_dic_1.copy()
                dic_2_c = self.bin_dic_2.copy()

                # permuted index
                per_ind = np.random.permutation(np.arange(nr_cells))
                subset_ind = per_ind[:nr_cells_subset+1]

                for key in dic_1_c:
                    # select random cells
                    dic_1_c[key] = dic_1_c[key][subset_ind,:]
                    dic_2_c[key] = dic_2_c[key][subset_ind,:]

                # go through all intervals
                for i, key in enumerate(dic_1_c):
                    avg_1 = np.expand_dims(np.average(dic_1_c[key], axis=1), axis=1)
                    avg_2 = np.expand_dims(np.average(dic_2_c[key], axis=1), axis=1)

                    cos_distance[random_selection, i] = calc_diff(avg_1, avg_2, distance_measure)

            cell_to_diff_contribution[nr_cells_subset, :] = np.nanmean(cos_distance,axis=0)  #init_cos_distance - cos_distance
            rel_cell_to_diff_contribution[nr_cells_subset, :] = np.nanmean(cos_distance,axis=0) / init_cos_distance

        # for i,bin in enumerate(cell_to_diff_contribution.T):
        #     print(np.sum(bin,axis=0),init_cos_distance[i])
            #print(abs(np.sum([x for x in bin if x < 0], axis=0)), init_cos_distance[i])

        return cell_to_diff_contribution, rel_cell_to_diff_contribution, init_cos_distance

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

        return cell_to_diff_contribution, rel_cell_to_diff_contribution

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

    def plot_spatial_information(self):
        # calculates angles between subsequent transitions

        bin_interval = self.param_dic["spatial_bin_size"]
        # length of linearized path: 200 cm
        nr_intervals = int(200 / bin_interval)

        x_axis = np.arange(0,200,self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        nr_cells = next(iter(self.bin_dic_1.values())).shape[0]
        nr_bins = len(self.bin_dic_1.keys())
        avg_rate_map = np.zeros((nr_cells, nr_bins))

        # go through all spatial bins
        for i, key in enumerate(self.bin_dic_1):
            avg_rate_map[:, i] = np.average(np.hstack((self.bin_dic_1[key], self.bin_dic_2[key])), axis=1)

        # sort according to appearance of peak
        peak_array = np.zeros(avg_rate_map.shape[0])
        # find peak in for every cell
        for i, cell in enumerate(avg_rate_map):
            # if no activity
            if max(cell) == 0.0:
                peak_array[i] = -1
            else:
                peak_array[i] = np.argmax(cell)

        peak_array += 1
        peak_order = peak_array.argsort()
        ordered_avg_rate_map = avg_rate_map[np.flip(peak_order[::-1], axis=0), :]

        fig, ax1 = plt.subplots()
        fig.subplots_adjust(wspace=0.1)

        im1 = ax1.imshow(ordered_avg_rate_map, interpolation='nearest', aspect='auto', cmap="jet",
                         extent=[min(x_axis), max(x_axis), ordered_avg_rate_map.shape[0] - 0.5, 0.5])
        ax1_divider = make_axes_locatable(ax1)
        # add an axes to the right of the main axes.
        cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
        cb1 = colorbar(im1, cax=cax1, orientation="horizontal")
        cax1.xaxis.set_ticks_position("top")
        ax1.set_xlabel("LINEARIZED POSITION / cm")
        ax1.set_ylabel("CELL ID")
        cax1.set_title("AVERAGE FIRING RATE")
        plt.show()

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
        own_dic = False

        # if no dictionaries are provided use dictionaries of instance
        if not (spat_bin_dic_1 and spat_bin_dic_2):
            spat_bin_dic_1 = self.bin_dic_1
            spat_bin_dic_2 = self.bin_dic_2
            own_dic = True

        if len(spat_bin_dic_1.keys()) != len(spat_bin_dic_2.keys()):
            print("Number of spatial bins in both dictionaries don't match")
            exit()

        nr_trials_rule_1 = next(iter(spat_bin_dic_1.values())).shape[1]
        nr_trials_rule_2 = next(iter(spat_bin_dic_2.values())).shape[1]
        nr_intervals = len(spat_bin_dic_1.keys())
        x_axis = np.arange(0,200,self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        nr_comparisons = next(iter(spat_bin_dic_1.values())).shape[1] * next(iter(spat_bin_dic_2.values())).shape[1]

        overal_cross_diff = np.zeros((nr_intervals, nr_comparisons))
        col_map = cm.rainbow(np.linspace(0, 1, nr_trials_rule_2))

        # calculate within diff to plot
        med_within_diff = np.zeros(nr_intervals)

        # go through each bin
        for i, key in enumerate(spat_bin_dic_2):
            overal_cross_diff[i, :] = calc_diff(spat_bin_dic_1[key], spat_bin_dic_2[key], "cos").flatten()
            temp1 = calc_diff(spat_bin_dic_1[key], spat_bin_dic_1[key], "cos")
            within_diff_1 = temp1[np.triu_indices(temp1.shape[0], 1)]
            temp2 = calc_diff(spat_bin_dic_2[key], spat_bin_dic_2[key], "cos")
            within_diff_2 = temp2[np.triu_indices(temp2.shape[0], 1)]
            med_within_diff[i] = np.median(np.hstack((within_diff_1, within_diff_2)))

        # if own dictionary of instance is used
        if own_dic:
            # get stats results
            self.cross_cos_diff(False)
            stats_result = self.stats_array
        else:
            _, _, _, stats_result = self.cross_cos_diff(False,spat_bin_dic_1,spat_bin_dic_2)

        # go through all trials for rule 1 and compare each trial with all trials for rule 2
        # --> each entry in dictionary is a spatial bin --> contains all population vectors

        for trial_rule_2 in range(nr_trials_rule_2):
            cross_diff = np.zeros((nr_intervals,nr_trials_rule_1))

            # go through all spatial bins and calculate difference --> rows: spatial bins, columns: cos distances
            for i, key in enumerate(spat_bin_dic_2):
                cross_diff[i, :] = calc_diff(spat_bin_dic_1[key], np.expand_dims(spat_bin_dic_2[key][:,trial_rule_2],1),"cos").flatten()

            #plt.subplot(2,1,1)
            plt.plot(x_axis, np.median(cross_diff,1), color= "grey", marker= "o")
            #plt.grid()
            plt.xlim([min(x_axis), max(x_axis) + 20])
            plt.title("DISTANCE BETWEEN EACH TRIAL OF RULE B AND ALL TRIALS OF RULE A")
            plt.xlabel("MAZE POSITION")
            plt.ylabel("MEDIAN COS DISTANCE")

        plt.ylim([0.05, 0.95])
        plt.plot(x_axis,np.median(overal_cross_diff,1), color= "red", marker= "o", label="ACROSS")
        plt.plot(x_axis, med_within_diff, color= "white", marker= "o", label="WITHIN")
        plt.legend(loc='upper left',ncol=1)

        # add significance marker
        for i, p_v in enumerate(stats_result[:, 1]):
            if p_v < self.param_dic["stats_alpha"]:
                plt.scatter(x_axis[i]+0.5, np.median(overal_cross_diff,1)[i]+0.05, marker="*", edgecolors="yellow",
                            label=self.param_dic["stats_method"] + ", " + str(self.param_dic["stats_alpha"]),zorder=10)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.show()
        # go through all spatial bins and see how the difference changes with trials after rule switch
        # --> each entry in dictionary is a spatial bin --> contains all population vectors

        col_map = cm.tab20b(np.linspace(0, 1, len(x_axis)))
        spat_position = x_axis
        x_axis = np.arange(nr_trials_rule_2)

        for i, key in enumerate(spat_bin_dic_2):
            cross_diff = np.zeros((nr_trials_rule_2, nr_trials_rule_1))
            for trial_after_switch in range(nr_trials_rule_2):
                cross_diff[trial_after_switch, :] = calc_diff(spat_bin_dic_1[key], np.expand_dims(spat_bin_dic_2[key][:,trial_after_switch],1),"cos").flatten()

            # plt.subplot(2, 1, 2)
            plt.plot(x_axis, np.median(cross_diff,1), color= col_map[i,:],marker= "o", label=str(spat_position[i])+" cm")
            plt.legend()
            plt.title("DISTANCE BETWEEN EACH TRIAL OF RULE B AND ALL TRIALS OF RULE A")
            plt.xlabel("TRIAL ID RULE B")
            plt.xlim([0, nr_trials_rule_2 + 3])
            plt.ylabel("MEDIAN COS DISTANCE")
            #plt.grid()

        plt.show()

    def characterize_cells(self):
        # performs different analysis steps to identify contribution/fingerprint of single cells

        x_axis = np.arange(0,200,self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        cell_avg_rate_map = self.cell_avg_rate_map()
        cohens_d = self.cell_rule_diff()
        _, rel_cell_to_diff_contribution = self.leave_one_out()

        # invert relative contribution for log --> if cell increases diff --> positive value
        rel_cell_to_diff_contribution = 1/rel_cell_to_diff_contribution

        plot_cell_charact(cell_avg_rate_map, cohens_d, rel_cell_to_diff_contribution,
                          x_axis, self.param_dic["sort_cells"])

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

    def cell_contribution_average_over_trials(self, distance_measure):
        # check how many cells contribute how much to the difference between two conditions (e.g. RULES)
        x_axis = np.arange(0, 200, self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        cell_to_diff_contribution, rel_cell_to_diff_contribution, init_cos_dist = \
            self.leave_one_out_average_over_trials(distance_measure)

        # are interested in contributions that make difference larger --> 1 - rel.cell.contr.
        rel_cell_to_diff_contribution = 1-rel_cell_to_diff_contribution

        # cells to consider
        nr_cells = 30

        # cells for remapping characteristics
        n_first_cells = 1

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


        col_map = cm.rainbow(np.linspace(0, 1, x_axis.shape[0]))

        for i, contribution in enumerate(rel_contribution_array.T):
            plt.plot(np.arange(0, nr_cells+1), contribution, color=col_map[i, :], label=str(x_axis[i])+" cm",
                     marker="o")

        plt.title("RELATIVE CELL CONTRIBUTION TO DIFFERENCE")
        plt.ylabel("CUM. REL. CONTRIBUTION TO DIFFERENCE")
        plt.xlabel("NR. CELLS")
        plt.legend()
        plt.show()

        # go through spatial bins
        for i, spat_bin in enumerate(cell_to_diff_contribution.T):
            # select all cells that contribute to diff
            temp = spat_bin[spat_bin > 0]
            # sort cells with positive contribution according to magnitude of contribution
            temp = np.cumsum(np.flip(np.sort(temp), axis=0))
            # copy to contribution array
            contribution_array[1:min(nr_cells, temp.shape[0])+1, i] = temp[:min(nr_cells, temp.shape[0])]

        col_map = cm.rainbow(np.linspace(0, 1, x_axis.shape[0]))

        for i, contribution in enumerate(contribution_array.T):
            plt.plot(np.arange(0, nr_cells+1), contribution, color=col_map[i, :], label=str(x_axis[i])+" cm",
                     marker="o")

        plt.title("ABS. CELL CONTRIBUTION TO DIFFERENCE")
        plt.ylabel("CUM. CONTRIBUTION TO DIFFERENCE")
        plt.xlabel("NR. CELLS")
        plt.legend()
        plt.show()

        norm_contribution = np.zeros(contribution_array.shape[1])
        norm_rel_contribution = np.zeros(contribution_array.shape[1])
        first_n_cells_contribution = np.zeros(contribution_array.shape[1])

        # go through contribution vector and get relative contribution / nr. of cells
        for i, (contribution, rel_contribution) in enumerate(zip(contribution_array.T,rel_contribution_array.T)):
            # go trough cell contribution
            for cell_ID in range(1, contribution.shape[0]):
                if abs(rel_contribution[cell_ID] - rel_contribution[cell_ID -1]) > 0.005:
                    norm_contribution[i] = contribution[cell_ID]/cell_ID
                    norm_rel_contribution[i] = rel_contribution[cell_ID] / cell_ID
                else:
                    break

        # check if first n cells exist for all spatial positions
        n_cells_exist = True

        while n_cells_exist:
            n_cells_exist = False
            # go through contribution vector and get contribution of first n cells
            for i, contribution in enumerate(contribution_array.T):
                if np.isnan(contribution[n_first_cells]):
                    first_n_cells_contribution = np.zeros(contribution_array.shape[1])
                    n_first_cells -= 1
                    n_cells_exist = True
                    break
                else:
                    # go trough cell contribution
                    first_n_cells_contribution[i] = contribution[n_first_cells]

        # plot for each spatial bin: magnitude of difference & contribution normalized by nr. of cells
        width = 8

        plt.bar(x_axis, norm_rel_contribution, width,color="orange")
        plt.title("REMAPPING CHARACTERISTICS (RELATIVE)")
        plt.ylabel("REL. CONTRIBUTION / NR. CELLS")
        plt.xlabel("MAZE POSITION")
        plt.show()

        plt.bar(x_axis, init_cos_dist,width, label="DIFF")
        plt.bar(x_axis, first_n_cells_contribution, width, label="CONTR. OF "+str(n_first_cells)+" CELLS", color="orange")
        plt.title("CONTRIBUTION TO DIFF BY "+str(n_first_cells)+" MOST INFLUENTIAL CELLS")
        plt.ylabel("DIFF (AVG & MAD) \n CUM CONTRIBUTION OF "+str(n_first_cells)+" CELLS")
        plt.xlabel("MAZE POSITION")
        plt.legend()
        plt.show()

    def fit_remapped_cell_number(self, distance_measure, nr_shuffles):
        # check how many cells contribute how much to the difference between two conditions (e.g. RULES)
        spat_pos = np.arange(0, 200, self.param_dic["spatial_bin_size"])
        spat_pos = spat_pos[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        cohens_d = self.cell_rule_diff()

        result_list = []

        for i, key in enumerate(self.bin_dic_1):

            avg_1 = np.expand_dims(np.average(self.bin_dic_1[key], axis=1),axis=1)
            avg_2 = np.expand_dims(np.average(self.bin_dic_2[key], axis=1), axis = 1)

            # sort according to cohens d in decreasing order
            #ind = np.argsort(abs(cohens_d[:, i]))[::-1]

            # shuffle n times
            #for ind in itertools.permutations(range(avg_1.shape[0])):

            count_res = np.zeros(nr_shuffles)

            for i in range(nr_shuffles):
                results = np.zeros(avg_1.shape[0]+1)
                results_shifted = np.zeros(avg_1.shape[0] + 1)
                ind = np.random.permutation(range(avg_1.shape[0]))

                avg_1_n = avg_1[ind]
                avg_2_n = avg_2[ind]
                avg_1_mod = avg_2_n.copy()

                for e in range(avg_1.shape[0]):
                    avg_1_mod[e] = avg_1_n[e]
                    results[e+1] = distance.cosine(avg_1_n,avg_1_mod)
                results[0] = distance.cosine(avg_1_n,avg_2_n)
                results_shifted[:-1] = results[1:]
                diff = results-results_shifted
                #ind = np.argsort(diff)[::-1]
                diff = np.sort(diff)[::-1]
                thres = 0.9 * results[0]
                sum = 0
                count = 0
                while sum < thres:
                    sum += diff[count]
                    count += 1

                count_res[i] = count

                #results = results[np.argsort(diff)]
                #plt.plot(results)
            #plt.show()
            result_list.append(count_res)
            #exit()
        #plt.show()


        for i,bin in enumerate(result_list):
            plt.errorbar(spat_pos[i],np.average(bin), yerr=np.std(bin), fmt='--o', ecolor="white")

        plt.show()

        exit()

        print((avg_1-avg_2))
        exit()
        cell_to_diff_contribution, rel_cell_to_diff_contribution, init_cos_dist = \
            self.leave_n_out_random_average_over_trials(distance_measure,nr_shuffles)

        x_axis = np.arange(1,cell_to_diff_contribution.shape[0]+1)

        exit()
        # go through all intervals
        for i, key in enumerate(self.bin_dic_1):

            avg_1 = np.expand_dims(np.average(self.bin_dic_1[key], axis=1),axis=1)
            avg_2 = np.expand_dims(np.average(self.bin_dic_2[key], axis=1), axis = 1)

        exit()

        plt.plot(x_axis,cell_to_diff_contribution.T[0], color="white", label="data curve",
                     marker="o")
        plt.legend()
        plt.title("REMAPPING CHARACTERISTICS")
        plt.ylabel("COS DIFFERENCE")
        plt.xlabel("NR. CELLS IN SUBSET")
        plt.show()

        exit()

        col_map = cm.rainbow(np.linspace(0, 1, cell_to_diff_contribution.shape[1]))
        # go through all spatial bins
        for i, cont in enumerate(cell_to_diff_contribution.T):
            plt.plot(x_axis,cont, color=col_map[i, :], label=str(spat_pos[i])+" cm",
                     marker="o")
        plt.legend()
        plt.title("REMAPPING CHARACTERISTICS")
        plt.ylabel("COS DIFFERENCE")
        plt.xlabel("NR. CELLS IN SUBSET")
        plt.show()

    def cell_contribution_average_over_trials_random(self, distance_measure, nr_shuffles):
        # check how many cells contribute how much to the difference between two conditions (e.g. RULES)
        spat_pos = np.arange(0, 200, self.param_dic["spatial_bin_size"])
        spat_pos = spat_pos[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        cell_to_diff_contribution, rel_cell_to_diff_contribution, init_cos_dist = \
            self.leave_n_out_random_average_over_trials(distance_measure,nr_shuffles)

        x_axis = np.arange(1,cell_to_diff_contribution.shape[0]+1)

        col_map = cm.rainbow(np.linspace(0, 1, cell_to_diff_contribution.shape[1]))
        # go through all spatial bins
        for i, cont in enumerate(cell_to_diff_contribution.T):
            plt.plot(x_axis,cont, color=col_map[i, :], label=str(spat_pos[i])+" cm",
                     marker="o")
        plt.legend()
        plt.title("REMAPPING CHARACTERISTICS")
        plt.ylabel("COS DIFFERENCE")
        plt.xlabel("NR. CELLS IN SUBSET")
        plt.show()

    def cell_contribution_cohen(self):
        diff = self.cell_rule_diff()
        plt.hist(np.abs(diff))
        plt.show()
        # significance according to cohen: effect size > 0.8
        remap_char = np.zeros(diff.shape[1])
        for i, spat_bin in enumerate(diff.T):
            remap_char[i] = len([x for x in spat_bin if abs(x) > 0.8])

        x_axis = np.arange(0, 200, self.param_dic["spatial_bin_size"])
        x_axis = x_axis[self.param_dic["spat_bins_excluded"][0]:self.param_dic["spat_bins_excluded"][-1]]

        plt.scatter(x_axis, remap_char)
        plt.title("REMAPPING CHARACTERISTICS USING EFFECT SIZE \n FOR AVERAGE FIRING RATE")
        plt.xlabel("MAZE POSITION")
        plt.ylabel("#CELLS WITH EFFECT SIZE > 0.8")
        plt.grid()
        plt.show()



########################################################################################################################
#   STATE TRANSITION ANALYSIS CLASS
########################################################################################################################

class StateTransitionAnalysis:
    """ Class state transition analysis"""

    def __init__(self, data_sets, loc_sets, param_dic):

        self.data_sets = data_sets
        self.loc_sets = loc_sets
        self.param_dic = param_dic
        self.stats_method = param_dic["stats_method"]

    def compare_distance(self, measure):
        # compares euclidean distance between subsequent population vectors of two data sets
        x_axis, dist_dic_1, rel_dist_dic_1 = self.distance(self.data_sets[0], self.loc_sets[0], measure)
        _, dist_dic_2, rel_dist_dic_2 = self.distance(self.data_sets[1], self.loc_sets[1], measure)

        stats_array = np.zeros(len(dist_dic_1.keys()))

        for i, key in enumerate(dist_dic_1):
            # check if difference is statistically significant

            set_1 = np.squeeze(dist_dic_1[key])
            set_2 = np.squeeze(dist_dic_2[key])

            if len(set_1) == 0 or len(set_2) == 0:
                stats_array[i] = np.nan
            else:
                if self.stats_method == "KW":
                    _, stats_array[i] = stats.kruskal(set_1, set_2)
                elif self.stats_method == "MWU":
                    _, stats_array[i] = stats.mannwhitneyu(set_1, set_2)

        plot_transition_comparison(x_axis, [dist_dic_1,dist_dic_2], [rel_dist_dic_1,rel_dist_dic_2],
                                   self.param_dic, stats_array,measure)

    def compare_operations(self):
        # compares operations (silencing, activation, unchanged) between states for two data sets
        x_axis, op_dic_1, nr_of_cells_1  = self.operations(self.data_sets[0], self.loc_sets[0])
        _, op_dic_2, nr_of_cells_2  = self.operations(self.data_sets[1], self.loc_sets[1])

        plot_operations_comparison(x_axis,[op_dic_1,op_dic_2],[nr_of_cells_1,nr_of_cells_2], self.param_dic)

    def compare_angle(self):
        # compares angles between subsequent population vectors for two data sets
        x_axis, angle_dic_1, rel_angle_dic_1 = self.angle(self.data_sets[0], self.loc_sets[0])
        _, angle_dic_2, rel_angle_dic_2 = self.angle(self.data_sets[1], self.loc_sets[1])

        stats_array = np.zeros(len(angle_dic_1.keys()))

        for i, key in enumerate(angle_dic_1):
            # check if difference is statistically significant

            set_1 = np.squeeze(angle_dic_1[key])
            set_2 = np.squeeze(angle_dic_2[key])

            if len(set_1) == 0 or len(set_2) == 0:
                stats_array[i] = np.nan
            else:
                if self.stats_method == "KW":
                    _, stats_array[i] = stats.kruskal(set_1, set_2)
                elif self.stats_method == "MWU":
                    _, stats_array[i] = stats.mannwhitneyu(set_1, set_2)

        plot_transition_comparison(x_axis, [angle_dic_1,angle_dic_2], [rel_angle_dic_1,rel_angle_dic_2],
                                   self.param_dic, stats_array,"ANGLE")

    def filter_cells(self, data_set, loc_set):
        # filter cells that do not show any activity at all

        # how many cells are in the data set
        nr_cells = len(next(iter(data_set.values())))

        dat_mat = np.array([]).reshape(nr_cells,0)

        # go through all trials
        for i, key in enumerate(data_set):
            # get binned activity matrix
            act_mat, loc_mat = get_activity_mat_time(data_set[key], self.param_dic,loc_set[key])

            # concatenate spike matrices
            dat_mat = np.hstack((dat_mat, act_mat))

        non_zero_indices = []

        # go through all cells
        for cell_ID, cell in enumerate(dat_mat):
            if np.count_nonzero(cell):
                non_zero_indices.append(cell_ID)

        return non_zero_indices

    def distance(self, data_set, loc_set, measure):
        # calculates euclidean distance between subsequent population vectors

        bin_interval = self.param_dic["spatial_bin_size"]
        # length of linearized path: 200 cm
        nr_intervals = int(200 / bin_interval)

        # initialize dictionary with positions
        distance_dic = {}
        rel_distance_dic = {}
        for i in range(nr_intervals):
            distance_dic["INT"+str(i)] = np.empty((1, 0))
            rel_distance_dic["INT" + str(i)] = np.empty((1, 0))

        non_zero_indices = self.filter_cells(data_set, loc_set)

        # go through all trials
        for i, key in enumerate(data_set):
            # get binned activity matrix
            act_mat, loc_mat = get_activity_mat_time(data_set[key], self.param_dic,loc_set[key])
            # filter non active cells
            act_mat = act_mat[non_zero_indices,:]
            # get distances
            dist_array, rel_dist_array = pop_vec_dist(act_mat, measure)
            # drop last position --> distance returns n-1 long row vector
            loc_mat = loc_mat[:-1]
            # drop last position --> relative distance has length n-2
            rel_loc_mat = loc_mat[:-1]
            # assign values to dictionary according to spatial position
            for int_counter in range(nr_intervals):
                # define spatial interval
                start_interval = int_counter * bin_interval
                end_interval = (int_counter + 1) * bin_interval
                # find entries that are within interval
                distance_dic["INT"+str(int_counter)] = np.hstack((distance_dic["INT"+str(int_counter)],
                    np.expand_dims(dist_array[(start_interval <= loc_mat) & (loc_mat < end_interval)],axis=0)))

                # find entries that are within interval
                rel_distance_dic["INT"+str(int_counter)] = np.hstack((rel_distance_dic["INT"+str(int_counter)],
                    np.expand_dims(rel_dist_array[(start_interval <= rel_loc_mat) & (rel_loc_mat < end_interval)],axis=0)))

            x_axis = np.linspace(bin_interval, bin_interval * nr_intervals, nr_intervals)

        return x_axis, distance_dic, rel_distance_dic

    def angle(self, data_set, loc_set):
        # calculates angles between subsequent transitions

        bin_interval = self.param_dic["spatial_bin_size"]
        # length of linearized path: 200 cm
        nr_intervals = int(200 / bin_interval)

        # initialize dictionary with positions
        angle_dic = {}
        rel_angle_dic = {}
        for i in range(nr_intervals):
            angle_dic["INT"+str(i)] = np.empty((1, 0))
            rel_angle_dic["INT" + str(i)] = np.empty((1, 0))

        non_zero_indices = self.filter_cells(data_set, loc_set)

        # go through all trials
        for i, key in enumerate(data_set):
            # get binned activity matrix
            act_mat, loc_mat = get_activity_mat_time(data_set[key], self.param_dic,loc_set[key])
            # filter non active cells
            act_mat = act_mat[non_zero_indices,:]
            # get difference matrix: transitions between pop-vectors
            diff_matrix = pop_vec_diff(act_mat)
            # calculate angles
            angle_array, rel_angle_array = angle_between_col_vectors(diff_matrix)
            # drop last two position --> angle returns n-2 long row vector
            loc_mat = loc_mat[:-2]
            rel_loc_mat = loc_mat[:-1]
            # assign values to dictionary accccording to spatial position
            for int_counter in range(nr_intervals):
                # define spatial interval
                start_interval = int_counter * bin_interval
                end_interval = (int_counter + 1) * bin_interval
                # find entries that are within interval
                angle_dic["INT"+str(int_counter)] = np.hstack((angle_dic["INT"+str(int_counter)],
                    np.expand_dims(angle_array[(start_interval <= loc_mat) & (loc_mat < end_interval)],axis=0)))

                # find entries that are within interval
                rel_angle_dic["INT"+str(int_counter)] = np.hstack((rel_angle_dic["INT"+str(int_counter)],
                    np.expand_dims(rel_angle_array[(start_interval <= rel_loc_mat) & (rel_loc_mat < end_interval)],axis=0)))

            x_axis = np.linspace(bin_interval, bin_interval * nr_intervals, nr_intervals)

        return x_axis, angle_dic, rel_angle_dic

    def operations(self, data_set, loc_set):
        # calculates number of zeros (no change), +1 (activation) and -1 (inhibition) in population difference
        # vectors

        bin_interval = self.param_dic["spatial_bin_size"]
        # length of linearized path: 200 cm
        nr_intervals = int(200 / bin_interval)

        # initialize dictionary with positions
        operation_dic = {}
        for i in range(nr_intervals):
            operation_dic["INT"+str(i)] = np.empty((3, 0))

        non_zero_indices = self.filter_cells(data_set, loc_set)
        nr_of_cells = len(non_zero_indices)

        # go through all trials
        for i, key in enumerate(data_set):

            # get binned activity matrix
            act_mat, loc_mat = get_activity_mat_time(data_set[key], self.param_dic,loc_set[key])
            # filter non active cells
            act_mat = act_mat[non_zero_indices,:]
            # get difference matrix: transitions between pop-vectors and make them signed binary
            diff_matrix = np.sign(pop_vec_diff(act_mat))
            count_operations = np.zeros((3,diff_matrix.shape[1]))

            # go through all difference vectors and count
            for e, diff_vec in enumerate(diff_matrix.T):
                unique, counts = np.unique(diff_vec, return_counts=True)
                for f, operation in enumerate(unique):
                    count_operations[int(operation)+1,e] = counts[f]

            # drop last position --> difference vector is n-1 long
            loc_mat = np.expand_dims(loc_mat[:-1], axis = 0)

            # assign values to dictionary accccording to spatial position
            for int_counter in range(nr_intervals):
                # define spatial interval
                start_interval = int_counter * bin_interval
                end_interval = (int_counter + 1) * bin_interval
                # find entries that are within interval
                operation_dic["INT"+str(int_counter)] = np.hstack((operation_dic["INT"+str(int_counter)],
                    count_operations[:,np.squeeze((start_interval <= loc_mat) & (loc_mat < end_interval))]))

        x_axis = np.linspace(bin_interval, bin_interval * nr_intervals, nr_intervals)

        return x_axis, operation_dic, nr_of_cells



