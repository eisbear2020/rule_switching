########################################################################################################################
#
#   Plotting functions
#
#   Description:
#
#       - functions that help plotting results from analysis
#
#   Author: Lars Bollmann
#
#   Created: 21/03/2019
#
#   Structure:
#
#       - plotActMat: plot activation matrix (matrix of population vectors)
#       - plot2DScatter: generates 2D scatter plots for one or multiple data sets
#       - plot3DScatter: generates 3D scatter plots for one or multiple data sets
#
#
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from scipy import stats
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib.colors import LogNorm
from statsmodels import robust
plt.style.use('dark_background')


def plot_act_mat(act_mat,bin_interval):
    # plot activation matrix (matrix of population vectors)
    plt.imshow(act_mat, vmin=0, vmax=act_mat.max(), cmap='jet', aspect='auto')
    plt.imshow(act_mat, interpolation='nearest', aspect='auto', cmap="jet",
               extent=[15, 185, act_mat.shape[0] - 0.5, 0.5])
    plt.ylabel("CELL ID")
    plt.xlabel("SPATIAL BINS / CM")
    plt.title("RATE MAP")
    a = plt.colorbar()
    a.set_label("FIRING RATE / Hz")


def plot_2D_scatter(ax,mds,param_dic,data_sep = None, loc_vec = []):
    # generates 2D scatter plot with selected data --> for more than 1 data set, data_sep needs to be defined to
    # separate the data sets
    # for more than one data set
    if data_sep:
        # if lines between points should be drawn
        if param_dic["lines"]:
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
        # color points instead of lines
        else:
            # check if positions are provided
            if len(loc_vec):
                # length of track
                s_l = param_dic["spat_seg_plotting"]
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map_blue = cm.Reds(np.linspace(0, 1, nr_seg + 5))
                col_map_red = cm.Blues(np.linspace(0, 1, nr_seg + 5))
                col_map_blue = col_map_blue[5:,:]
                col_map_red = col_map_red[5:, :]
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i,c in enumerate(mds):
                    if i <= data_sep:
                        ax.scatter(mds[i, 0], mds[i, 1],  color=col_map_blue[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                                   label=param_dic["data_descr"][0]+ str(int(np.ceil(norm_loc_vec[i][0]))*s_l)+" cm")
                    elif data_sep < i:
                        ax.scatter(mds[i, 0], mds[i, 1], color=col_map_red[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                                   label=param_dic["data_descr"][1] + str(
                                       int(np.ceil(norm_loc_vec[i + 1][0])) * s_l) + " cm")

            else:
                for i,c in enumerate(mds):
                    if i <= data_sep:
                        ax.scatter(mds[i, 0], mds[i, 1], color="lightblue",label=param_dic["data_descr"][0])
                    elif data_sep < i:
                        ax.scatter(mds[i, 0], mds[i, 1], color="lightcoral",label=param_dic["data_descr"][1])

    # for one data set
    else:
        # draw lines if option is set to True
        if param_dic["lines"]:
            # use locations for line coloring if location vector is provided
            if len(loc_vec):
                # length of track
                s_l = param_dic["spat_seg_plotting"]
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map = cm.rainbow(np.linspace(0, 1, nr_seg))
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i in range(0, mds.shape[0] - 1):
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], color=col_map[int(np.ceil(norm_loc_vec[i+1]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i+1]))*s_l)+" cm")

            else:
                colors = cm.cool(np.linspace(0, 1, mds.shape[0] - 1))
                for i, c in zip(range(0, mds.shape[0] - 1), colors):
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], color=c)
            # plt.title(title)
            ax.scatter(mds[:, 0], mds[:, 1], color="grey")
            ax.scatter(mds[0, 0], mds[0, 1], color="white", marker="x", label="start", zorder=200)
            ax.scatter(mds[-1, 0], mds[-1, 1], color="white", label="end", zorder=200)

        else:
            # use locations for point coloring if location vector is provided
            if len(loc_vec):
                # length of track
                s_l = param_dic["spat_seg_plotting"]
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map = cm.rainbow(np.linspace(0, 1, nr_seg))
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i in range(mds.shape[0]):
                    ax.scatter(mds[i, 0], mds[i, 1], color=col_map[int(np.ceil(norm_loc_vec[i]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i]))*s_l)+" cm")
            else:
                colors = cm.cool(np.linspace(0, 1, mds.shape[0] - 1))
                for i, c in zip(range(0, mds.shape[0] - 1), colors):
                    ax.scatter(mds[i, 0], mds[i, 1], color=c)

        # if axis limits are defined apply them
        if len(param_dic["axis_lim"]):
            axis_lim = param_dic["axis_lim"]
            ax.set_xlim(axis_lim[0], axis_lim[1])
            ax.set_ylim(axis_lim[2], axis_lim[3])

    ax.set_yticklabels([])
    ax.set_xticklabels([])

def plot_3D_scatter(ax, mds, param_dic, data_sep=None, loc_vec=[]):
    # generates 3D scatter plot with selected data --> for more than 1 data set, data_sep needs to be defined to
    # separate the data sets

    # for more than one data set
    if data_sep:
        data_sep = int(data_sep)

        # if lines between points should be drawn
        if param_dic["lines"]:
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

        # color points instead of lines
        else:
            # check if positions are provided
            if len(loc_vec):
                # length of track
                s_l = param_dic["spatial_bin_size"]
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map_blue = cm.Reds(np.linspace(0, 1, nr_seg + 2))
                col_map_red = cm.Blues(np.linspace(0, 1, nr_seg +2 ))
                col_map_blue = col_map_blue[2:,:]
                col_map_red = col_map_red[2:, :]
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i,c in enumerate(mds):
                    if i <= data_sep:
                        ax.scatter(mds[i, 0], mds[i, 1],mds[i,2],  color=col_map_blue[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                                   label=param_dic["data_descr"][0]+ str(int(np.ceil(norm_loc_vec[i][0]))*s_l)+" cm")
                    elif data_sep < i:
                        ax.scatter(mds[i, 0], mds[i, 1], mds[i,2], color=col_map_red[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                                   label=param_dic["data_descr"][1] + str(
                                       int(np.ceil(norm_loc_vec[i + 1][0])) * s_l) + " cm")

    # for one data set
    else:
        if param_dic["lines"]:
            # use locations for line coloring if location vector is provided
            if len(loc_vec):
                # length of track
                s_l = param_dic["spatial_bin_size"]
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map = cm.rainbow(np.linspace(0, 1, nr_seg))
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i in range(0, mds.shape[0] - 1):
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], mds[i:i + 2, 2],color=col_map[int(np.ceil(norm_loc_vec[i+1][0]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i+1][0]))*s_l)+" cm")
            else:
                colors = cm.cool(np.linspace(0, 1, mds.shape[0] - 1))
                for i, c in zip(range(0, mds.shape[0] - 1), colors):
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], mds[i:i + 2, 2], color=c)

            ax.scatter(mds[:, 0], mds[:, 1], mds[:, 2], color="white")
            # ax.scatter(mds[0, 0], mds[0, 1], mds[0, 2], color="white", marker="x", label="start", zorder=200)
            # ax.scatter(mds[-1, 0], mds[-1, 1], mds[-1, 2], color="white", label="end", zorder=200)
        else:
            # use locations for point coloring if location vector is provided
            if len(loc_vec):
                # length of track
                s_l = param_dic["spat_seg_plotting"]
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map = cm.rainbow(np.linspace(0, 1, nr_seg))
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i in range(mds.shape[0]):
                    ax.scatter(mds[i, 0], mds[i, 1], mds[i, 2], color=col_map[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i][0]))*s_l)+" cm")
            else:
                colors = cm.cool(np.linspace(0, 1, mds.shape[0] - 1))
                for i, c in zip(range(0, mds.shape[0] - 1), colors):
                    ax.scatter(mds[i, 0], mds[i, 1], mds[i, 2], color=c)

        # if axis limits are defined apply them
        if len(param_dic["axis_lim"]):
            axis_lim = param_dic["axis_lim"]
            ax.set_xlim(axis_lim[0], axis_lim[1])
            ax.set_ylim(axis_lim[2], axis_lim[3])
            ax.set_zlim(axis_lim[4], axis_lim[5])
    # hide labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    # set pane alpha value to zero --> transparent
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))


def plot_compare(data, param_dic, data_sep, rule_sep = []):
# plots data in one plot and colors rules differently if "rule_sep" is provided. Otherwise each trial is colored with a
# different color

    if rule_sep:
        # to color 2 different subsets of trials (e.g. for different rules): new_rule_trial --> first trial with new
        # rule

        # create rgba color map
        col_map = np.zeros((len(data_sep)+1,4))
        col_map[:rule_sep] = colors.to_rgba_array("r")
        col_map[rule_sep:] = colors.to_rgba_array("b")

        # create label array
        label_arr = np.zeros((len(data_sep)+1), dtype=object)
        label_arr[:rule_sep] = param_dic["data_descr"][0]
        label_arr[rule_sep:] = param_dic["data_descr"][1]

    else:
        col_map = cm.rainbow(np.linspace(0, 1, len(data_sep)))

    # 2D plot
    if param_dic["dr_method_p2"] == 2:
        fig, ax = plt.subplots()
        for data_ID in range(len(data_sep)-1):
            data_subset = data[int(data_sep[data_ID]):int(data_sep[data_ID + 1]), :]

            for i in range(0, data_subset.shape[0] - 1):
                # check if lines should be plotted
                if param_dic["lines"]:
                    # check if trial or rule is meant to be colored
                    # rule
                    if rule_sep:
                        # check if lines should be plotted
                        ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], color=col_map[data_ID, :],
                                label=label_arr[data_ID])
                    # trials
                    else:
                        ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], color=col_map[data_ID, :],
                                label="TRIAL " + str(data_ID))
                    ax.scatter(data_subset[:, 0], data_subset[:, 1], color="grey")
                    # ax.scatter(data_subset[0, 0], data_subset[0, 1], color="black", marker="x", label="start", zorder=200)
                    # ax.scatter(data_subset[-1, 0], data_subset[-1, 1], color="black", label="end", zorder=200)
                # plot without lines
                else:
                    # color rules
                    if rule_sep:
                        ax.scatter(data_subset[i, 0], data_subset[i, 1], color=col_map[data_ID, :],
                                label=label_arr[data_ID])
                    # color trial
                    else:
                        ax.scatter(data_subset[i, 0], data_subset[i, 1], color=col_map[data_ID, :],
                                   label="TRIAL " + str(data_ID))

    # 3D plot

    if param_dic["dr_method_p2"] == 3:
        # create figure instance
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for data_ID in range(len(data_sep)-1):
            data_subset = data[int(data_sep[data_ID]):int(data_sep[data_ID + 1]), :]

            for i in range(0, data_subset.shape[0] - 1):
                # check if lines should be plotted
                if param_dic["lines"]:
                    if rule_sep:
                        ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], data_subset[i:i + 2, 2],
                                color=col_map[data_ID, :],
                                label=label_arr[data_ID])
                    else:
                        ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], data_subset[i:i + 2, 2],
                                color=col_map[data_ID, :],
                                label="TRIAL " + str(data_ID))
                    ax.scatter(data_subset[:, 0], data_subset[:, 1], data_subset[:, 2], color="grey")
                    # ax.scatter(data_subset[0, 0], data_subset[0, 1], data_subset[0, 2], color="white", marker="x", label="start",
                    #            zorder=200)
                    # ax.scatter(data_subset[-1, 0], data_subset[-1, 1], data_subset[-1, 2], color="white", label="end", zorder=200)
                else:
                    if rule_sep:
                        ax.scatter(data_subset[i, 0], data_subset[i, 1], data_subset[i, 2],
                                color=col_map[data_ID, :],
                                label=label_arr[data_ID])
                    else:
                        ax.scatter(data_subset[i, 0], data_subset[i, 1], data_subset[i, 2],
                                color=col_map[data_ID, :],
                                label="TRIAL " + str(data_ID))
            # hide z label
            ax.set_zticklabels([])
            # set pane alpha value to zero --> transparent
            ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
            ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
            ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))

    # hide labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])


    # return figure for saving
    return fig


def plot_remapping_summary(cross_diff, within_diff_1, within_diff_2, stats_array, param_dic):
    x_axis = np.arange(0, 200, param_dic["spatial_bin_size"])
    x_axis = x_axis[param_dic["spat_bins_excluded"][0]:param_dic["spat_bins_excluded"][-1]]

    med_1 = np.median(within_diff_1, axis=1)
    med_2 = np.median(within_diff_2, axis=1)

    med = np.median(cross_diff, axis=1)
    err = robust.mad(cross_diff, c=1, axis=1)


    plt.subplot(2, 2, 1)
    plt.errorbar(x_axis, med, yerr=err, fmt="o")
    plt.grid()
    plt.xlabel("MAZE LOCATION / cm")
    plt.ylabel("MEDIAN DISTANCE (COS) - MED & MAD")
    plt.title("ABSOLUTE")

    # add significance marker
    for i, p_v in enumerate(stats_array[:, 1]):
        if p_v < param_dic["stats_alpha"]:
            plt.scatter(x_axis[i] + 2, med[i] + 0.02, marker="*", edgecolors="Red",
                        label=param_dic["stats_method"]+", "+str(param_dic["stats_alpha"]))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.subplot(2, 2, 2)
    plt.scatter(x_axis, med / med_1)
    plt.xlabel("MAZE LOCATION / cm")
    plt.ylabel("MEDIAN DISTANCE (COS)")
    plt.title("NORMALIZED BY DATA SET 1")
    plt.grid()

    # add significance marker
    for i, p_v in enumerate(stats_array[:, 1]):
        if p_v < param_dic["stats_alpha"]:
            plt.scatter(x_axis[i] + 2, med[i] / med_1[i] + 0.05, marker="*", edgecolors="Red",
                        label=param_dic["stats_method"]+", "+str(param_dic["stats_alpha"]))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.subplot(2, 2, 3)
    plt.scatter(x_axis, med / med_2)
    plt.xlabel("MAZE LOCATION / cm")
    plt.ylabel("MEDIAN DISTANCE (COS)")
    plt.title("NORMALIZED BY DATA SET 2")
    plt.grid()

    # add significance marker
    for i, p_v in enumerate(stats_array[:, 1]):
        if p_v < param_dic["stats_alpha"]:
            plt.scatter(x_axis[i] + 2, med[i] / med_2[i] + 0.05, marker="*", edgecolors="Red",
                        label=param_dic["stats_method"]+", "+str(param_dic["stats_alpha"]))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.subplot(2, 2, 4)
    plt.scatter(x_axis, stats_array[:, 1])
    plt.hlines(param_dic["stats_alpha"], min(x_axis), max(x_axis), colors="Red", label=str(param_dic["stats_alpha"]))
    plt.xlabel("MAZE LOCATION / cm")
    plt.ylabel("P-VALUE")
    plt.title(param_dic["stats_method"]+": WITHIN-RULE vs. ACROSS-RULES")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_cell_charact(cell_avg_rate_map, cohens_d, cell_to_diff_contribution, xlabel,
                      sort_cells):

    if sort_cells:
        # sort according to appearance of peak
        peak_array = np.zeros(cell_avg_rate_map.shape[0])
        # find peak in for every cell
        for i, cell in enumerate(cell_avg_rate_map):
            # if no activity
            if max(cell) == 0.0:
                peak_array[i] = -1
            else:
                peak_array[i] = np.argmax(cell)

        peak_array += 1
        peak_order = peak_array.argsort()
        cell_avg_rate_map = cell_avg_rate_map[np.flip(peak_order[::-1], axis=0), :]
        cohens_d = cohens_d[np.flip(peak_order[::-1], axis=0), :]
        cell_to_diff_contribution = cell_to_diff_contribution[np.flip(peak_order[::-1], axis=0), :]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.subplots_adjust(wspace=0.1)

    im1 = ax1.imshow(cell_avg_rate_map, interpolation='nearest', aspect='auto',cmap="jet",extent=[min(xlabel),max(xlabel),cell_avg_rate_map.shape[0]-0.5,0.5])
    ax1_divider = make_axes_locatable(ax1)
    # add an axes to the right of the main axes.
    cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
    cb1 = colorbar(im1, cax=cax1,orientation="horizontal")
    cax1.xaxis.set_ticks_position("top")
    #ax1.set_xlabel("LINEARIZED POSITION / cm")
    ax1.set_ylabel("CELLS SORTED")
    cax1.set_title("AVERAGE \n FIRING RATE")

    # hide y label
    ax2.set_yticklabels([])
    im2 = ax2.imshow(cohens_d, interpolation='nearest', aspect='auto',cmap="jet",extent=[min(xlabel),max(xlabel),cell_avg_rate_map.shape[0]-0.5,0.5])
    ax2_divider = make_axes_locatable(ax2)
    # add an axes above the main axes.
    cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
    cb2 = colorbar(im2, cax=cax2, orientation="horizontal")
    # change tick position to top. Tick position defaults to bottom and overlaps
    # the image.
    cax2.xaxis.set_ticks_position("top")
    ax2.set_xlabel("LINEARIZED POSITION / cm")
    cax2.set_title("EFFECT SIZE: \n RULE A vs. RULE B")

    # hide y label
    ax3.set_yticklabels([])
    im3 = ax3.imshow(np.log(cell_to_diff_contribution), interpolation='nearest',
                     aspect='auto',cmap="jet",extent=[min(xlabel),max(xlabel),cell_avg_rate_map.shape[0]-0.5,0.5])
    ax3_divider = make_axes_locatable(ax3)
    # add an axes above the main axes.
    cax3 = ax3_divider.append_axes("top", size="7%", pad="2%")
    cb3 = colorbar(im3, cax=cax3, orientation="horizontal")
    # change tick position to top. Tick position defaults to bottom and overlaps
    # the image.
    cax3.xaxis.set_ticks_position("top")
    #ax3.set_xlabel("LINEARIZED POSITION / cm")
    cax3.set_title("REL. CONTRIBUTION TO DIFF \n (RULE A vs. RULE B) / LOG")

    # # hide y label
    # ax4.set_yticklabels([])
    # im4 = ax4.imshow(cell_to_p_value_contribution, interpolation='nearest', aspect='auto',cmap="jet",extent=[min(xlabel),max(xlabel),cell_avg_rate_map.shape[0]-0.5,0.5])
    # ax4_divider = make_axes_locatable(ax4)
    # # add an axes above the main axes.
    # cax4 = ax4_divider.append_axes("top", size="7%", pad="2%")
    # cb4 = colorbar(im4, cax=cax4, orientation="horizontal")
    # # change tick position to top. Tick position defaults to bottom and overlaps
    # # the image.
    # cax4.xaxis.set_ticks_position("top")
    # ax4.set_xlabel("LINEARIZED POSITION / cm")
    # cax4.set_title("CHANGE OF P-VALUE KW(RULE A vs. RULE B)")

    plt.show()

def plot_transition_comparison(x_axis, dics, rel_dics, param_dic, stats_array, measure):

    fig, ax = plt.subplots(2,2)

    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 0]
    ax4 = ax[1, 1]

    for data_set_ID, dic in enumerate(dics):
        med = np.full(x_axis.shape[0], np.nan)
        all_values = np.empty((1, 0))
        for i, key in enumerate(dic):
            if dic[key].size == 0:
                continue
            med[i] = np.median(dic[key],axis=1)
            all_values = np.hstack((all_values, dic[key]))
            err = robust.mad(dic[key], c=1, axis=1)
            ax1.errorbar(x_axis[i]+data_set_ID*2, med[i], yerr=err,ecolor="gray")
        ax1.plot(x_axis+data_set_ID*2,med, marker="o", label=param_dic["data_descr"][data_set_ID])
        ax1.set_title(measure.upper() + " BETWEEN SUBS. POP. VECT.")
        ax1.set_ylabel(measure.upper() + " - MED & MAD")
        ax1.set_xlabel("MAZE POSITION / CM")
        ax1.legend()

        ax3.hist(all_values[~np.isnan(all_values)],bins=50, alpha=0.5, label=param_dic["data_descr"][data_set_ID])
        ax3.set_title("HIST OF " +measure.upper()+" BETWEEN SUBS. POP. VECT.")
        ax3.set_xlabel(measure.upper())
        ax3.set_ylabel("COUNTS")
        ax3.legend()

    # add significance marker
    for i, p_v in enumerate(stats_array):
        if p_v < param_dic["stats_alpha"]:
            ax1.scatter(x_axis[i] + 2, ax1.get_ylim()[1]-0.1*ax1.get_ylim()[1], marker="*", edgecolors="Red",
                        label=param_dic["stats_method"] + ", " + str(param_dic["stats_alpha"]))
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())


    for data_set_ID, rel_dic in enumerate(rel_dics):

        # calculate relative step length
        rel_med = np.full(x_axis.shape[0], np.nan)
        all_rel_values = np.empty((1, 0))
        for i, key in enumerate(rel_dic):
            if rel_dic[key].size == 0:
                continue
            rel_med[i] = np.median(rel_dic[key],axis=1)
            all_rel_values = np.hstack((all_rel_values, rel_dic[key]))
            err = robust.mad(rel_dic[key], c=1, axis=1)
            ax2.errorbar(x_axis[i]+data_set_ID*2, rel_med[i], yerr=err,ecolor="gray")
        ax2.plot(x_axis+data_set_ID*2, rel_med, marker="o", label=param_dic["data_descr"][data_set_ID])
        ax2.set_title("RELATIVE CHANGE OF "+measure.upper()+" BETWEEN SUBS. POP. VECT.")
        ax2.set_ylabel("RELATIVE CHANGE")
        ax2.set_xlabel("MAZE POSITION / CM")

        ax4.hist(all_rel_values[~np.isnan(all_rel_values) & ~np.isinf(all_rel_values)],
                 bins=50, alpha=0.5, label=param_dic["data_descr"][data_set_ID])
        ax4.set_title("HIST OF RELATIVE CHANGE OF "+measure.upper()+" BETWEEN SUBS. POP. VECT.")
        ax4.set_xlabel("RELATIVE CHANGE")
        ax4.set_ylabel("COUNTS")
        ax4.legend()

    plt.show()

def plot_operations_comparison(x_axis, operation_dics, nr_of_cells_arr, param_dic):

    fig, ax = plt.subplots(2, 2)
    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 0]

    for data_set_ID, (operation_dic, nr_of_cells) in enumerate(zip(operation_dics, nr_of_cells_arr)):
        med = np.full(x_axis.shape[0], np.nan)
        all_values = np.empty((1, 0))
        for i, key in enumerate(operation_dic):
            if operation_dic[key].size == 0:
                continue
            med[i] = np.median(operation_dic[key][0]/nr_of_cells*100)
            #all_values = np.hstack((all_values, operation_dic[key][0]))
            err = robust.mad(operation_dic[key][0]/nr_of_cells*100, c=1)
            ax1.errorbar(x_axis[i]+data_set_ID*2, med[i], yerr=err, ecolor="gray")
        ax1.plot(x_axis+data_set_ID*2, med, marker="o", label=param_dic["data_descr"][data_set_ID])
        ax1.set_title("SILENCED CELLS")
        ax1.set_ylabel("% OF CELLS")
        ax1.set_xlabel("MAZE POSITION / CM")
        ax1.set_ylim(0, 30)
        ax1.legend(loc=1)

        med = np.full(x_axis.shape[0], np.nan)
        all_values = np.empty((1, 0))
        for i, key in enumerate(operation_dic):
            if operation_dic[key].size == 0:
                continue
            med[i] = np.median(operation_dic[key][1]/nr_of_cells*100)
            #all_values = np.hstack((all_values, operation_dic[key][0]))
            err = robust.mad(operation_dic[key][1]/nr_of_cells*100, c=1)
            ax2.errorbar(x_axis[i]+data_set_ID*2, med[i], yerr=err, ecolor="gray")
        ax2.plot(x_axis+data_set_ID*2, med, marker="o", label=param_dic["data_descr"][data_set_ID])
        ax2.set_title("UNCHANGED CELLS")
        ax2.set_ylabel("% OF CELLS")
        ax2.set_xlabel("MAZE POSITION / CM")
        ax2.set_ylim(60, 95)
        ax2.legend(loc=1)

        med = np.full(x_axis.shape[0], np.nan)
        all_values = np.empty((1, 0))
        for i, key in enumerate(operation_dic):
            if operation_dic[key].size == 0:
                continue
            med[i] = np.median(operation_dic[key][2]/nr_of_cells*100)
            #all_values = np.hstack((all_values, operation_dic[key][0]))
            err = robust.mad(operation_dic[key][2]/nr_of_cells*100, c=1)
            ax3.errorbar(x_axis[i]+data_set_ID*2, med[i], yerr=err, ecolor="gray")
        ax3.plot(x_axis+data_set_ID*2, med, marker="o", label=param_dic["data_descr"][data_set_ID])
        ax3.set_title("ACTIVATED CELLS")
        ax3.set_ylabel("% OF CELLS")
        ax3.set_xlabel("MAZE POSITION / CM")
        ax3.set_ylim(0,30)
        ax3.legend(loc=1)

    plt.show()
