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



def plot_act_mat(act_mat,bin_interval):
# plot activation matrix (matrix of population vectors)
    plt.imshow(act_mat, vmin=0, vmax=act_mat.max(), cmap='jet', aspect='auto')
    plt.ylabel("CELL ID")
    plt.xlabel("TIME BINS / " + str(bin_interval) + " s")
    plt.title("CELL ACTIVATION / SPIKES PER TIME BIN")
    a = plt.colorbar()
    a.set_label("SPIKES")

def plot_2D_scatter(ax,mds,param_dic,data_sep = [], loc_vec = []):
# generates 2D scatter plot with selected data --> for more than 1 data set, data_sep needs to be defined to separate
# the data sets
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
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], color=col_map[int(np.ceil(norm_loc_vec[i+1][0]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i+1][0]))*s_l)+" cm")

            else:
                colors = cm.cool(np.linspace(0, 1, mds.shape[0] - 1))
                for i, c in zip(range(0, mds.shape[0] - 1), colors):
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], color=c)
            # plt.title(title)
            ax.scatter(mds[:, 0], mds[:, 1], color="grey")
            ax.scatter(mds[0, 0], mds[0, 1], color="black", marker="x", label="start", zorder=200)
            ax.scatter(mds[-1, 0], mds[-1, 1], color="black", label="end", zorder=200)

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
                    ax.scatter(mds[i, 0], mds[i, 1], color=col_map[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i][0]))*s_l)+" cm")
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


def plot_3D_scatter(ax,mds,param_dic,data_sep = [], loc_vec = []):
# generates 3D scatter plot with selected data --> for more than 1 data set, data_sep needs to be defined to separate
# the data sets

    # for more than one data set
    if data_sep:
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

    # for one data set
    else:
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
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], mds[i:i + 2, 2],color=col_map[int(np.ceil(norm_loc_vec[i+1][0]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i+1][0]))*s_l)+" cm")
            else:
                colors = cm.cool(np.linspace(0, 1, mds.shape[0] - 1))
                for i, c in zip(range(0, mds.shape[0] - 1), colors):
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], mds[i:i + 2, 2], color=c)

            ax.scatter(mds[:, 0], mds[:, 1], mds[:, 2], color="grey")
            ax.scatter(mds[0, 0], mds[0, 1], mds[0, 2], color="black", marker="x", label="start", zorder=200)
            ax.scatter(mds[-1, 0], mds[-1, 1], mds[-1, 2], color="black", label="end", zorder=200)
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
                # check if trial or rule is meant to be colored
                if rule_sep:
                    ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], color=col_map[data_ID, :],
                            label=label_arr[data_ID])
                else:
                    ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], color=col_map[data_ID, :],
                            label="TRIAL " + str(data_ID))
            ax.scatter(data_subset[:, 0], data_subset[:, 1], color="grey")
            ax.scatter(data_subset[0, 0], data_subset[0, 1], color="black", marker="x", label="start", zorder=200)
            ax.scatter(data_subset[-1, 0], data_subset[-1, 1], color="black", label="end", zorder=200)

    # 3D plot

    if param_dic["dr_method_p2"] == 3:
        # create figure instance
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for data_ID in range(len(data_sep)-1):
            data_subset = data[int(data_sep[data_ID]):int(data_sep[data_ID + 1]), :]

            for i in range(0, data_subset.shape[0] - 1):
                if rule_sep:
                    ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], data_subset[i:i + 2, 2],
                            color=col_map[data_ID, :],
                            label=label_arr[data_ID])
                else:
                    ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], data_subset[i:i + 2, 2],
                            color=col_map[data_ID, :],
                            label="TRIAL " + str(data_ID))
            ax.scatter(data_subset[:, 0], data_subset[:, 1], data_subset[:, 2], color="grey")
            ax.scatter(data_subset[0, 0], data_subset[0, 1], data_subset[0, 2], color="black", marker="x", label="start",
                       zorder=200)
            ax.scatter(data_subset[-1, 0], data_subset[-1, 1], data_subset[-1, 2], color="black", label="end", zorder=200)
            # hide z label
            ax.set_zticklabels([])

    # hide labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    fig.suptitle(param_dic["dr_method"]+" : "+param_dic["dr_method_p1"], fontweight='bold')
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys())
    plt.show()
    # return figure for saving
    return fig
