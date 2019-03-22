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
from mpl_toolkits.mplot3d import Axes3D



def plot_act_mat(act_mat,bin_interval):
# plot activation matrix (matrix of population vectors)
    plt.imshow(act_mat, vmin=0, vmax=act_mat.max(), cmap='jet', aspect='auto')
    plt.ylabel("CELL ID")
    plt.xlabel("TIME BINS / " + str(bin_interval) + " s")
    plt.title("CELL ACTIVATION / SPIKES PER TIME BIN")
    a = plt.colorbar()
    a.set_label("SPIKES")

def plot_2D_scatter(ax,mds,param_dic,data_sep = []):
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
            colors = cm.rainbow(np.linspace(0, 1, mds.shape[0] - 1))
            for i, c in zip(range(0, mds.shape[0] - 1), colors):
                ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], color=c)
        # plt.title(title)
        ax.scatter(mds[:, 0], mds[:, 1], color="grey")
        ax.scatter(mds[0, 0], mds[0, 1], color="black", marker="x", label="start", zorder=200)
        ax.scatter(mds[-1, 0], mds[-1, 1], color="black", label="end", zorder=200)

        # if axis limits are defined apply them
        if len(param_dic["axis_lim"]):
            axis_lim = param_dic["axis_lim"]
            ax.set_xlim(axis_lim[0], axis_lim[1])
            ax.set_ylim(axis_lim[2], axis_lim[3])

    ax.set_yticklabels([])
    ax.set_xticklabels([])


def plot_3D_scatter(ax,mds,param_dic,data_sep = []):
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
            colors = cm.rainbow(np.linspace(0, 1, mds.shape[0] - 1))
            for i, c in zip(range(0, mds.shape[0] - 1), colors):
                ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], mds[i:i + 2, 2], color=c)

        ax.scatter(mds[:, 0], mds[:, 1], mds[:, 2], color="grey")
        ax.scatter(mds[0, 0], mds[0, 1], mds[0, 2], color="black", marker="x", label="start", zorder=200)
        ax.scatter(mds[-1, 0], mds[-1, 1], mds[-1, 2], color="black", label="end", zorder=200)
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
