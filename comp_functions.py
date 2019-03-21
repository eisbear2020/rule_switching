########################################################################################################################
#
#   Computation functions
#
#   Description:
#
#       - functions that are needed for neural data analysis
#
#   Author: Lars Bollmann
#
#   Created: 21/03/2019
#
#   Structure:
#
#       - getActivityMat: computes activity matrix (matrix with population vectors)
#       - calcPopVectorEntropy: calculates shannon entropy for each population vector in act_mat
#       - multiDimScaling: returns fitted multi scale model using defined difference measure
#       - popVecDiff: computes difference vectors between subsequent population vectors
#
########################################################################################################################

import numpy as np
import math
import scipy.stats as sp
from sklearn.metrics import jaccard_similarity_score
from sklearn.manifold import MDS
from scipy.spatial import distance

def getActivityMat(firing_times,param_dic):
# computes activity matrix: bin_interval in seconds --> sums up the activity within one time interval
# rows: cells
# columns: time bins
    bin_interval = param_dic["bin_interval"]

    # find first and last firing for each trial
    first_firing = np.inf
    last_firing = 0

    for key,value in firing_times.items():
        if value:
            first_firing = int(np.amin([first_firing, np.amin(value)]))
            last_firing = int(np.amax([last_firing, np.amax(value)]))

    # duration of trial (one time bin: 0.05ms)
    dur_trial = (last_firing-first_firing)* 0.05*1e-3
    nr_intervals = int(dur_trial/bin_interval)
    size_intervals = int((last_firing-first_firing)/nr_intervals)

    # binary matrix
    act_mat = np.zeros([len(firing_times.keys()),nr_intervals])

    # go through all cells
    for cell_ID, [key,cell] in enumerate(firing_times.items()):
        # go through all time intervals
        for i in range(nr_intervals):
            start_intv = first_firing+i*size_intervals
            end_intv = first_firing+(i+1)*size_intervals
            cell_spikes_intv = [x for x in cell if start_intv <= x < end_intv]
            act_mat[cell_ID,i] = len(cell_spikes_intv)

    return act_mat

def calcPopVectorEntropy(act_mat):
# calculates shannon entropy for each population vector in act_mat
    pop_vec_entropy = np.zeros(act_mat.shape[1])
    # calculate entropy
    for i,pop_vec in enumerate(act_mat.T):
        # add small value because of log
        pop_vec_entropy[i] = sp.entropy(pop_vec+0.000001)
    return pop_vec_entropy

def multiDimScaling(act_mat,param_dic):
# returns fitted multi scale model using defined difference measure

    if param_dic["dr_method_p1"] == "jaccard":
        # calculate difference matrix: Jaccard
        D = np.zeros([act_mat.shape[1],act_mat.shape[1]])

        # Jaccard similarity
        for i,pop_vec_ref in enumerate(act_mat.T):
            for j,pop_vec_comp in enumerate(act_mat.T):
                D[i,j] = jaccard_similarity_score(pop_vec_ref,pop_vec_comp)

        # want difference --> diff_jaccard = 1 - sim_jaccard
        D = 1 - D
        # plt.imshow(D)
        # plt.colorbar()
        # plt.show()
    elif param_dic["dr_method_p1"] == "cos":
        # calculate difference matrix: cosine
        D = np.zeros([act_mat.shape[1], act_mat.shape[1]])

        # cosine
        for i, pop_vec_ref in enumerate(act_mat.T):
            for j, pop_vec_comp in enumerate(act_mat.T):
                    D[i, j] = distance.cosine(pop_vec_ref, pop_vec_comp)
                    # if one of the vectors contains only zeros --> division by zero for cosine
                    if math.isnan(D[i,j]):
                        D[i, j] = 1

    model = MDS(n_components=param_dic["dr_method_p2"], dissimilarity='precomputed', random_state=1)
    return model.fit_transform(D)


def popVecDiff(data_set):
# computes difference vectors between subsequent population vectors and returns matrix of difference vectors
    # calculate transition vector between two subsequent population states --> rows: cells, col: time bins
    diffMat = np.zeros((data_set.shape[0],data_set.shape[1]-1))
    for i,pop_vec in enumerate(data_set.T[:-1,:]):
        diffMat.T[i,:] = data_set.T[i+1,:] - data_set.T[i,:]
    return diffMat

