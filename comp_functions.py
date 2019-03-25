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
#       - perform_PCA: performs and applies PCA
#       - popVecDiff: computes difference vectors between subsequent population vectors
#       - calcLocAndSpeed: calculates location and speed
#
########################################################################################################################

import numpy as np
import math
import scipy.stats as sp
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_similarity_score
from sklearn.manifold import MDS
from scipy.spatial import distance
from scipy import signal
from sklearn.decomposition import PCA

def get_activity_mat(firing_times,param_dic,location=[]):
# computes activity matrix: bin_interval in seconds --> sums up the activity within one time interval
# rows: cells
# columns: time bins
    bin_interval = param_dic["time_bin_size"]

    # find first and last firing for each trial
    first_firing = np.inf
    last_firing = 0

    for key,value in firing_times.items():
        if value:
            first_firing = int(np.amin([first_firing, np.amin(value)]))
            last_firing = int(np.amax([last_firing, np.amax(value)]))

    # duration of trial (one time bin: 0.05ms --> 20kHz)
    dur_trial = (last_firing-first_firing)* 0.05*1e-3
    nr_intervals = int(dur_trial/bin_interval)
    size_intervals = int((last_firing-first_firing)/nr_intervals)

    # matrix with population vectors
    act_mat = np.zeros([len(firing_times.keys()),nr_intervals])

    # go through all cells: cell_ID is not used --> only firing times
    for cell_iter, (cell_ID, cell) in enumerate(firing_times.items()):
        # go through all time intervals
        for i in range(nr_intervals):
            start_intv = first_firing+i*size_intervals
            end_intv = first_firing+(i+1)*size_intervals
            # write population vectors
            cell_spikes_intv = [x for x in cell if start_intv <= x < end_intv]
            act_mat[cell_iter, i] = len(cell_spikes_intv)

    # write locations & speed for each interval if location file is provided
    if len(location):
        # vector with locations
        loc_vec = np.zeros(nr_intervals)
        # vector with speeds
        speed_vec = np.zeros(nr_intervals)

        # calculate location and speed for each interval
        loc, speed = calc_loc_and_speed(location)
        for i in range(nr_intervals):
            start_intv = i * size_intervals
            end_intv = (i + 1) * size_intervals
            loc_vec[i] = np.mean(loc[start_intv:end_intv])
            speed_vec[i] = np.mean(speed[start_intv:end_intv])

    # filter time bins with low velocity (high synchrony events)
    #-------------------------------------------------------------------------------------------------------------------
    if param_dic["speed_filter"]:
        if not len(location):
            raise Exception("No location data for speed filtering provided")

        int_sel = np.full(nr_intervals, False)

        # check speed for each time bin (interval), if above threshold --> include time bin
        for i in range(nr_intervals):
            if speed_vec[i] > param_dic["speed_filter"]:
                int_sel[i] = True
        act_mat = act_mat[:,int_sel]
        loc_vec = loc_vec[int_sel]
        speed_vec = speed_vec[int_sel]

    # filter all zero population vectors
    #-------------------------------------------------------------------------------------------------------------------
    if param_dic["zero_filter"]:
        # set all to vectors to False
        int_sel = np.full(act_mat.shape[1], False)
        # go through all population vectors
        for i,pop_vec in enumerate(act_mat.T):
            if np.count_nonzero(pop_vec):
                int_sel[i] = True
        act_mat = act_mat[:, int_sel]
        loc_vec = loc_vec[int_sel]
        speed_vec = speed_vec[int_sel]


    return act_mat, loc_vec

def calc_pop_vector_entropy(act_mat):
# calculates shannon entropy for each population vector in act_mat
    pop_vec_entropy = np.zeros(act_mat.shape[1])
    # calculate entropy
    for i,pop_vec in enumerate(act_mat.T):
        # add small value because of log
        pop_vec_entropy[i] = sp.entropy(pop_vec+0.000001)
    return pop_vec_entropy

def multi_dim_scaling(act_mat,param_dic):
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

def perform_PCA(act_mat,param_dic):
    pca = PCA(n_components=param_dic["dr_method_p2"])
    pca_result = pca.fit_transform(act_mat.T)
    param_dic["dr_method_p1"] = str(pca.explained_variance_ratio_)

    return pca_result










def pop_vec_diff(data_set):
# computes difference vectors between subsequent population vectors and returns matrix of difference vectors
    # calculate transition vector between two subsequent population states --> rows: cells, col: time bins
    diffMat = np.zeros((data_set.shape[0],data_set.shape[1]-1))
    for i,pop_vec in enumerate(data_set.T[:-1,:]):
        diffMat.T[i,:] = data_set.T[i+1,:] - data_set.T[i,:]
    return diffMat


def calc_loc_and_speed(whl):
# computes speed from the whl and returns speed in cm/s
# need to smooth position data --> accuracy of measurement: about +-1cm --> error for speed: +-40m/s
# last element of velocity vector is zero --> velocity is calculated using 2 locations

    #savitzky golay
    w_l = 15 # window length
    p_o = 4 # order of polynomial
    whl = signal.savgol_filter(whl, 15, 5)

    # one time bin: whl is recorded at 20kHz/512
    t_b = 1/(20e3/512)

    # upsampling to synchronize with spike data
    location = np.zeros(whl.shape[0] * 512)
    for i, loc in enumerate(whl):
        location[512 * i:(i + 1) * 512] = 512 * [loc]

    # calculate speed: x1-x0/dt
    temp_speed = np.zeros(whl.shape[0]-1)
    for i in range(temp_speed.shape[0]):
        temp_speed[i] = (whl[i+1]-whl[i])/t_b

    # smoothen speed using savitzky golay
    temp_speed = signal.savgol_filter(temp_speed, 15, 5)

    # upsampling to synchronize with spike data
    speed = np.zeros(whl.shape[0]*512)
    for i,bin_speed in enumerate(temp_speed):
        speed[512*i:(i+1)*512] = 512*[bin_speed]

    # plotting
    # t = np.arange(speed.shape[0])
    #
    # plt.plot(temp_speed,label="speed / m/s")
    # plt.plot(t/512,location, label = "location / cm")
    # plt.plot([0,350],[5,5], label = "threshold: 5cm/s")
    # plt.xlabel("time bins / 25.6ms")
    # plt.scatter(t/512,speed,color="b")
    # plt.legend()
    # plt.show()

    return location, speed
