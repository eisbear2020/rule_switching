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
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_similarity_score
from sklearn.manifold import MDS
from scipy.spatial import distance
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_activity_mat_spatial(firing_times,param_dic,location):

    bin_interval = param_dic["spatial_bin_size"]
    bin_to_exc = param_dic["spat_bins_excluded"]

    # find first and last firing for each trial
    first_firing = np.inf
    last_firing = 0

    for key,value in firing_times.items():
        if value:
            first_firing = int(np.amin([first_firing, np.amin(value)]))
            last_firing = int(np.amax([last_firing, np.amax(value)]))

    # calculate location and speed for each interval
    loc, speed = calc_loc_and_speed(location)

    # trim location to same length as firing data --> remove last location entries
    loc = loc[:(last_firing-first_firing)]

    # trim speed to same length as firing data --> remove last location entries
    speed = speed[:(last_firing-first_firing)]

    #TODO: delete section below
    # plt.plot(speed)
    # plt.plot(loc)
    # plt.hlines(5,0,loc.shape[0])
    # plt.show()
    # len_after_filtering = len([x for x in speed if x >  param_dic["speed_filter"]])
    # print("duration before speed filtering: "+str((last_firing/512 - first_firing/512)*0.0256)+"s")
    # print("duration after speed filtering: "+str(len_after_filtering/512*0.0256)+"s")

    # length of linearized path: 200 cm
    nr_intervals = int(200 / bin_interval)

    # matrix with population vectors
    act_mat = np.zeros([len(firing_times.keys()), nr_intervals])

    # occupation vector for normalization
    occ_vec = np.zeros(nr_intervals)

    for interval in range(nr_intervals):
        # define spatial interval
        start_interval = interval*bin_interval
        end_interval = (interval+1)*bin_interval

        # with speed filtering
        if param_dic["speed_filter"]:
            for loc_ID, loc_ in enumerate(loc):
                if (start_interval <= loc_ < end_interval) and speed[loc_ID] > param_dic["speed_filter"]:
                    occ_vec[interval] += 1

        # without speed filter
        else:
            occ_vec[interval] = len([occ for occ in loc if (start_interval <= occ < end_interval)])

    # go through all cells: cell_ID is not used --> only firing times
    for cell_iter, (cell_ID, cell) in enumerate(firing_times.items()):
        # go through all spatial intervals
        for i in range(nr_intervals):
            # define spatial interval
            start_interval = i*bin_interval
            end_interval = (i+1)*bin_interval
            cell_spikes_interval = 0

            # with speed filtering
            if param_dic["speed_filter"]:
                # go through all spikes and check if they are in the interval and above the speed threshold
                for cell_firing_time in cell:
                    if (start_interval <= loc[(cell_firing_time-first_firing-1)] < end_interval) and \
                            speed[(cell_firing_time-first_firing-1)] > param_dic["speed_filter"]:
                        cell_spikes_interval += 1

                act_mat[cell_iter, i] = cell_spikes_interval
            # without speed filter
            else:
                cell_spikes_interval_array = [x for x in cell if start_interval <= loc[(x-first_firing-1)] < end_interval]
                # write population vectors
                act_mat[cell_iter, i] = len(cell_spikes_interval_array)

    act_mat = act_mat[:, bin_to_exc[0]:bin_to_exc[1]]
    occ_vec = occ_vec[bin_to_exc[0]:bin_to_exc[1]]

    # normalize by time
    act_mat = (act_mat * 20e3) / occ_vec

    # replace NANs with zeros
    act_mat = np.nan_to_num(act_mat)
    loc_vec = np.arange(0, 200, bin_interval)
    loc_vec = loc_vec[bin_to_exc[0]:bin_to_exc[1]]

    return act_mat, loc_vec


def get_activity_mat_time(firing_times,param_dic,location=[]):
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


def unit_vector(vector):
    # Returns the unit vector of the vector
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    #  Returns the angle in radians between vectors 'v1' and 'v2'::
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

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
                D[i,j] = jaccard_similarity_score(pop_vec_ref.astype(int),pop_vec_comp.astype(int))

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

    elif param_dic["dr_method_p1"] == "euclidean":
        # calculate difference matrix: cosine
        D = np.zeros([act_mat.shape[1], act_mat.shape[1]])

        # euclidean distance
        for i, pop_vec_ref in enumerate(act_mat.T):
            for j, pop_vec_comp in enumerate(act_mat.T):
                    D[i, j] = distance.euclidean(pop_vec_ref, pop_vec_comp)

    elif param_dic["dr_method_p1"] == "L1":
        # calculate difference matrix: L1
        D = np.zeros([act_mat.shape[1], act_mat.shape[1]])

        # euclidean distance
        for i, pop_vec_ref in enumerate(act_mat.T):
            for j, pop_vec_comp in enumerate(act_mat.T):
                    D[i, j] = norm(pop_vec_ref-pop_vec_comp,1)

    model = MDS(n_components=param_dic["dr_method_p2"], dissimilarity='precomputed', random_state=1)
    return model.fit_transform(D)


def perform_PCA(act_mat,param_dic):
    # performs PCA
    pca = PCA(n_components=param_dic["dr_method_p2"])
    pca_result = pca.fit_transform(act_mat.T)

    return pca_result, str(pca.explained_variance_ratio_)


def perform_TSNE(act_mat,param_dic):
    # performs TSNE
    return TSNE(n_components=param_dic["dr_method_p2"]).fit_transform(act_mat.T)


def calc_diff(a, b, diff_meas):

    # calculates column-wise difference between two matrices a and b
    D = np.zeros((a.shape[1],b.shape[1]))

    if diff_meas == "jaccard":
        # calculate difference using Jaccard

        # Jaccard similarity
        for i,pop_vec_ref in enumerate(a.T):
            for j,pop_vec_comp in enumerate(b.T):
                D[i,j] = jaccard_similarity_score(pop_vec_ref,pop_vec_comp)

        # want difference --> diff_jaccard = 1 - sim_jaccard
        D = 1 - D
        # plt.imshow(D)
        # plt.colorbar()
        # plt.show()
    elif diff_meas == "cos":
    # calculates column-wise difference between two matrices a and b

        # cosine
        for i, pop_vec_ref in enumerate(a.T):
            for j, pop_vec_comp in enumerate(b.T):
                D[i, j] = distance.cosine(pop_vec_ref, pop_vec_comp)
                # if one of the vectors contains only zeros --> division by zero for cosine
                # if math.isnan(D[i,j]):
                #     D[i, j] = 1

    elif diff_meas == "euclidean":
        # calculate difference matrix: euclidean distance

        # euclidean distance
        for i, pop_vec_ref in enumerate(a.T):
            for j, pop_vec_comp in enumerate(b.T):
                    D[i, j] = distance.euclidean(pop_vec_ref, pop_vec_comp)

    elif diff_meas == "L1":
        # calculate difference matrix: euclidean distance

        # euclidean distance
        for i, pop_vec_ref in enumerate(a.T):
            for j, pop_vec_comp in enumerate(b.T):
                    D[i, j] = norm(pop_vec_ref-pop_vec_comp,1)

    return D


def pop_vec_diff(data_set):
    # computes difference vectors between subsequent population vectors and returns matrix of difference vectors
    # calculate transition vector between two subsequent population states --> rows: cells, col: time bins
    diffMat = np.zeros((data_set.shape[0],data_set.shape[1]-1))
    for i,pop_vec in enumerate(data_set.T[:-1,:]):
        diffMat.T[i,:] = data_set.T[i+1,:] - data_set.T[i,:]
    return diffMat


def pop_vec_dist(data_set, measure):
    # computes euclidean distance between column vectors of data set
    # returns row vector with euclidean distances
    dist_mat = np.zeros(data_set.shape[1]-1)

    for i, _ in enumerate(data_set.T[:-1,:]):
        if measure == "euclidean":
            dist_mat[i] = distance.euclidean(data_set.T[i+1,:],data_set.T[i,:])
        elif measure == "cos":
            dist_mat[i] = distance.cosine(data_set.T[i + 1, :], data_set.T[i, :])
        elif measure == "L1":
            dist_mat[i] = norm(data_set.T[i + 1, :] - data_set.T[i, :],1)

    # calculate relative change between subsequent vectors
    rel_dist_mat = np.zeros(dist_mat.shape[0] - 1)

    for i, _ in enumerate(dist_mat[:-1]):
        rel_dist_mat[i] = abs(1- dist_mat[i+1]/dist_mat[i])

    return dist_mat, rel_dist_mat


def angle_between_col_vectors(data_set):
    # computes angle between two subsequent transitions --> transitions: from one pop-vec to the next
    # returns row vector with angles in radiant
    angle_mat = np.zeros(data_set.shape[1]-1)

    for i,_ in enumerate(data_set.T[:-1,:]):
        angle_mat[i] = angle_between(data_set.T[i+1,:],data_set.T[i,:])

    # calculate relative change between subsequent vectors
    rel_angle_mat = np.zeros(angle_mat.shape[0] - 1)

    for i, _ in enumerate(angle_mat[:-1]):
        rel_angle_mat[i] = abs(1- angle_mat[i+1]/angle_mat[i])

    return angle_mat, rel_angle_mat


def calc_loc_and_speed(whl):
    # computes speed from the whl and returns speed in cm/s
    # need to smooth position data --> accuracy of measurement: about +-1cm --> error for speed: +-40m/s
    # last element of velocity vector is zero --> velocity is calculated using 2 locations

    #savitzky golay
    w_l = 31 # window length
    p_o = 5 # order of polynomial
    whl = signal.savgol_filter(whl, w_l, p_o)

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


def calc_cohens_d(dat1, dat2):

    # calculates cohens D --> assumption of normal distributions
    pooled_std = np.sqrt(((dat1.shape[1]-1)*np.std(dat1,axis=1)**2+(dat2.shape[1]-1)*np.std(dat2,axis=1)**2)/
                         (dat1.shape[1]+dat2.shape[1]-2))
    # add small value to avoid division by zero
    pooled_std += 0.000001

    diff_avg = np.average(dat1,axis=1) - np.average(dat2,axis=1)

    return diff_avg/pooled_std


