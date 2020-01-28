########################################################################################################################
#
#   SPIKE DATA
#
#   Description: contains classes that are used to perform analysis on spike data
#
#   Author: Lars Bollmann
#
#   Created: 22/07/2019
#
#   Structure:
#
#
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import OrderedDict

########################################################################################################################
#   SPIKE DATA BASE CLASS
########################################################################################################################


class SpikeData:
    ''' Base class for spike data analysis'''

    def __init__(self, firing_times, loc_data):

        # location data
        self.location = loc_data

        # binning method
        self.firing_times = firing_times

        # speed
        self.speed = []

        # calculate speed and upsample location data
        self.calc_loc_and_speed()

        # rate map
        self.rate_map = []
        self.rate_map_locations = []

    def calc_loc_and_speed(self):
        # computes speed from the whl and returns speed in cm/s and upsamples location data to match spike timing
        # need to smooth position data --> accuracy of measurement: about +-1cm --> error for speed: +-40cm/s
        # last element of velocity vector is zero --> velocity is calculated using 2 locations

        # savitzky golay
        w_l = 31  # window length
        p_o = 5  # order of polynomial
        whl = signal.savgol_filter(self.location, w_l, p_o)

        # one time bin: whl is recorded at 20kHz/512
        t_b = 1 / (20e3 / 512)

        # upsampling to synchronize with spike data
        location = np.zeros(whl.shape[0] * 512)
        for i, loc in enumerate(whl):
            location[512 * i:(i + 1) * 512] = 512 * [loc]

        # calculate speed: x1-x0/dt
        temp_speed = np.zeros(whl.shape[0] - 1)
        for i in range(temp_speed.shape[0]):
            temp_speed[i] = (whl[i + 1] - whl[i]) / t_b

        # smoothen speed using savitzky golay
        temp_speed = signal.savgol_filter(temp_speed, 15, 5)

        # upsampling to synchronize with spike data
        speed = np.zeros(whl.shape[0] * 512)
        for i, bin_speed in enumerate(temp_speed):
            speed[512 * i:(i + 1) * 512] = 512 * [bin_speed]

        self.location = location
        self.speed = speed

    def plot_loc_and_speed(self):

        # plotting
        t = np.arange(len(self.speed))
        plt.plot(t/20e3, self.speed,label="speed")
        plt.plot(t/20e3,self.location, label = "location")
        plt.plot([0, t[-1]/20e3],[5,5], label = "threshold")
        plt.xlabel("time / s")
        plt.ylabel("location / cm - speed / (cm/s)")
        plt.legend()
        plt.show()

    def time_binning(self, bin_interval, speed_filter, bool_zero_filter, bool_plot_vel_and_data = False):
        # computes activity matrix: bin_interval in seconds --> sums up the activity within one time interval
        # rows: cells
        # columns: time bins

        # find first and last firing for each trial
        first_firing = np.inf
        last_firing = 0

        for key,value in self.firing_times.items():
            if value:
                first_firing = int(np.amin([first_firing, np.amin(value)]))
                last_firing = int(np.amax([last_firing, np.amax(value)]))

        # trim location to same length as firing data --> remove last location entries
        loc = self.location[:(last_firing-first_firing)]

        # trim speed to same length as firing data --> remove last location entries
        speed = self.speed[:(last_firing-first_firing)]

        if bool_plot_vel_and_data:
            fig, (ax1,ax2) = plt.subplots(2, 1)

            t = np.arange(len(speed))
            ax1.plot(t / 20e3, speed, label="SPEED")
            ax1.plot(t / 20e3, loc, label="LOCATION")
            ax1.set_xlabel("TIME / s")
            ax1.set_ylabel("LOCATION / cm - SPEED / (cm/s)")
            ax1.plot([0, t[-1] / 20e3], [speed_filter, speed_filter], label="THRESHOLD")
            len_after_filtering = len([x for x in speed if x > speed_filter])
            print("duration before speed filtering: "+str((last_firing/512 - first_firing/512)*0.0256)+"s")
            print("duration after speed filtering: "+str(len_after_filtering/512*0.0256)+"s")

        # duration of trial (one time bin: 0.05ms --> 20kHz)
        dur_trial = (last_firing-first_firing)* 0.05*1e-3
        nr_intervals = int(dur_trial/bin_interval)
        size_intervals = int((last_firing-first_firing)/nr_intervals)
        size_interval_sec = size_intervals* 0.05*1e-3

        # matrix with population vectors
        act_mat = np.zeros([len(self.firing_times.keys()),nr_intervals])
        # loc vector
        loc_vec = np.zeros(nr_intervals)
        # speed vector
        speed_vec = np.zeros(nr_intervals)

        # go through all cells: cell_ID is not used --> only firing times
        for cell_iter, (cell_ID, cell) in enumerate(self.firing_times.items()):
            # go through all temporal intervals
            for i in range(nr_intervals):
                start_interval = first_firing+i*size_intervals
                end_interval = first_firing+(i+1)*size_intervals
                cell_spikes_interval = 0
                if bool_plot_vel_and_data:
                    ax1.axvline((end_interval- first_firing - 1)/20e3, label = "BINS")

                # go through all spikes and check if they are in the interval and above the speed threshold
                for cell_firing_time in cell:

                    if (start_interval <= cell_firing_time < end_interval) and \
                            speed[(cell_firing_time - first_firing - 1)] > speed_filter:
                        if bool_plot_vel_and_data:
                            ax1.scatter((cell_firing_time - first_firing - 1)/ 20e3, 1, marker=".", c="red", label=
                                        "SELECTED SPIKES")
                        cell_spikes_interval += 1
                act_mat[cell_iter, i] = cell_spikes_interval/size_interval_sec

                loc_vec[i] = np.mean(loc[(start_interval - first_firing):(end_interval - first_firing)])
                speed_vec[i] = np.mean(speed[(start_interval - first_firing):(end_interval - first_firing)])

        if bool_zero_filter:
            # set all to vectors to False
            int_sel = np.full(act_mat.shape[1], False)
            # go through all population vectors
            for i,pop_vec in enumerate(act_mat.T):
                if np.count_nonzero(pop_vec):
                    int_sel[i] = True
            act_mat = act_mat[:, int_sel]
            loc_vec = loc_vec[int_sel]

        if bool_plot_vel_and_data:

            handles, labels = ax1.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys())
            ax1.set_xlim(0, t[-1] / 20e3)
            im = ax2.imshow(act_mat, interpolation='nearest', aspect='auto')
            cbar = fig.colorbar(im, orientation="horizontal", pad=0.4)
            cbar.set_label("FIRING RATE / Hz")
            ax2.set_xlabel("TIME BIN ID")
            ax2.set_ylabel("CELL ID")
            plt.show()

        return act_mat, loc_vec

