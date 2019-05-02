
from scipy.spatial import distance
import numpy as np
from comp_functions import calc_diff
import matplotlib.pyplot as plt

a = np.random.rand(100,1)
b = a.copy()
nr_cells = a.shape[0]

remap_nr = 10

avg_distance = np.zeros((remap_nr,nr_cells))

nr_remapping_cells_array = np.linspace(1,nr_cells, remap_nr, dtype=int)
print(nr_remapping_cells_array.shape)

for nr_remapping_cells in range(remap_nr):

    for i in range(nr_remapping_cells_array[nr_remapping_cells]):
        a[i] = 1

    # different subsets of cells
    for nr_cells_subset in range(nr_cells):

        # do random selection n times
        n_r_s = 100

        cos_distance = np.zeros(n_r_s)

        for random_selection in range(n_r_s):

            # permuted index
            per_ind = np.random.permutation(np.arange(nr_cells))
            subset_ind = per_ind[:nr_cells_subset + 1]
            # print("a: "+str(a[subset_ind]))
            # print("b: "+str(b[subset_ind]))

            cos_distance[random_selection] = distance.cosine(a[subset_ind],b[subset_ind])

        avg_distance[nr_remapping_cells, nr_cells_subset] = np.nanmean(cos_distance)


for i in range(avg_distance.shape[0]):
    plt.plot(avg_distance[i,:], label=str(nr_remapping_cells_array[i])+" remapped cells")

plt.legend()
plt.show()


