
from scipy.spatial import distance
import numpy as np
from comp_functions import calc_diff
import matplotlib.pyplot as plt


def synthetic_remapping():

    m_d = 1
    s_d = 5

    mu = np.log(m_d**2/np.sqrt(m_d**2+s_d**2))
    si = np.sqrt(np.log(s_d**2/m_d**2+1))

    # choose lognormal distribution
    a = np.random.lognormal(mu,si,100)
    a /= np.sum(a)
    nr_cells = a.shape[0]

    remap_nr = 5

    avg_distance = np.zeros((remap_nr,nr_cells))

    nr_remapping_cells_array = np.linspace(1,50, remap_nr, dtype=int)

    for nr_remapping_cells in range(remap_nr):
        b = np.random.permutation(a.copy())
        b[:nr_remapping_cells_array[nr_remapping_cells]+1] = b[:nr_remapping_cells_array[nr_remapping_cells]+1]+1/(
                                                                     nr_remapping_cells_array[nr_remapping_cells]+1)
        b /=np.sum(b)

        # different subsets of cells
        for nr_cells_subset in range(nr_cells):

            # do random selection n times
            n_r_s = 400

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
        plt.plot(avg_distance[i,:], label=str(nr_remapping_cells_array[i])+" remapped cells", marker="o")

    plt.legend()
    plt.show()


def synth_remapping():

    n_cells = 100
    n_rand = 400

    m_d = 1
    s_d = 5

    mu = np.log(m_d**2/np.sqrt(m_d**2+s_d**2))
    si = np.sqrt(np.log(s_d**2/m_d**2+1))

    # original population vectors
    v1 = np.random.lognormal(mu, si, n_cells)

    # how many cells are remapped
    for n_remap in np.arange(1, 5, 1):
        # remap cells
        v2 = v1.copy()
        ind_to_remap = np.random.randint(0,v2.shape[0],n_remap)
        v2[ind_to_remap] += np.random.random()#*n_remap

        cos_distance = np.zeros((10, n_rand))

        # how many cells are selected for shuffling
        for i,subset in enumerate(np.arange(1, 100, 10)):
            for random_selection in range(n_rand):
                # permuted index
                ss=np.random.permutation(n_cells)

                cos_distance[i,random_selection] = distance.cosine(v1[ss[:subset + 1]],v2[ss[:subset + 1]])

        print(np.nanmean(cos_distance,axis=1))

        plt.plot(np.nanmean(cos_distance,axis=1), label="Nr. of cells: "+str(n_remap))

    plt.legend()
    plt.show()

def synth_remapping_fred():

    n_cells = 100
    n_rand = 400

    m_d = 1
    s_d = 5

    mu = np.log(m_d**2/np.sqrt(m_d**2+s_d**2))
    si = np.sqrt(np.log(s_d**2/m_d**2+1))

    for n_remap in range(1,80):
        v1 = np.random.lognormal(mu,si,n_cells)
        #normalizee
        v1 /= np.sum(v1)
        v2 =  v1.copy()

        v2 += np.random.rand(v2.shape[0])*n_remap/50
        v2 /= np.sum(v2)

        d_vec = np.zeros((10,n_rand))

        c = -1
        for cc in np.arange(1,100,10):
            c += 1

            for e in range(0,n_rand-1):
                ss=np.random.permutation(n_cells)

                d_vec[c,e] = distance.cosine(v1[ss[:cc]],v2[ss[:cc]])

        plt.plot(np.nanmean(d_vec,axis=1), label="Nr. of cells: "+str(n_remap))

    plt.legend()
    plt.show()


if __name__ == '__main__':
    synth_remapping_fred()
    #synth_remapping()