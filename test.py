
from scipy.spatial import distance
import numpy as np
from comp_functions import calc_diff
import matplotlib.pyplot as plt
plt.style.use('dark_background')

def synthetic_remapping():

    m_d = 1
    s_d = 5

    mu = np.log(m_d**2/np.sqrt(m_d**2+s_d**2))
    si = np.sqrt(np.log(s_d**2/m_d**2+1))

    # choose lognormal distribution
    a = np.random.lognormal(mu,si,100)
    a /= np.sum(a)
    nr_cells = a.shape[0]

    remap_nr = 10

    avg_distance = np.zeros((remap_nr,nr_cells))

    nr_remapping_cells_array = np.linspace(1,50, remap_nr, dtype=int)

    for nr_remapping_cells in range(remap_nr):
        b = a.copy()
        b[:nr_remapping_cells_array[nr_remapping_cells]+1] = b[:nr_remapping_cells_array[nr_remapping_cells]+1] +2.8

        b /=np.sum(b)

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
        plt.plot(avg_distance[i,:], label=str(nr_remapping_cells_array[i])+" remapped cells", marker="o")

    plt.legend()
    plt.xlabel("NR. CELLS IN SUBSET")
    plt.ylabel("COS DIFFERENCE")
    plt.title("REMAPPING CHARACTERISTICS - SYNTHETIC DATA SET")
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
        v2[ind_to_remap] = 0# += np.random.random()#*n_remap

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

    for n_remap in range(1,5):
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


def influence_of_dim_shuffling():

    frac_to_remap = 2

    m_d = 1
    s_d = 5

    mu = np.log(m_d ** 2 / np.sqrt(m_d ** 2 + s_d ** 2))
    si = np.sqrt(np.log(s_d ** 2 / m_d ** 2 + 1))

    # choose lognormal distribution
    a = np.random.lognormal(mu, si, 100)
    a = np.ones(100)
    a /= np.sum(a)
    nr_cells = a.shape[0]

    remap_nr = 10

    avg_distance = np.zeros(nr_cells)


    # different subsets of cells
    for nr_cells_subset in range(nr_cells):

        nr_cells_to_remap = int(nr_cells_subset/frac_to_remap)

        b = a.copy()
        b[:nr_cells_to_remap + 1] = b[:nr_cells_to_remap +1 ] + 1
        b /= np.sum(b)

        # do random selection n times
        n_r_s = 1000

        cos_distance = np.zeros(n_r_s)

        for random_selection in range(n_r_s):
            # permuted index
            per_ind = np.random.permutation(np.arange(nr_cells))
            subset_ind = per_ind[:nr_cells_subset + 1]
            # print("a: "+str(a[subset_ind]))
            # print("b: "+str(b[subset_ind]))

            cos_distance[random_selection] = distance.cosine(a[subset_ind], b[subset_ind])

        avg_distance[nr_cells_subset] = np.nanmean(cos_distance)


    plt.plot(avg_distance)

    plt.xlabel("NR. CELLS IN SUBSET")
    plt.ylabel("COS DIFFERENCE")
    plt.title("INFLUENCE OF DIMENSIONALITY")
    plt.show()

def influence_of_dim_measures():
    a = np.ones(100)

    b = a.copy()

    cos_dis = np.zeros(a.shape[0])
    euc_dis = np.zeros(a.shape[0])
    jac_dis = np.zeros(a.shape[0])
    L1_dis = np.zeros(a.shape[0])
    for i in range(2,a.shape[0]):
        short_a = a[:i].copy()
        short_b = b[:i].copy()
        short_b[:int(short_b.shape[0]/2)] += 3
        print(short_b, short_a)
        cos_dis[i] = distance.cosine(short_a, short_b)
        euc_dis[i] = distance.euclidean(short_a, short_b)
        jac_dis[i] = distance.jaccard(short_a, short_b)
        L1_dis[i] = np.linalg.norm((short_a - short_b), ord=1)

    plt.plot(cos_dis / max(cos_dis), label="cos")
    plt.plot(euc_dis / max(euc_dis), label="euc")
    plt.plot(L1_dis / max(L1_dis), label="L1")
    plt.plot(jac_dis / max(jac_dis), label="jac")
    plt.title("300% INCREASE IN FIRING RATE IN HALF OF THE CELLS")
    plt.xlabel("#CELLS IN SUBSET")
    plt.ylabel("NORM. DISTANCE")
    plt.legend()
    plt.show()

def compare_measures():
    a = np.ones(100)

    b = a.copy()

    cos_dis = np.zeros(a.shape[0])
    euc_dis = np.zeros(a.shape[0])
    jac_dis = np.zeros(a.shape[0])
    L1_dis = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        b[i] += 2
        cos_dis[i] = distance.cosine(a, b)
        euc_dis[i] = distance.euclidean(a, b)
        jac_dis[i] = distance.jaccard(a, b)
        L1_dis[i] = np.linalg.norm((a - b), ord=1)

    plt.plot(cos_dis / max(cos_dis), label="cos")
    plt.plot(euc_dis / max(euc_dis), label="euc")
    plt.plot(L1_dis / max(L1_dis), label="L1", linewidth=3)
    plt.plot(jac_dis / max(jac_dis), label="jac")
    plt.title("INCREASE IN FIRING RATE: 300%")
    plt.xlabel("#CHANGED CELLS / %")
    plt.ylabel("NORM. DISTANCE")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # synth_remapping_fred()
    synthetic_remapping()
    #influence_of_dim_measures()
    # compare_measures()