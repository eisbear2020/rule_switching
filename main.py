########################################################################################################################
#
#   NEURAL ACTIVITY ANALYSIS
#
#
#   Author: Lars Bollmann
#
#   Created: 07/03/2019
#
#   Structure:
#
#
#
#
#
#
#
########################################################################################################################

import numpy as np
from helper_func import getCellID

if __name__ == '__main__':

    # data directory
    data_dir = "../02 Data"

    # select experiment
    s_exp = "mjc189-1905-0517"

    # load data
    # to get rate map:
    # data["binmat"][SESSION][CELL][TIME INTERVAL]
    # TIME INTERVAL from data["tist"]
    # data = np.load("../02 Data/"+s_exp+"/data.npy").item()
    # x = data['binmat'][2][3][853:1514]
    # print(x)

    cell_type_array = ["p2", "p3"]

    pfc_cell_IDs = getCellID(data_dir, s_exp, cell_type_array)

    # select environment:
    # 2: first session
    # 4: session after sleep (with rule switch)
    # 6: last session

    env = "2"

    timestamps = np.loadtxt(data_dir+"/"+s_exp+"/"+s_exp+"_"+env+".timestamps").astype(int)
    print(timestamps)



# for s in ses:
#     print(s)
#
#     with open(s + '/' + s + ".des") as f:
#         des = f.read()
#     des = des.splitlines()
#
#     pp_cells = [i + 2 for i in range(len(des)) if des[i] == 'p2' or des[i] == 'p3']
#     p1_cells = [i + 2 for i in range(len(des)) if des[i] == 'p1']
#     p2_cells = [i + 2 for i in range(len(des)) if des[i] == 'p2']
#     p3_cells = [i + 2 for i in range(len(des)) if des[i] == 'p3']
#
#     data = {'pp_cells': pp_cells,
#             'p1_cells': p1_cells,
#             'p2_cells': p2_cells,
#             'p3_cells': p3_cells,
#             'des': des,
#             'tist': {},
#             'lin_whl': {},
#             'speed': {},
#             'binmat': {}}
#     for env in [2, 4, 6]:
#         whl = np.loadtxt(s + '/' + s + '_' + str(env) + ".whl_rot")
#         # new timestamps
#         timestamps = np.loadtxt('new_timestamps/' + s + '_' + str(env) + '.timestamps').astype(int)
#         data['tist'][env] = timestamps
#         # save also linearized whl
#         data['lin_whl'][env], data['speed'][env] = linearize_whl(whl, timestamps)
#
#         clu = np.loadtxt(s + '/' + s + '_' + str(env) + '.clu', dtype=int)
#         res = np.loadtxt(s + '/' + s + '_' + str(env) + '.res', dtype=int)
#         spkls = r.clures_to_spkl(clu, res)
#         binmat = np.zeros((len(des) + 2, len(data['speed'][env])))
#         # save matrix binned as whl (25.6 ms)
#
#         for i in range(2, len(des) + 2):
#             for j in spkls[i]:
#                 binmat[i, j] += 1
#         data['binmat'][env] = binmat
#     np.save(s + '/data.npy', data)

