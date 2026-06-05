from utils.datas.domain17 import *
from utils.fitting_algos.law2init import *
M_data=[200]*h_data.shape[0]
N_data=[2500]*h_data.shape[0]

M_data=np.array(M_data)
N_data=np.array(N_data)
for seed in [234,12,5]:
    law2(h_data, L_data, N_data, M_data, seed=seed,MIN_X=0.3,N_BASINS=6)