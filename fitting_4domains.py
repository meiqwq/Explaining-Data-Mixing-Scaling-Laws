from utils.datas.domain4 import *
from utils.fitting_algos.law2 import *
M_data=[200]*h_data.shape[0]
N_data=[800]*h_data.shape[0]

M_data=np.array(M_data)
N_data=np.array(N_data)
for seed in range(100):
    law2(h_data, L_data, N_data, M_data, seed=seed+114514, MIN_X=1e-6, BH_TEMPERATURE=1e-6,N_BASINS=5,STEP_SIZE=0.1)