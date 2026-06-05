from utils.datas.domain17 import *
from utils.fitting_algos.law1 import *
for seed in [11414,324,423,54,32]:
    law1(h_data, L_data,K=200,MIN_X=0.3,use_13train=False,seed=seed,N_BASINS=10)   