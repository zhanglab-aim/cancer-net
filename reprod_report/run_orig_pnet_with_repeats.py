"""
call `run_me.py` multiple times and extract performance metric each time
"""

import os
import pandas as pd
import pickle

PNET_DIR = "/mnt/ceph/users/zzhang/pnet_prostate_paper/"
PNET_LOG_DIR = PNET_DIR + "_logs/p1000/pnet/onsplit_average_reg_10_tanh_large_testing/"
PNET_EXEC = PNET_DIR + "py3/train/run_me.py"
METHODS = ['LogisticRegression', 'P-net']
TEST_SET = PNET_DIR + "_database/prostate/splits/test_set.csv"

REPEATS = 30

test_idx = pd.read_csv(TEST_SET, index_col=0)
method_dfs = {x:[] for x in METHODS}

for _ in range(REPEATS):
    # call external p-net trainer
    os.system(f"python {PNET_EXEC} {_}")

    # read in evals
    for method in METHODS:
        fp = PNET_LOG_DIR + f"{method}_ALL_testing.csv"
        df = pd.read_csv(fp, index_col=0)
        df = df.loc[test_idx.id]
        method_dfs[method].append(df)

    # merge
    tot_dfs = {}
    for method, df_list in method_dfs.items():
        total = pd.concat([x[['pred', 'pred_scores']] for x in df_list] + [df_list[0]['y']], axis=1)
        tot_dfs[method] = total

    with open("orig_pnet.pkl", "wb") as f:
        pickle.dump(tot_dfs, f)
    
    # clean-up
    os.system(f"rm -r {PNET_LOG_DIR}")