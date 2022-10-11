import sys
from os.path import join, dirname, realpath

current_dir = dirname(realpath(__file__))
sys.path.insert(0, dirname(current_dir))
import os
import importlib
import logging
import random
import timeit
import datetime
import numpy as np
import tensorflow as tf

from cancernet.pnet.utils.logs import set_logging, DebugFolder
from cancernet.pnet.config_path import PROSTATE_LOG_PATH, POSTATE_PARAMS_PATH
from cancernet.pnet.pipeline.train_validate import TrainValidatePipeline
from cancernet.pnet.pipeline.one_split import OneSplitPipeline
from cancernet.pnet.pipeline.crossvalidation_pipeline import CrossvalidationPipeline
from cancernet.pnet.pipeline.LeaveOneOut_pipeline import LeaveOneOutPipeline

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

random_seed = 234
random.seed(random_seed)
np.random.seed(random_seed)
# tf.random.set_random_seed(random_seed)

timeStamp = "_{0:%b}-{0:%d}_{0:%H}-{0:%M}".format(datetime.datetime.now())


def elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


params_file_list = []

# pnet
params_file_list.append(".pnet.onsplit_average_reg_10_tanh_large_testing")
# params_file_list.append('.pnet.onsplit_average_reg_10_tanh_large_testing_inner')
# params_file_list.append('.pnet.crossvalidation_average_reg_10_tanh')
#
# # other ML models
# params_file_list.append('.compare.onsplit_ML_test')
# params_file_list.append('.compare.crossvalidation_ML_test')
#
# # dense
# params_file_list.append('.dense.onesplit_number_samples_dense_sameweights')
# params_file_list.append('.dense.onsplit_dense')
#
# # number_samples
# params_file_list.append('.number_samples.crossvalidation_average_reg_10')
## params_file_list.append('.number_samples.crossvalidation_average_reg_10_tanh')
# params_file_list.append('.number_samples.crossvalidation_number_samples_dense_sameweights')
#
# # external_validation
# params_file_list.append('.external_validation.pnet_validation')
#
# #reviews------------------------------------
# #LOOCV
# params_file_list.append('.review.LOOCV_reg_10_tanh')
# #ge
# params_file_list.append('.review.onsplit_average_reg_10_tanh_large_testing_ge')
# #fusion
# params_file_list.append('.review.fusion.onsplit_average_reg_10_tanh_large_testing_TMB')
# params_file_list.append('.review.fusion.onsplit_average_reg_10_tanh_large_testing_fusion')
# params_file_list.append('.review.fusion.onsplit_average_reg_10_tanh_large_testing_fusion_zero')
# params_file_list.append('.review.fusion.onsplit_average_reg_10_tanh_large_testing_inner_fusion_genes')
#
# #single copy
# params_file_list.append('.review.9single_copy.onsplit_average_reg_10_tanh_large_testing_single_copy')
# params_file_list.append('.review.9single_copy.crossvalidation_average_reg_10_tanh_single_copy')
#
# #custom arch
# params_file_list.append('.review.10custom_arch.onsplit_kegg')
#
# #learning rate
# params_file_list.append('.review.learning_rate.onsplit_average_reg_10_tanh_large_testing_inner_LR')

# hotspot
# params_file_list.append('.review.9hotspot.onsplit_average_reg_10_tanh_large_testing_hotspot')
# params_file_list.append('.review.9hotspot.onsplit_average_reg_10_tanh_large_testing_count')

# cancer genes
# params_file_list.append('.review.onsplit_average_reg_10_tanh_large_testing')
# params_file_list.append('.review.onsplit_average_reg_10_cancer_genes_testing')
# params_file_list.append('.review.crossvalidation_average_reg_10_tanh_cancer_genes')

# review 2 (second iteration of reviews)
# params_file_list.append('.review.cnv_burden_training.onsplit_average_reg_10_tanh_large_testing_TMB2')
# params_file_list.append('.review.cnv_burden_training.onsplit_average_reg_10_tanh_large_testing_account_zero2')
# params_file_list.append('.review.cnv_burden_training.onsplit_average_reg_10_tanh_large_testing_TMB_cnv')
# params_file_list.append('.review.cnv_burden_training.onsplit_average_reg_10_tanh_large_testing_cnv_burden2')

for params_file in params_file_list:
    log_dir = join(PROSTATE_LOG_PATH, params_file)
    log_dir = log_dir
    set_logging(log_dir)
    # params_file = join(POSTATE_PARAMS_PATH, params_file)
    logging.info("random seed %d" % random_seed)
    # params_file_full = params_file + ".py"
    params_file_full = "cancernet.pnet.train.params.P1000" + params_file
    print(params_file_full)
    # params = imp.load_source(params_file, params_file_full)
    params = importlib.import_module(params_file_full)

    DebugFolder(log_dir)
    if params.pipeline["type"] == "one_split":
        pipeline = OneSplitPipeline(
            task=params.task,
            data_params=params.data,
            model_params=params.models,
            pre_params=params.pre,
            feature_params=params.features,
            pipeline_params=params.pipeline,
            exp_name=log_dir,
        )

    elif params.pipeline["type"] == "crossvalidation":
        pipeline = CrossvalidationPipeline(
            task=params.task,
            data_params=params.data,
            feature_params=params.features,
            model_params=params.models,
            pre_params=params.pre,
            pipeline_params=params.pipeline,
            exp_name=log_dir,
        )
    elif params.pipeline["type"] == "Train_Validate":
        pipeline = TrainValidatePipeline(
            data_params=params.data,
            model_params=params.models,
            pre_params=params.pre,
            feature_params=params.features,
            pipeline_params=params.pipeline,
            exp_name=log_dir,
        )

    elif params.pipeline["type"] == "LOOCV":
        pipeline = LeaveOneOutPipeline(
            task=params.task,
            data_params=params.data,
            feature_params=params.features,
            model_params=params.models,
            pre_params=params.pre,
            pipeline_params=params.pipeline,
            exp_name=log_dir,
        )
    start = timeit.default_timer()
    pipeline.run()
    stop = timeit.default_timer()
    mins, secs = elapsed_time(start, stop)
    logging.info("Elapsed Time: {}m {}s".format(mins, secs))