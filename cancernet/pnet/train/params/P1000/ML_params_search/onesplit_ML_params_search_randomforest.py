import os
from os.path import dirname

from sklearn.model_selection import ParameterGrid

base_dirname = dirname(dirname(__file__))
print(base_dirname)
filename = os.path.basename(__file__)
task = 'classification_binary'

selected_genes = '~/DATA/P1000/tcga_prostate_expressed_genes_and_cancer_genes.csv'
data_base = {'id': 'ALL', 'type': 'prostate_paper',
             'params': {
                 'data_type': ['mut_important', 'cnv_del', 'cnv_amp'],
                 'drop_AR': False,
                 'cnv_levels': 3,
                 'mut_binary': True,
                 'balanced_data': False,
                 'combine_type': 'union',
                 'use_coding_genes_only': True,
                 'selected_genes': selected_genes,
                 'training_split': 0,
             }
             }
data = [data_base]

pre = {'type': None}
features = {}

logistic_params = []
print(len(logistic_params))
class_weight = {0: 0.75, 1: 1.5}
forest_params = []

param_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 30, 50, 70, None],
    'n_estimators': [10, 50, 100, 200],
    'class_weight': [class_weight]
}

param_grid_list = list(ParameterGrid(param_grid))
for i, param in enumerate(param_grid_list):
    forest_params.append({'type': 'random_forest', 'id': 'Random Forest_{}'.format(i), 'params': param})

models = forest_params
pipeline = {'type': 'one_split', 'params': {'save_train': True, 'eval_dataset': 'validation'}}
