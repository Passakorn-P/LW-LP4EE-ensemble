# =============================================================================
# configs.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : Central configuration shared by every stage: datasets, kernels,
#            bandwidths, the named kernel/bandwidth subsets of RQ3, the
#            adopted ensemble of RQ4, the hyper-parameter grids of the
#            baselines, statistical-test thresholds, and all input/output
#            paths (relative to the codes/ directory).
# =============================================================================

import math
import os
from sklearn import svm, neural_network, tree


class Configurations:

    DATASETS_PATH = '../datasets/'
    CACHE_PATH = '../results/lw_cache/'
    EXP_PATH = '../results/exp_from_cache/'
    BASELINE_PATH = '../results/baseline_results/'
    PLOT_PATH = '../results/plot/'
    TABLE_PATH = '../results/tables/'
    LP_R_FILE = 'models/lp.R'

    EXP1_FILE = EXP_PATH + 'exp1_fixed_all_kernels.parquet'
    EXP2_FILE = EXP_PATH + 'exp2_cv_all_subsets_kernels_full.parquet'
    EXP3_FILE = EXP_PATH + 'exp3_ensemble_full.parquet'

    DATASET_NAMES = ['Cocomo81-e', 'Cocomo81-non-e', 'Desharnais', 'Kitchenham', 'Maxwell', 'Miyazaki94', 'Nasa93-c2', 'Nasa93-c5']

    METRICS = ['MAE', 'MdAE', 'LSD']

    KERNELS = ('rectangular', 'epanechnikov', 'tricube', 'gaussian', 'triangle', 'triweight', 'biweight', 'cosine', 'logistic', 'sigmoid')

    COMPACT_KERNELS = frozenset({'rectangular', 'epanechnikov', 'tricube', 'triangle', 'triweight', 'biweight', 'cosine'})
    BANDWIDTHS = tuple(round(value / 10, 1) for value in range(1, 11))

    N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
    CACHE_OUTER_COLUMNS = ['dataset', 'index', 'actual', 'kernel', 'bandwidth', 'pred']
    CACHE_NESTED_COLUMNS = ['dataset', 'outer_index', 'inner_index', 'actual', 'kernel', 'bandwidth', 'pred']

    NORMALIZATIONS = ('nothing', 'minmax', 'ln')
    FEATURE_SELECTIONS = ('all', 'rfe', 'pca')

    NAMED_KERNEL_SUBSETS = {
        'All$_{10}$': ('rectangular', 'epanechnikov', 'tricube', 'gaussian', 'triangle', 'triweight', 'biweight', 'cosine', 'logistic', 'sigmoid'),
        'Top$_3$': ('epanechnikov', 'triangle', 'cosine'),
        'Top$_4$': ('rectangular', 'epanechnikov', 'triangle', 'cosine'),
        'Top$_5$': ('rectangular', 'epanechnikov', 'gaussian', 'triangle', 'cosine'),
        'Top$_6$': ('rectangular', 'epanechnikov', 'gaussian', 'triangle', 'cosine', 'logistic'),
        'Diverse$_3$': ('rectangular', 'cosine', 'sigmoid'),
        'Diverse$_4$': ('rectangular', 'epanechnikov', 'gaussian', 'sigmoid'),
        'Diverse$_5$': ('rectangular', 'epanechnikov', 'gaussian', 'triangle', 'logistic'),
        'Diverse$_6$': ('rectangular', 'epanechnikov', 'gaussian', 'triangle', 'logistic', 'sigmoid'),
        'Similar$_3$': ('epanechnikov', 'biweight', 'triweight'),
    }
    NAMED_BANDWIDTH_SUBSETS = {
        'All$_{10}$': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        'Low$_1$+High$_1$': (0.1, 1.0),
        'Low$_2$+High$_1$': (0.1, 0.2, 1.0),
        'Low$_3$+High$_1$': (0.1, 0.2, 0.3, 1.0),
        'Low$_4$': (0.1, 0.2, 0.3, 0.4),
        'Low$_1$+High$_2$': (0.1, 0.9, 1.0),
        'Low$_1$+High$_3$': (0.1, 0.8, 0.9, 1.0),
        'High$_4$': (0.7, 0.8, 0.9, 1.0),
        'Mid$_4$': (0.4, 0.5, 0.6, 0.7),
        'Low$_2$+High$_2$': (0.1, 0.2, 0.9, 1.0),
        'Spread$_4$': (0.1, 0.4, 0.7, 1.0),
    }
    NAMED_COL_SEPARATORS = ('All$_{10}$', 'Low$_4$', 'High$_4$')

    ENSEMBLE_KERNEL_SUBSET = 'rectangular-cosine-sigmoid'
    ENSEMBLE_BANDWIDTH_SUBSET = '1-9-10'

    BASELINE_MODEL_NAMES = ['LP4EE', 'LP4EE_SROpt', 'ABE', 'ANN', 'CART',
                            'GBM', 'RF', 'SVR']

    COMPARE_NAMES = {
        'Ensemble': 'LW-LP4EE',
        'LP4EE': 'Orig.LP4EE',
        'LP4EE_SROpt': 'LP4EE$_{SROpt}$',
        'ABE': 'ABE',
        'CART': 'CART',
        'ANN': 'ANN',
        'SVR': 'SVR',
        'GBM': 'GBM',
        'RF': 'RF',
    }

    P_THRESHOLD = 0.05
    DELTA_THRESHOLD = 0.147

    @staticmethod
    def get_baseline_params(target_size):
        max_power = int(math.log2(target_size))
        hidden_layer_sizes = [int(2 ** n) for n in range(1, max_power)]
        max_depth_values = list(range(2, math.ceil(math.sqrt(target_size)) + 1))
        n_estimators_values = [int(2 ** n) for n in range(1, max_power + 1)]
        norm_all = ['nothing', 'minmax', 'ln']
        feat_all = ['all', 'rfe', 'pca']

        return {
            'LP4EE_SROpt': {
                'model': None,
                'params': {
                    'deviation_fraction': [0.1 * 2 ** j for j in range(11)],
                    'n_levels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'norm': ['minmax'],
                    'feat': ['all'],
                },
            },
            'LP4EE': {
                'model': None,
                'params': {'norm': ['minmax'], 'feat': ['all']},
            },
            'ABE': {
                'model': None,
                'params': {
                    'k': [1, 2, 3, 4, 5],
                    'metric': ['euclidean', 'manhattan'],
                    'norm': norm_all,
                    'feat': feat_all,
                },
            },
            'SVR': {
                'model': svm.SVR,
                'params': {
                    'kernel': [('poly', 2), ('poly', 3), ('poly', 4), ('poly', 5), 'linear', 'rbf', 'sigmoid'],
                    'C': [0.1, 0.5, 1.0, 5, 10, 50, 100],
                    'epsilon': [0.01, 0.05, 0.1, 0.5, 1.0, 2],
                    'cache_size': [1000],
                    'max_iter': [100000],
                    'norm': norm_all,
                    'feat': feat_all,
                },
            },
            'ANN': {
                'model': neural_network.MLPRegressor,
                'params': {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'solver': ['adam'],
                    'early_stopping': [True],
                    'learning_rate': ['adaptive'],
                    'n_iter_no_change': [10],
                    'norm': norm_all,
                    'feat': feat_all,
                },
            },
            'CART': {
                'model': tree.DecisionTreeRegressor,
                'params': {
                    'max_depth': max_depth_values,
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 5, 10, 20],
                    'norm': norm_all,
                    'feat': feat_all,
                },
            },
            'GBM': {
                'model': None,
                'params': {
                    'num_boost_round': n_estimators_values,
                    'num_leaves': [7, 15, 31, 63],
                    'bagging_fraction': [0.5, 0.75, 0.9],
                    'bagging_freq': [1],
                    'colsample_bytree': [0.5, 0.75, 0.9],
                    'min_data_in_leaf': [5, 10, 20],
                    'feature_pre_filter': [False],
                    'verbose': [-1],
                    'norm': norm_all,
                    'feat': feat_all,
                },
            },
            'RF': {
                'model': None,
                'params': {
                    'boosting_type': ['rf'],
                    'num_boost_round': n_estimators_values,
                    'num_leaves': [7, 15, 31, 63],
                    'bagging_fraction': [0.5, 0.75, 0.9],
                    'bagging_freq': [1],
                    'colsample_bytree': [0.5, 0.75, 0.9],
                    'min_data_in_leaf': [5, 10, 20],
                    'feature_pre_filter': [False],
                    'verbose': [-1],
                    'norm': norm_all,
                    'feat': feat_all,
                },
            },
        }
