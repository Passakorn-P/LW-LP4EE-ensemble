# =============================================================================
# experiments/baselines.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : Runs the eight baselines compared in RQ4 with outer leave-one-out
#            evaluation: LP4EE, LP4EE_SROpt, ABE, ANN, CART, GBM, RF, SVR.
#            For every outer fold the nine preprocessing variants
#            (nothing/minmax/ln x all/rfe/pca) are built on the training
#            part only; tunable baselines are optimised by exhaustive grid
#            search with inner leave-one-out MSE.  LP4EE and LP4EE_SROpt
#            are fixed to minmax / all features.
# =============================================================================

import multiprocessing
import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from configs import Configurations
from data_handler import DataHandler
from error_measures import ErrorMeasures
from models.lp4ee import LP4EE
from models.sropt import SROpt

warnings.simplefilter('ignore', category=ConvergenceWarning)


class Baselines:

    @staticmethod
    def run(selected_models=None, is_reset=False, dataset_names=None):
        selected_models = Configurations.BASELINE_MODEL_NAMES if selected_models is None else selected_models
        datasets = DataHandler.read_all_datasets(dataset_names)

        for dataset, ds in datasets.items():
            print(dataset, flush=True)
            actual = pd.DataFrame({'actual': ds['Effort'].values})
            all_model_configs = Configurations.get_baseline_params(ds.shape[0])

            for model_name in selected_models:
                model_config = all_model_configs[model_name]
                cpu_cnt = Configurations.N_WORKERS
                param_names = list(model_config['params'])

                output_dir = f'{Configurations.BASELINE_PATH}{model_name}/'
                os.makedirs(output_dir, exist_ok=True)
                output_file = f'{output_dir}{dataset}.parquet'

                if model_name == 'SVR':
                    param_names = param_names + ['degree']

                y_preds = [np.nan] * len(ds)
                y_params = [{} for _ in range(len(ds))]
                completed_rows = [False] * len(ds)

                if not is_reset and os.path.exists(output_file):
                    cached_df = pd.read_parquet(output_file)
                    pred_col = f'pred__{model_name}'
                    param_cols = [f'params__{p}' for p in param_names]
                    # SVR 'degree' is legitimately NaN for non-poly kernels
                    required_cols = [c for c in param_cols if c != 'params__degree']
                    if pred_col in cached_df.columns and all(column in cached_df.columns for column in param_cols):
                        for idx in range(min(len(cached_df), len(ds))):
                            row = cached_df.iloc[idx]
                            if not pd.isna(row[pred_col]) and not row[required_cols].isnull().any():
                                y_preds[idx] = row[pred_col]
                                y_params[idx] = {name: row[f'params__{name}'] for name in param_names}
                                completed_rows[idx] = True
                    print(f'{model_name}: resuming with {sum(completed_rows)} completed rows', flush=True)

                def save_progress():
                    y_pred_df = pd.DataFrame({f'pred__{model_name}': y_preds})
                    y_pred_df[y_pred_df < 0] = 0
                    filled_params = [p if p else {k: np.nan for k in param_names} for p in y_params]
                    y_params_df = pd.DataFrame(filled_params)
                    y_params_df = y_params_df.rename(columns={c: f'params__{c}' for c in param_names})
                    pd.concat([actual, y_pred_df, y_params_df], axis=1).to_parquet(output_file)

                if model_name == 'LP4EE':
                    selected_params = {name: values[0] for name, values in model_config['params'].items()}
                    tasks = [
                        (i, ds.drop(i), ds.iloc[[i]], model_name, model_config, selected_params)
                        for i in range(len(ds)) if not completed_rows[i]
                    ]
                    if tasks:
                        with ProcessPoolExecutor(max_workers=min(max(1, cpu_cnt), len(tasks)),
                                                 mp_context=multiprocessing.get_context('spawn')) as executor:
                            for i, pred in executor.map(Baselines.evaluate_lp4ee_loo_row, tasks, chunksize=1):
                                y_preds[i] = pred
                                y_params[i] = selected_params
                                completed_rows[i] = True
                                save_progress()
                else:
                    for i in range(len(ds)):
                        if completed_rows[i]:
                            continue
                        print(f'{model_name}: fold {i}', flush=True)

                        all_processed_ds = DataHandler.get_preprocessed_variants(ds.drop(i), ds.iloc[[i]])

                        if model_name == 'ABE':
                            selected_params = Baselines.optimize_abe_grid(model_config, all_processed_ds)
                        elif model_name == 'LP4EE_SROpt':
                            selected_params = Baselines.optimize_grid(
                                cpu_cnt, model_config, all_processed_ds, Baselines.evaluate_sropt_candidate)
                        elif model_name in ('GBM', 'RF'):
                            selected_params = Baselines.optimize_grid(
                                cpu_cnt, model_config, all_processed_ds, Baselines.evaluate_lgb_candidate)
                        else:
                            selected_params = Baselines.optimize_grid(
                                cpu_cnt, model_config, all_processed_ds, Baselines.evaluate_sklearn_candidate)
                            if model_name == 'SVR':
                                selected_params = Baselines.split_svr_kernel(selected_params)

                        y_preds[i] = Baselines.predict_model(model_name, model_config, all_processed_ds, selected_params)
                        y_params[i] = selected_params
                        completed_rows[i] = True
                        save_progress()

                save_progress()
                y_pred = np.maximum(np.asarray(y_preds, dtype=float), 0)
                print(f'{dataset} | {model_name} | MAE: {ErrorMeasures.mae(ds["Effort"].values, y_pred)} | '
                      f'MdAE: {ErrorMeasures.mdae(ds["Effort"].values, y_pred)}', flush=True)

    @staticmethod
    def evaluate_lp4ee_loo_row(task):
        i, df_train, df_single_test, model_name, model_config, params = task
        all_processed_ds = DataHandler.get_preprocessed_variants(df_train, df_single_test)
        return i, Baselines.predict_model(model_name, model_config, all_processed_ds, params)

    @staticmethod
    def select_processed_dataset(all_processed_ds, params):
        return all_processed_ds[f'{params["norm"]}__{params["feat"]}']

    @staticmethod
    def build_grid(model_config):
        param_names = list(model_config['params'])
        return [dict(zip(param_names, values)) for values in product(*(model_config['params'][name] for name in param_names))]

    @staticmethod
    def pick_best(scores, grid, model_name):
        valid_results = [(score, index) for index, score in enumerate(scores) if np.isfinite(score)]
        if not valid_results:
            raise RuntimeError(f'All grid-search candidates failed for {model_name}')
        _, best_index = min(valid_results)
        return grid[best_index]

    @staticmethod
    def optimize_grid(cpu_cnt, model_config, all_processed_ds, evaluate_candidate):
        grid = Baselines.build_grid(model_config)
        with ProcessPoolExecutor(max_workers=max(1, cpu_cnt), mp_context=multiprocessing.get_context('spawn')) as executor:
            scores = [
                evaluate_candidate(executor, params, model_config['model'],
                                   Baselines.select_processed_dataset(all_processed_ds, params))
                for params in grid
            ]
        return Baselines.pick_best(scores, grid, evaluate_candidate.__name__)

    @staticmethod
    def optimize_abe_grid(model_config, all_processed_ds):
        grid = Baselines.build_grid(model_config)
        loo_cache = {}
        scores = []
        for params in grid:
            key = (params['norm'], params['feat'], params['metric'])
            processed_ds = Baselines.select_processed_dataset(all_processed_ds, params)
            if key not in loo_cache:
                loo_cache[key] = Baselines.abe_loo_predictions_all_k(
                    processed_ds['X_train'], processed_ds['y_train'], params['metric'])
            loo_preds_by_k = loo_cache[key]
            k = params['k']
            if k > loo_preds_by_k.shape[1]:
                scores.append(np.inf)
                continue
            loo_results = loo_preds_by_k[:, k - 1].copy()
            scores.append(mean_squared_error(np.asarray(processed_ds['y_train'], dtype=float), loo_results))
        return Baselines.pick_best(scores, grid, 'ABE')

    @staticmethod
    def abe_loo_predictions_all_k(train_X, train_y, metric, max_k=None):
        X = np.asarray(train_X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(train_y, dtype=float).ravel()
        n = X.shape[0]
        if n < 2:
            return np.empty((n, 0))
        max_k = n - 1 if max_k is None else min(max_k, n - 1)

        dist = pairwise_distances(X, metric=metric)
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1, kind='stable')[:, :max_k]
        return np.cumsum(y[order], axis=1) / np.arange(1, max_k + 1)

    @staticmethod
    def evaluate_sropt_candidate(executor, params, model_cls, processed_ds):
        train_X, train_y = processed_ds['X_train'], processed_ds['y_train']
        fold_tasks = [
            (loo_i, train_X, train_y, params['deviation_fraction'], params['n_levels'])
            for loo_i in range(len(train_X))
        ]
        loo_results = np.asarray(list(executor.map(Baselines.evaluate_sropt_inner_fold, fold_tasks, chunksize=1)), dtype=float)
        return mean_squared_error(train_y, loo_results)

    @staticmethod
    def evaluate_sropt_inner_fold(task):
        loo_i, train_X, train_y, deviation_fraction, n_levels = task
        x_train = np.delete(train_X, loo_i, axis=0)
        y_train = np.delete(train_y, loo_i)
        x_test = np.atleast_2d(train_X[loo_i])
        pred, _ = SROpt.predict(x_train, y_train, x_test, deviation_fraction, n_levels)
        return pred.item()

    @staticmethod
    def split_svr_kernel(params):
        params = dict(params)
        kernel = params.get('kernel')
        if isinstance(kernel, tuple):
            kernel_name, degree = kernel
            params['kernel'] = kernel_name
            params['degree'] = degree
        else:
            params.setdefault('degree', np.nan)
        return params

    @staticmethod
    def build_svr_estimator_params(estimator_params):
        estimator_params = dict(estimator_params)
        if estimator_params.get('kernel') != 'poly':
            estimator_params.pop('degree', None)
        return estimator_params

    @staticmethod
    def evaluate_sklearn_candidate(executor, params, model_cls, processed_ds):
        train_X, train_y = processed_ds['X_train'], processed_ds['y_train']
        if len(train_X) < 2:
            return np.inf

        estimator_params = {name: value for name, value in params.items() if name not in ('norm', 'feat')}
        estimator_params = Baselines.build_svr_estimator_params(Baselines.split_svr_kernel(estimator_params))

        fold_tasks = [(loo_i, model_cls, estimator_params, train_X, train_y) for loo_i in range(len(train_X))]
        loo_results = np.asarray(list(executor.map(Baselines.evaluate_sklearn_inner_fold, fold_tasks, chunksize=1)), dtype=float)
        return mean_squared_error(train_y, loo_results)

    @staticmethod
    def evaluate_sklearn_inner_fold(task):
        loo_i, model_cls, estimator_params, train_X, train_y = task
        x_train = np.delete(train_X, loo_i, axis=0)
        y_train = np.delete(train_y, loo_i)
        x_test = np.atleast_2d(train_X[loo_i])
        model = model_cls(**estimator_params)
        with threadpool_limits(limits=1, user_api='blas'):
            model.fit(x_train, y_train)
        return float(model.predict(x_test)[0])

    @staticmethod
    def build_lgb_estimator_params(params):
        estimator_params = {name: value for name, value in params.items() if name not in ('norm', 'feat', 'num_boost_round')}
        estimator_params['num_threads'] = 1
        estimator_params.setdefault('verbose', -1)
        estimator_params.setdefault('feature_pre_filter', False)
        return estimator_params

    @staticmethod
    def evaluate_lgb_candidate(executor, params, model_cls, processed_ds):
        train_X, train_y = processed_ds['X_train'], processed_ds['y_train']
        if len(train_X) < 2:
            return np.inf

        lgb_params = Baselines.build_lgb_estimator_params(params)
        num_boost_round = int(params['num_boost_round'])
        fold_tasks = [(loo_i, lgb_params, num_boost_round, train_X, train_y) for loo_i in range(len(train_X))]
        try:
            loo_results = np.asarray(list(executor.map(Baselines.evaluate_lgb_inner_fold, fold_tasks, chunksize=1)), dtype=float)
        except Exception:
            return np.inf
        return mean_squared_error(train_y, loo_results)

    @staticmethod
    def evaluate_lgb_inner_fold(task):
        loo_i, lgb_params, num_boost_round, train_X, train_y = task
        x_train = np.ascontiguousarray(np.delete(train_X, loo_i, axis=0))
        y_train = np.ascontiguousarray(np.delete(train_y, loo_i))
        x_test = np.ascontiguousarray(np.atleast_2d(train_X[loo_i]))
        model = lgb.train(lgb_params, lgb.Dataset(x_train, label=y_train), num_boost_round=num_boost_round)
        return float(model.predict(x_test)[0])

    @staticmethod
    def predict_model(model_name, model_config, all_processed_ds, params):
        selected = Baselines.select_processed_dataset(all_processed_ds, params)
        train_X, train_y = selected['X_train'], selected['y_train']
        test_X = selected['X_test']
        estimator_params = {name: value for name, value in params.items() if name not in ('norm', 'feat')}

        if model_name == 'LP4EE_SROpt':
            pred, _ = SROpt.predict(train_X, train_y, test_X, estimator_params['deviation_fraction'], estimator_params['n_levels'])
            return pred[0].item()
        elif model_name == 'LP4EE':
            predictions, _ = LP4EE.lp_regression_r(
                np.asarray(train_X, dtype=float), np.asarray(train_y, dtype=float),
                np.asarray(test_X, dtype=float), np.asarray(selected['y_test'], dtype=float))
            return float(np.atleast_1d(predictions)[0])
        elif model_name == 'ABE':
            neighbors = NearestNeighbors(n_neighbors=estimator_params['k'], algorithm='brute', metric=estimator_params['metric'])
            neighbors.fit(train_X)
            indices = neighbors.kneighbors(test_X, return_distance=False)[0]
            return float(np.mean(train_y[indices]))
        elif model_name in ('GBM', 'RF'):
            train_X = np.ascontiguousarray(train_X)
            train_y = np.ascontiguousarray(train_y)
            test_X = np.ascontiguousarray(test_X)
            model = lgb.train(Baselines.build_lgb_estimator_params(params), lgb.Dataset(train_X, label=train_y),
                              num_boost_round=int(params['num_boost_round']))
            return model.predict(test_X)[0].item()
        elif model_name in ('ANN', 'CART', 'SVR'):
            if model_name == 'SVR':
                estimator_params = Baselines.build_svr_estimator_params(estimator_params)
            model = model_config['model'](**estimator_params)
            with threadpool_limits(limits=1, user_api='blas'):
                model.fit(train_X, train_y)
            return model.predict(test_X)[0].item()
        raise ValueError(f'Unsupported model: {model_name}')
