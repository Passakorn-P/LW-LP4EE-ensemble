# =============================================================================
# experiments/lw_experiments.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : RQ1-RQ3 experiments computed entirely from the prediction cache.
# =============================================================================

import os
import numpy as np
import pandas as pd
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

from configs import Configurations
from models.cached_lw_lp4ee import CachedLW_LP4EE
from error_measures import ErrorMeasures


class Exp1Fixed:

    @staticmethod
    def run_dataset(dataset, kernel, bandwidth):
        y_pred, actual = CachedLW_LP4EE.fixed_predictions(dataset, kernel, bandwidth)
        y_pred = np.where(y_pred < 0, 0.01, y_pred)
        return {
            'dataset': dataset,
            'MAE': ErrorMeasures.mae(actual, y_pred),
            'MdAE': ErrorMeasures.mdae(actual, y_pred),
            'LSD': ErrorMeasures.lsd(actual, y_pred),
        }

    @staticmethod
    def run():
        frames = []
        for kernel in Configurations.KERNELS:
            for bandwidth in Configurations.BANDWIDTHS:
                results_df = pd.DataFrame([
                    Exp1Fixed.run_dataset(dataset, kernel, bandwidth)
                    for dataset in CachedLW_LP4EE.list_cached_datasets()
                ])
                results_df.insert(0, 'kernel', kernel)
                results_df.insert(0, 'bandwidth', bandwidth)
                frames.append(results_df)

        combined_df = pd.concat(frames, ignore_index=True)
        os.makedirs(Configurations.EXP_PATH, exist_ok=True)
        combined_df.to_parquet(Configurations.EXP1_FILE, index=False)
        return combined_df


class Exp2LOOCV:

    @staticmethod
    def run_dataset(dataset, kernel, bw_candidates):
        y_pred, actual = CachedLW_LP4EE.cv_predictions(dataset, kernel, bw_candidates)
        y_pred = np.where(y_pred < 0, 0.01, y_pred)
        return {
            'dataset': dataset,
            'MAE': ErrorMeasures.mae(actual, y_pred),
            'MdAE': ErrorMeasures.mdae(actual, y_pred),
            'LSD': ErrorMeasures.lsd(actual, y_pred),
        }

    @staticmethod
    def process_subset(bw_candidates):
        subset_label = '-'.join(str(b) for b in bw_candidates)
        frames = []
        for kernel in Configurations.KERNELS:
            results_df = pd.DataFrame([
                Exp2LOOCV.run_dataset(dataset, kernel, bw_candidates)
                for dataset in CachedLW_LP4EE.list_cached_datasets()
            ])
            results_df.insert(0, 'kernel', kernel)
            results_df.insert(1, 'subset', subset_label)
            frames.append(results_df)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def run():
        all_subsets = [subset for r in range(2, len(Configurations.BANDWIDTHS) + 1)
                       for subset in combinations(Configurations.BANDWIDTHS, r)]

        all_results = []
        with ProcessPoolExecutor(max_workers=Configurations.N_WORKERS) as executor:
            futures = {executor.submit(Exp2LOOCV.process_subset, subset): subset for subset in all_subsets}
            for done, future in enumerate(as_completed(futures), start=1):
                all_results.append(future.result())
                print(f'completed {done} of {len(all_subsets)} (subset {futures[future]})', flush=True)

        final_df = pd.concat(all_results, ignore_index=True)
        os.makedirs(Configurations.EXP_PATH, exist_ok=True)
        final_df.to_parquet(Configurations.EXP2_FILE, index=False)
        return final_df


class Exp3Ensemble:

    @staticmethod
    def kernel_subsets():
        return [subset for r in range(1, len(Configurations.KERNELS) + 1)
                for subset in combinations(Configurations.KERNELS, r)]

    @staticmethod
    def bandwidth_subsets():
        return [subset for r in range(1, len(Configurations.BANDWIDTHS) + 1)
                for subset in combinations(Configurations.BANDWIDTHS, r)]

    @staticmethod
    def process_subset(k_subset):
        k_pos = {kernel: pos for pos, kernel in enumerate(Configurations.KERNELS)}
        b_pos = {bandwidth: pos for pos, bandwidth in enumerate(Configurations.BANDWIDTHS)}
        k_idx = [k_pos[kernel] for kernel in k_subset]
        k_label = '-'.join(k_subset)
        b_subsets = Exp3Ensemble.bandwidth_subsets()

        rows = []
        for dataset in CachedLW_LP4EE.list_cached_datasets():
            preds, actual = CachedLW_LP4EE.load_pivot(dataset)
            sub_k = preds[:, k_idx, :]

            for b_subset in b_subsets:
                b_idx = [b_pos[bandwidth] for bandwidth in b_subset]
                b_label = '-'.join(str(int(round(bandwidth * 10))) for bandwidth in b_subset)

                y_pred = sub_k[:, :, b_idx].mean(axis=(1, 2))
                y_pred = np.where(y_pred < 0, 0.01, y_pred)

                rows.append({
                    'kernel_subset': k_label,
                    'bandwidth_subset': b_label,
                    'dataset': dataset,
                    'MAE': ErrorMeasures.mae(actual, y_pred),
                    'MdAE': ErrorMeasures.mdae(actual, y_pred),
                    'LSD': ErrorMeasures.lsd(actual, y_pred),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def run():
        k_subsets = Exp3Ensemble.kernel_subsets()

        all_results = []
        with ProcessPoolExecutor(max_workers=Configurations.N_WORKERS) as executor:
            futures = {executor.submit(Exp3Ensemble.process_subset, subset): subset for subset in k_subsets}
            for done, future in enumerate(as_completed(futures), start=1):
                all_results.append(future.result())
                if done % 25 == 0 or done == len(k_subsets):
                    print(f'completed {done} of {len(k_subsets)} kernel subsets', flush=True)

        final_df = pd.concat(all_results, ignore_index=True)
        os.makedirs(Configurations.EXP_PATH, exist_ok=True)
        final_df.to_parquet(Configurations.EXP3_FILE, index=False)
        return final_df
