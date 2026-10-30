# =============================================================================
# models/cached_lw_lp4ee.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : Cache-backed LW-LP4EE.  Serves fixed-bandwidth predictions from
#            the outer leave-one-out cache, and cross-validated bandwidth
#            selection from the nested cache: for every outer test index the
#            bandwidth with the lowest inner-LOO mean squared error over the
#            candidate set is chosen and the corresponding outer prediction
#            is returned.
# =============================================================================

import os
import numpy as np
import pandas as pd

from configs import Configurations


class CachedLW_LP4EE:

    outer_cache_by_dataset = {}
    nested_cache_by_dataset = {}
    selection_by_key = {}
    pivot_by_dataset = {}

    @staticmethod
    def list_cached_datasets():
        names = []
        for file_name in sorted(os.listdir(Configurations.CACHE_PATH)):
            if file_name.endswith('.parquet') and not file_name.endswith('_nested.parquet'):
                names.append(file_name[:-len('.parquet')])
        return names

    @staticmethod
    def load_outer_cache(dataset):
        if dataset not in CachedLW_LP4EE.outer_cache_by_dataset:
            CachedLW_LP4EE.outer_cache_by_dataset[dataset] = pd.read_parquet(f'{Configurations.CACHE_PATH}{dataset}.parquet')
        return CachedLW_LP4EE.outer_cache_by_dataset[dataset]

    @staticmethod
    def load_nested_cache(dataset):
        if dataset not in CachedLW_LP4EE.nested_cache_by_dataset:
            CachedLW_LP4EE.nested_cache_by_dataset[dataset] = pd.read_parquet(f'{Configurations.CACHE_PATH}{dataset}_nested.parquet')
        return CachedLW_LP4EE.nested_cache_by_dataset[dataset]

    @staticmethod
    def load_pivot(dataset):
        if dataset in CachedLW_LP4EE.pivot_by_dataset:
            return CachedLW_LP4EE.pivot_by_dataset[dataset]

        df = CachedLW_LP4EE.load_outer_cache(dataset)
        indices = sorted(df['index'].unique())
        pivot = df.pivot_table(index='index', columns=['kernel', 'bandwidth'], values='pred', dropna=False)
        columns = pd.MultiIndex.from_product([Configurations.KERNELS, Configurations.BANDWIDTHS])
        pivot = pivot.reindex(index=indices, columns=columns)

        preds = pivot.to_numpy().reshape(len(indices), len(Configurations.KERNELS), len(Configurations.BANDWIDTHS))
        actual = df.drop_duplicates('index').set_index('index')['actual'].reindex(indices).to_numpy()

        CachedLW_LP4EE.pivot_by_dataset[dataset] = (preds, actual)
        return preds, actual

    @staticmethod
    def fixed_predictions(dataset, kernel, bandwidth):
        preds, actual = CachedLW_LP4EE.load_pivot(dataset)
        k_pos = Configurations.KERNELS.index(kernel)
        b_pos = Configurations.BANDWIDTHS.index(bandwidth)
        return preds[:, k_pos, b_pos], actual

    @staticmethod
    def selection(dataset, kernel, bw_candidates):
        key = (dataset, kernel, tuple(sorted(bw_candidates)))
        if key in CachedLW_LP4EE.selection_by_key:
            return CachedLW_LP4EE.selection_by_key[key]

        nested = CachedLW_LP4EE.load_nested_cache(dataset)
        nested = nested[(nested['kernel'] == kernel) & (nested['bandwidth'].isin(bw_candidates))].copy()
        nested['sq_err'] = ((nested['pred'] - nested['actual']) ** 2).fillna(np.inf)

        inner_mse = nested.groupby(['outer_index', 'bandwidth'])['sq_err'].mean().rename('inner_mse').reset_index()
        best_bw = inner_mse.loc[inner_mse.groupby('outer_index')['inner_mse'].idxmin()]
        best_bw = best_bw.set_index('outer_index')['bandwidth']

        CachedLW_LP4EE.selection_by_key[key] = best_bw
        return best_bw

    @staticmethod
    def cv_predictions(dataset, kernel, bw_candidates):
        preds, actual = CachedLW_LP4EE.load_pivot(dataset)
        indices = sorted(CachedLW_LP4EE.load_outer_cache(dataset)['index'].unique())
        best_bw = CachedLW_LP4EE.selection(dataset, kernel, bw_candidates)
        k_pos = Configurations.KERNELS.index(kernel)

        y_pred = np.zeros(len(indices))
        for pos, index in enumerate(indices):
            b_pos = Configurations.BANDWIDTHS.index(round(float(best_bw.loc[index]), 1))
            y_pred[pos] = preds[pos, k_pos, b_pos]
        return y_pred, actual
