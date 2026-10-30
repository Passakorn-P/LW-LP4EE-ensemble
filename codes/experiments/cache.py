# =============================================================================
# experiments/cache.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : Builds the LW-LP4EE prediction cache that every downstream
#            analysis (RQ1-RQ3 and the adopted ensemble of RQ4) reads from.
#            For each dataset two parquet files are written:
#              <dataset>.parquet         outer leave-one-out predictions
#              <dataset>_nested.parquet  inner leave-one-out predictions for
#                                        every outer training fold (used for
#                                        cross-validated bandwidth selection)
# =============================================================================

import os
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from pyarrow import Table as ArrowTable
import numpy as np

from configs import Configurations
from data_handler import DataHandler
from models.lw_lp4ee import LW_LP4EE


class CacheBuilder:

    @staticmethod
    def predict_grid(train, test):
        train_x, train_y, test_x = DataHandler.minmax_fold(train, test)
        for kernel in Configurations.KERNELS:
            for bandwidth in Configurations.BANDWIDTHS:
                try:
                    pred = float(LW_LP4EE.predict(train_x, train_y, test_x, kernel=kernel, bandwidth=bandwidth)[0])
                except Exception:
                    pred = np.nan
                yield kernel, bandwidth, pred

    @staticmethod
    def build_outer_fold(task):
        dataset_name, dataset, outer_index = task
        outer_test = dataset.loc[[outer_index]]
        outer_train = dataset.drop(index=outer_index)

        actual = float(outer_test['Effort'].iloc[0])
        outer_rows = pd.DataFrame(
            ((dataset_name, outer_index, actual, *row) for row in CacheBuilder.predict_grid(outer_train, outer_test)),
            columns=Configurations.CACHE_OUTER_COLUMNS,
        )

        nested_rows = []
        for inner_index in outer_train.index:
            inner_test = outer_train.loc[[inner_index]]
            inner_train = outer_train.drop(index=inner_index)
            inner_actual = float(inner_test['Effort'].iloc[0])
            nested_rows.extend(
                (dataset_name, outer_index, int(inner_index), inner_actual, *row)
                for row in CacheBuilder.predict_grid(inner_train, inner_test)
            )
        return outer_rows, pd.DataFrame(nested_rows, columns=Configurations.CACHE_NESTED_COLUMNS)

    @staticmethod
    def write_chunk(writer, frame, path):
        table = ArrowTable.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
        return writer

    @staticmethod
    def generate_dataset_cache(dataset_name, dataset, workers=Configurations.N_WORKERS):
        dataset = dataset.reset_index(drop=True)
        os.makedirs(Configurations.CACHE_PATH, exist_ok=True)
        outer_path = f'{Configurations.CACHE_PATH}{dataset_name}.parquet'
        nested_path = f'{Configurations.CACHE_PATH}{dataset_name}_nested.parquet'
        outer_temp, nested_temp = outer_path + '.tmp', nested_path + '.tmp'

        tasks = [(dataset_name, dataset, int(index)) for index in dataset.index]
        outer_writer = nested_writer = None
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for completed, (outer_frame, nested_frame) in enumerate(
                        executor.map(CacheBuilder.build_outer_fold, tasks, chunksize=1), start=1):
                    outer_writer = CacheBuilder.write_chunk(outer_writer, outer_frame, outer_temp)
                    nested_writer = CacheBuilder.write_chunk(nested_writer, nested_frame, nested_temp)
                    print(f'{dataset_name}: completed outer fold {completed}/{len(dataset)}', flush=True)
            outer_writer.close()
            nested_writer.close()
            os.replace(outer_temp, outer_path)
            os.replace(nested_temp, nested_path)
        except BaseException:
            for writer in (outer_writer, nested_writer):
                if writer is not None:
                    writer.close()
            for path in (outer_temp, nested_temp):
                if os.path.exists(path):
                    os.remove(path)
            raise

    @staticmethod
    def build_all(dataset_names=None):
        datasets = DataHandler.read_all_datasets(dataset_names)
        for dataset_name, dataset in datasets.items():
            outer_path = f'{Configurations.CACHE_PATH}{dataset_name}.parquet'
            nested_path = f'{Configurations.CACHE_PATH}{dataset_name}_nested.parquet'
            if os.path.exists(outer_path) and os.path.exists(nested_path):
                print(f'{dataset_name}: cache exists, skipping', flush=True)
                continue
            CacheBuilder.generate_dataset_cache(dataset_name, dataset)
