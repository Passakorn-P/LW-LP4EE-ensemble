# =============================================================================
# data_handler.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : Loads the SEACRAFT-style XML/conf effort datasets, renames the
#            target to 'Effort' and the size attribute to 'Size', drops
#            ignored and constant columns, one-hot encodes categorical
#            columns, and builds the nine preprocessing variants
#            (nothing/minmax/ln x all/rfe/pca) used by the baselines.
# =============================================================================

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler

from configs import Configurations


class DataHandler:

    @staticmethod
    def parse_xml_to_df(xml_file):
        root = ET.parse(xml_file).getroot()
        data = []
        for case in root.find('Data').findall('Case'):
            data.append([item.get('value') for item in case.findall('DataItem')])
        columns = [field.get('name') for field in root.find('Template').findall('Field')]
        return pd.DataFrame(data, columns=columns)

    @staticmethod
    def read_dataset(dataset_name):
        datasets_path = Configurations.DATASETS_PATH + dataset_name
        config = pd.read_csv(datasets_path + '.conf', sep='=', header=None).set_index(0)[1]
        df = DataHandler.parse_xml_to_df(datasets_path + '.xml')

        df = df.rename(columns={df.columns[int(config['target_index'])]: 'Effort'})
        corr_idx = list(map(int, config['correlation_index'].split(',')))
        df = df.rename(columns={df.columns[corr_idx[0]]: 'Size'})
        if not pd.isna(config['ignore_index']):
            df = df.drop(columns=df.columns[list(map(int, config['ignore_index'].split(',')))])
        df = df.drop(columns=df.columns[df.nunique() == 1])

        for column in df.columns:
            try:
                df[column] = df[column].astype(float)
            except ValueError:
                df = pd.concat([df.drop(columns=column), pd.get_dummies(df[column], prefix=column) * 1], axis=1)
        return df

    @staticmethod
    def read_all_datasets(dataset_names=None):
        dataset_names = Configurations.DATASET_NAMES if dataset_names is None else dataset_names
        return {dataset_name: DataHandler.read_dataset(dataset_name) for dataset_name in dataset_names}

    @staticmethod
    def get_preprocessed_variants(df_train, df_single_test):
        X_train, y_train = df_train.drop(columns=['Effort']), df_train['Effort']
        X_test, y_test = df_single_test.drop(columns=['Effort']), df_single_test['Effort']

        all_ds = {}
        for norm_name in Configurations.NORMALIZATIONS:
            if norm_name == 'nothing':
                X_train_norm, X_test_norm = X_train.values, X_test.values
            elif norm_name == 'minmax':
                scaler = MinMaxScaler()
                X_train_norm = scaler.fit_transform(X_train)
                X_test_norm = scaler.transform(X_test)
            else:
                X_train_norm, X_test_norm = np.log1p(X_train).values, np.log1p(X_test).values

            for feat_name in Configurations.FEATURE_SELECTIONS:
                if feat_name == 'all':
                    X_train_sel, X_test_sel = X_train_norm, X_test_norm
                elif feat_name == 'rfe':
                    rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42))
                    X_train_sel = rfe.fit_transform(X_train_norm, y_train)
                    X_test_sel = rfe.transform(X_test_norm)
                else:
                    pca = PCA(n_components=0.999)
                    X_train_sel = pca.fit_transform(X_train_norm)
                    X_test_sel = pca.transform(X_test_norm)

                all_ds[f'{norm_name}__{feat_name}'] = {
                    'X_train': X_train_sel, 'size_train': X_train['Size'].values, 'y_train': y_train.values,
                    'X_test': X_test_sel, 'size_test': X_test['Size'].values, 'y_test': y_test.values,
                }
        return all_ds

    @staticmethod
    def minmax_fold(df_train, df_test):
        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(df_train.drop(columns='Effort'))
        test_x = scaler.transform(df_test.drop(columns='Effort'))
        train_y = df_train['Effort'].to_numpy(dtype=float)
        return train_x, train_y, test_x
