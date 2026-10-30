# =============================================================================
# error_measures.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : Accuracy measures used throughout the study: MAE, MdAE and LSD,
#            plus the per-instance basis errors (absolute error and absolute
#            log residual) that the effect-size tests are computed on.
# =============================================================================

import numpy as np


class ErrorMeasures:

    @staticmethod
    def mae(actual, predicted):
        return np.abs(actual - predicted).mean()

    @staticmethod
    def mdae(actual, predicted):
        return np.median(np.abs(actual - predicted))

    @staticmethod
    def log_residuals(actual, predicted):
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        if np.any(actual < 0) or np.any(predicted < 0):
            raise ValueError('LSD is undefined when actual or predicted effort is not strictly positive')
        return np.log1p(actual) - np.log1p(predicted)

    @staticmethod
    def lsd(actual, predicted):
        actual = np.asarray(actual, dtype=float)
        residual = ErrorMeasures.log_residuals(actual, predicted)
        return float(np.sqrt(np.sum((residual - np.mean(residual)) ** 2) / (len(actual) - 1)))

    @staticmethod
    def basis_errors(actual, pred, key):
        actual = np.asarray(actual, float)
        pred = np.maximum(np.asarray(pred, float), 0.01)
        if key == 'ae':
            return np.abs(actual - pred)
        if key == 'abslog':
            return np.abs(ErrorMeasures.log_residuals(actual, pred))
        raise ValueError(f'Unknown basis error key: {key}')
