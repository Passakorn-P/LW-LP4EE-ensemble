# =============================================================================
# models/lw_lp4ee.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : Locally Weighted LP4EE (LW-LP4EE).  For each query point the
#            kernel weights are derived from the distance to the k-th nearest
#            training case (k = round(n * bandwidth)), and the weighted
#            absolute-error LP is solved with lpSolve (lpsolve55), the same
#            engine used by R's linprog::solveLP(lpSolve = TRUE).
# =============================================================================

import warnings
import numpy as np
from lpsolve55 import lpsolve, IMPORTANT

from configs import Configurations

warnings.filterwarnings('ignore')


class LW_LP4EE:

    @staticmethod
    def kernel_weights(kernel, distances, bandwidth_radius):
        h = distances / max(bandwidth_radius, 1e-9)
        abs_h = np.abs(h)

        if kernel == 'rectangular':
            w = np.where(abs_h < 1.0, 0.5, 0.0)
        elif kernel == 'epanechnikov':
            w = np.where(abs_h < 1.0, 0.75 * (1.0 - h ** 2), 0.0)
        elif kernel == 'tricube':
            w = np.where(abs_h < 1.0, (70.0 / 81.0) * (1.0 - abs_h ** 3) ** 3, 0.0)
        elif kernel == 'gaussian':
            w = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * h ** 2)
        elif kernel == 'triangle':
            w = np.where(abs_h < 1.0, 1.0 - abs_h, 0.0)
        elif kernel == 'triweight':
            w = np.where(abs_h < 1.0, (35.0 / 32.0) * (1.0 - h ** 2) ** 3, 0.0)
        elif kernel == 'biweight':
            w = np.where(abs_h < 1.0, (15.0 / 16.0) * (1.0 - h ** 2) ** 2, 0.0)
        elif kernel == 'cosine':
            w = np.where(abs_h < 1.0, (np.pi / 4.0) * np.cos((np.pi / 2.0) * h), 0.0)
        elif kernel == 'logistic':
            w = 1.0 / (np.exp(h) + 2.0 + np.exp(-h))
        elif kernel == 'sigmoid':
            w = (2.0 / np.pi) * (1.0 / (np.exp(h) + np.exp(-h)))
        else:
            raise ValueError(f"Unknown kernel: '{kernel}'. Choose from {Configurations.KERNELS}")

        return np.asarray(w, dtype=float)

    @staticmethod
    def support_mask(kernel, distances, bandwidth_radius):
        if kernel in Configurations.COMPACT_KERNELS:
            return np.abs(distances / max(bandwidth_radius, 1e-9)) < 1.0
        return np.ones(len(distances), dtype=bool)

    @staticmethod
    def solve_lp(train_X, train_y, target_x, weights, active_mask, upper_bound=1e16):
        n_samples, n_features = train_X.shape
        active_X = train_X[active_mask]
        active_y = train_y[active_mask]
        active_weights = weights[active_mask]
        n_active = len(active_y)

        if n_active == 0:
            raise RuntimeError('Kernel support contains no training observations')

        weight_sum = float(np.sum(active_weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            raise RuntimeError('Active kernel weights have no positive mass')

        n_variables = n_features + 2 * n_active
        t_indices = n_features + np.arange(n_active) * 2
        e_indices = t_indices + 1
        rows = np.arange(n_active)
        cols = np.arange(n_features)

        c = np.zeros(n_variables, dtype=np.float64)
        c[t_indices] = active_weights / weight_sum

        effective_ub = min(float(upper_bound), 1e7)

        mat1 = np.zeros((n_samples, n_variables), dtype=np.float64)
        mat1[:, :n_features] = train_X

        mat2 = np.zeros((n_active, n_variables), dtype=np.float64)
        mat2[:, :n_features] = active_X
        mat2[rows, t_indices] = -1.0
        mat2[rows, e_indices] = -1.0

        mat3 = np.zeros((n_active, n_variables), dtype=np.float64)
        mat3[:, :n_features] = active_X
        mat3[rows, t_indices] = 1.0
        mat3[rows, e_indices] = -1.0

        mat4 = np.zeros((n_features, n_variables), dtype=np.float64)
        mat4[cols, cols] = 1.0

        mat5 = np.zeros((n_features, n_variables), dtype=np.float64)
        mat5[cols, cols] = 1.0

        mat6 = np.zeros((n_active, n_variables), dtype=np.float64)
        mat6[rows, e_indices] = 1.0

        A = np.vstack([mat1, mat2, mat3, mat4, mat5, mat6])
        B = np.concatenate([
            np.zeros(n_samples), np.zeros(n_active), np.zeros(n_active),
            np.zeros(n_features), np.full(n_features, effective_ub), active_y.astype(np.float64),
        ])
        directions = [2] * n_samples + [1] * n_active + [2] * n_active + \
                     [2] * n_features + [1] * n_features + [3] * n_active

        lp = lpsolve('make_lp', 0, n_variables)
        try:
            lpsolve('set_verbose', lp, IMPORTANT)
            lpsolve('set_obj_fn', lp, c.tolist())
            lpsolve('set_minim', lp)
            for j in range(n_variables):
                lpsolve('set_lowbo', lp, j + 1, 0.0)

            lpsolve('set_add_rowmode', lp, True)
            for i in range(A.shape[0]):
                lpsolve('add_constraint', lp, A[i].tolist(), directions[i], float(B[i]))
            lpsolve('set_add_rowmode', lp, False)

            lpsolve('set_scaling', lp, 196)
            status = lpsolve('solve', lp)
            if status not in (0, 1):
                raise RuntimeError(f'lpSolve returned status {status}')
            solution = np.asarray(lpsolve('get_variables', lp)[0], dtype=np.float64).ravel()
        except Exception as exc:
            raise RuntimeError(f'lpSolve execution error: {exc}') from exc
        finally:
            lpsolve('delete_lp', lp)

        return solution[:n_features], float(np.dot(target_x, solution[:n_features]))

    @staticmethod
    def predict(train_X, train_y, test_X, kernel='gaussian', bandwidth=0.2, upper_bound=1e16):
        if kernel not in Configurations.KERNELS:
            raise ValueError(f"Unknown kernel '{kernel}'. Choose from {Configurations.KERNELS}")
        if not np.isfinite(bandwidth) or not 0 < bandwidth <= 1:
            raise ValueError('bandwidth must be finite and in (0, 1]')

        if test_X.ndim == 1:
            test_X = test_X.reshape(1, -1)

        n_train = train_X.shape[0]
        predictions = np.zeros(test_X.shape[0])
        for j in range(test_X.shape[0]):
            target_x = test_X[j]
            distances = np.linalg.norm(train_X - target_x, axis=1)
            k = min(max(1, int(np.round(n_train * bandwidth))), n_train)
            bandwidth_radius = max(float(np.sort(distances)[k - 1]), 1e-9)
            weights = LW_LP4EE.kernel_weights(kernel, distances, bandwidth_radius)
            active_mask = LW_LP4EE.support_mask(kernel, distances, bandwidth_radius)
            _, predictions[j] = LW_LP4EE.solve_lp(train_X, train_y, target_x, weights, active_mask, upper_bound)
        return predictions
