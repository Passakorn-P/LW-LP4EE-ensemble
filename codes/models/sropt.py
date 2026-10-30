# =============================================================================
# models/sropt.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : SROpt baseline (robust multi-scenario LP4EE).  The training
#            features are perturbed by +/- level * deviation for
#            level = 1..n_levels, all scenarios are stacked, and an
#            LP is solved with lpSolve (lpsolve55).
# =============================================================================

import numpy as np
from lpsolve55 import lpsolve, IMPORTANT


class SROpt:

    @staticmethod
    def predict(train_X, train_y, test_X, deviation=0.01, n_levels=1, upper_bound=1e16):
        n_samples, n_features = train_X.shape

        scenarios = [train_X]
        for level in range(1, n_levels + 1):
            dev_factor = level * deviation
            scenarios.append(train_X * (1 - dev_factor))
            scenarios.append(train_X * (1 + dev_factor))

        combined_X = np.vstack(scenarios)
        combined_y = np.tile(train_y, len(scenarios))
        total_samples = n_samples * len(scenarios)
        n_variables = n_features + 2 * total_samples

        c = np.zeros(n_variables)
        c[n_features + 2 * np.arange(total_samples)] = 1
        c[n_features + 2 * np.arange(total_samples) + 1] = 1

        A_eq = np.zeros((total_samples, n_variables))
        A_eq[:, :n_features] = combined_X
        A_eq[np.arange(total_samples), n_features + 2 * np.arange(total_samples)] = -1
        A_eq[np.arange(total_samples), n_features + 2 * np.arange(total_samples) + 1] = 1

        effective_ub = min(float(upper_bound), 1e7)

        lp = lpsolve('make_lp', 0, n_variables)
        try:
            lpsolve('set_verbose', lp, IMPORTANT)
            lpsolve('set_obj_fn', lp, c.tolist())
            lpsolve('set_minim', lp)

            for j in range(n_variables):
                lpsolve('set_lowbo', lp, j + 1, 0.0)
            for j in range(n_features):
                lpsolve('set_upbo', lp, j + 1, effective_ub)

            lpsolve('set_add_rowmode', lp, True)
            for i in range(A_eq.shape[0]):
                lpsolve('add_constraint', lp, A_eq[i].tolist(), 3, float(combined_y[i]))
            lpsolve('set_add_rowmode', lp, False)

            lpsolve('set_scaling', lp, 196)
            lpsolve('solve', lp)
            solution = np.asarray(lpsolve('get_variables', lp)[0], dtype=np.float64).ravel()
        except Exception as exc:
            raise RuntimeError(f'Optimization failed: lpSolve execution error: {exc}')
        finally:
            lpsolve('delete_lp', lp)

        coefs = solution[:n_features]
        return np.dot(test_X, coefs), coefs
