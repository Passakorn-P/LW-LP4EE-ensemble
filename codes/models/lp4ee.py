# =============================================================================
# models/lp4ee.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : LP4EE baseline evaluated through the original R implementation
#            (models/lp.R, linprog::solveLP with lpSolve = TRUE) via rpy2.
# =============================================================================

import numpy as np
import pandas as pd

from configs import Configurations


class LP4EE:

    @staticmethod
    def lp_regression_r(train_X, train_y, test_X, test_y):
        import rpy2.robjects as ro
        from rpy2.robjects import conversion, default_converter, pandas2ri

        df_train = pd.concat([pd.DataFrame(train_X), pd.DataFrame({'Effort': train_y})], axis=1)
        df_single_test = pd.concat([pd.DataFrame(test_X), pd.DataFrame({'Effort': test_y})], axis=1)
        df_train.columns = df_train.columns.map(str)
        df_single_test.columns = df_single_test.columns.map(str)

        ro.r('suppressPackageStartupMessages(library(linprog))')
        ro.r['source'](Configurations.LP_R_FILE)
        lp = ro.r['lp']
        r_columns = ro.StrVector([column for column in df_train if column != 'Effort'])

        with conversion.localconverter(default_converter + pandas2ri.converter):
            r_df_train = conversion.get_conversion().py2rpy(df_train)
            r_df_test = conversion.get_conversion().py2rpy(df_single_test)
            result = lp(r_df_train, r_df_test, r_columns)
            pred_df = conversion.get_conversion().rpy2py(result[0])
            coefficients = conversion.get_conversion().rpy2py(result[1])

        return pred_df['prediction'].values, coefficients
