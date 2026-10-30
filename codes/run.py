# =============================================================================
# run.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : Unified entry point for the entire experimental pipeline.
#            Provides four runnable stages via the --stage flag:
#              cache       build the LW-LP4EE prediction cache (outer and
#                          nested leave-one-out, 10 kernels x 10 bandwidths)
#                          in ../results/lw_cache/
#              rq1-rq3     run the cache-backed experiments and report
#                          RQ1-RQ3 (tables and figures) plus the kernel
#                          dissimilarity analysis
#              baselines   run the eight baselines with outer leave-one-out
#                          and write predictions to ../results/baseline_results/
#              rq4         compare the adopted ensemble against the eight
#                          baselines (WRS bprm/cid, Holm, win/tie/loss table
#                          and raw-error figure)
#            The stages must be run in the order listed.  The rq1-rq3 stage
#            needs the LP4EE reference; if it is missing, only LP4EE is computed.
# =============================================================================

import argparse
import os
import sys
import time

_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)
os.chdir(_cur_dir)

from configs import Configurations


def check_datasets(dataset_names):
    missing = [name for name in dataset_names
               if not (os.path.exists(f'{Configurations.DATASETS_PATH}{name}.xml')
                       and os.path.exists(f'{Configurations.DATASETS_PATH}{name}.conf'))]
    if missing:
        raise FileNotFoundError(
            f'Dataset files not found in {Configurations.DATASETS_PATH}: {missing}. '
            f'Restrict the run with --datasets.')


def run_cache(dataset_names):
    print('STAGE: cache')
    check_datasets(dataset_names)

    from experiments.cache import CacheBuilder

    t0 = time.time()
    CacheBuilder.build_all(dataset_names)
    print(f'\nCache built in {(time.time() - t0) / 3600:.2f} h')


def run_baselines(selected_models, dataset_names, is_reset=False):
    print(f'STAGE: baselines  {selected_models}')
    check_datasets(dataset_names)

    from experiments.baselines import Baselines

    t0 = time.time()
    Baselines.run(selected_models=selected_models, is_reset=is_reset, dataset_names=dataset_names)
    print(f'\nBaselines finished in {(time.time() - t0) / 3600:.2f} h')


def run_rq1_rq3(is_reset):
    print('STAGE: rq1-rq3')

    from models.cached_lw_lp4ee import CachedLW_LP4EE
    from experiments.lw_experiments import Exp1Fixed, Exp2LOOCV, Exp3Ensemble
    from evaluations.plots import Rq1Eval, Rq2Eval, Rq3Eval, Rq3_2Eval
    from evaluations.kernel_dissimilarity import KernelDissimilarity

    cached = CachedLW_LP4EE.list_cached_datasets() if os.path.isdir(Configurations.CACHE_PATH) else []
    if not cached:
        raise FileNotFoundError('No cache found. Run --stage cache first.')

    lp4ee_dir = f'{Configurations.BASELINE_PATH}LP4EE/'
    lp4ee_ready = all(os.path.exists(f'{lp4ee_dir}{name}.parquet') for name in cached)
    if not lp4ee_ready:
        print('LP4EE reference missing, computing LP4EE on demand')
        run_baselines(['LP4EE'], cached)

    os.makedirs(Configurations.EXP_PATH, exist_ok=True)
    os.makedirs(Configurations.TABLE_PATH, exist_ok=True)
    os.makedirs(Configurations.PLOT_PATH, exist_ok=True)

    t0 = time.time()
    if is_reset or not os.path.exists(Configurations.EXP1_FILE):
        Exp1Fixed.run()
    if is_reset or not os.path.exists(Configurations.EXP2_FILE):
        Exp2LOOCV.run()
    if is_reset or not os.path.exists(Configurations.EXP3_FILE):
        Exp3Ensemble.run()
    print(f'\nCache-backed experiments finished in {time.time() - t0:.1f} s')

    print('Reporting RQ1 ...')
    rq1_table = Rq1Eval.run(verbose=False)
    rq1_table.to_csv(f'{Configurations.TABLE_PATH}rq1_improvement_counts.csv')

    print('Reporting RQ2 ...')
    rq2_table = Rq2Eval.run(verbose=False)
    Rq2Eval.plot_performance(rq2_table, output_path=f'{Configurations.PLOT_PATH}rq2_loocv_kernel_performance.eps')

    print('Reporting RQ3 ...')
    _, _, rq3_table, _ = Rq3Eval.run(verbose=False)
    Rq3Eval.plot_performance_and_joint_region(
        rq3_table, output_path=f'{Configurations.PLOT_PATH}rq3_ensemble_joint_region.eps')

    Rq3_2Eval.plot_joint_heatmap(
        Configurations.NAMED_KERNEL_SUBSETS, Configurations.NAMED_BANDWIDTH_SUBSETS,
        output_path=f'{Configurations.PLOT_PATH}rq3_2_joint_region.eps')

    KernelDissimilarity.report()
    KernelDissimilarity.plot_grid()
    print('\nRQ1-RQ3 reporting done.')


def run_rq4():
    print('STAGE: rq4')

    from evaluations.stats import Rq4Eval

    for model_name in Configurations.BASELINE_MODEL_NAMES:
        if not os.path.isdir(f'{Configurations.BASELINE_PATH}{model_name}/'):
            raise FileNotFoundError(f'Baseline results for {model_name} not found. Run --stage baselines first.')

    t0 = time.time()
    table, stat_results = Rq4Eval.run()
    os.makedirs(Configurations.TABLE_PATH, exist_ok=True)
    print(f'\nRQ4 finished in {time.time() - t0:.1f} s')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='run.py',
        description=(
            "Entry point for the 'LW-LP4EE Ensemble' experimental pipeline.\n\n"
            'Stages:\n'
            '  cache       Build the LW-LP4EE prediction cache\n'
            '  rq1-rq3     Run cache-backed experiments and report RQ1-RQ3\n'
            '  baselines   Run the eight baselines\n'
            '  rq4         Compare the ensemble against the baselines\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--stage', choices=['cache', 'rq1-rq3', 'baselines', 'rq4'], required=True,
                        help='Pipeline stage to execute.')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Restrict the cache and baselines stages to these datasets (default: all 8).')
    parser.add_argument('--models', nargs='+', default=None, choices=Configurations.BASELINE_MODEL_NAMES,
                        help='Restrict the baselines stage to these models (default: all 8).')
    parser.add_argument('--reset', action='store_true',
                        help='Recompute cached experiment files (rq1-rq3) or baseline predictions (baselines).')
    return parser


def main():
    args = build_parser().parse_args()
    dataset_names = Configurations.DATASET_NAMES if args.datasets is None else args.datasets

    if args.stage == 'cache':
        run_cache(dataset_names)

    elif args.stage == 'rq1-rq3':
        run_rq1_rq3(args.reset)

    elif args.stage == 'baselines':
        run_baselines(Configurations.BASELINE_MODEL_NAMES if args.models is None else args.models,
                      dataset_names, is_reset=args.reset)

    elif args.stage == 'rq4':
        run_rq4()


if __name__ == '__main__':
    main()
