# =============================================================================
# evaluations/stats.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : RQ4 evaluation of the adopted LW-LP4EE ensemble (Diverse3 kernel
#            subset x bandwidth subset 1-9-10, simple mean) against the eight
#            baselines.  For every dataset, all method pairs are compared on
#            paired per-instance errors (absolute error for MAE/MdAE, absolute
#            log error for LSD) with the WRS matched-pairs Brunner test
#            (bprm) and Cliff's delta (cid) through rpy2.  P-values are Holm
#            adjusted per dataset over unique method pairs.  A win/loss
#            requires adjusted p <= 0.05, |delta| >= 0.147 and a better/worse
#            metric; everything else is a tie.
# =============================================================================

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from configs import Configurations
from error_measures import ErrorMeasures

_METRIC_KEY = {'MAE': 'mae', 'MdAE': 'mdae', 'LSD': 'lsd'}


class Rq4Eval:

    KERNEL_SUBSET = Configurations.ENSEMBLE_KERNEL_SUBSET
    BANDWIDTH_SUBSET = Configurations.ENSEMBLE_BANDWIDTH_SUBSET
    METRICS = Configurations.METRICS
    P_THRESHOLD = Configurations.P_THRESHOLD
    DELTA_THRESHOLD = Configurations.DELTA_THRESHOLD
    COMPARE_NAMES = Configurations.COMPARE_NAMES

    RIVAL = list(COMPARE_NAMES.keys())[0]
    METHOD_NAME_MAP = COMPARE_NAMES
    FIXED_ENSEMBLE_PATH = Configurations.EXP3_FILE
    BW_CACHE_DIR = Configurations.CACHE_PATH
    BASELINE_RESULTS_DIR = Configurations.BASELINE_PATH
    STAT_PLOT_DIR = Configurations.PLOT_PATH
    VERBOSE = True
    RAW_VALUE_COLORS = {
        'LW-LP4EE': '#0072B2',
        'Orig.LP4EE': '#D55E00',
        'LP4EE$_{SROpt}$': '#009E73',
    }
    RAW_VALUE_MARKERS = {'MAE': 'o', 'MdAE': 's', 'LSD': '^'}
    RAW_VALUE_DATASET_NAMES = {
        'Cocomo81-e': 'Cocomo81-e',
        'Cocomo81-non-e': 'Cocomo81-non-e',
        'Desharnais': 'Desharnais',
        'Kitchenham': 'Kitchenham',
        'Maxwell': 'Maxwell',
        'Miyazaki94': 'Miyazaki94',
        'Nasa93-c2': 'Nasa93-c2',
        'Nasa93-c5': 'Nasa93-c5',
    }
    RAW_VALUE_ZORDERS = {
        'LW-LP4EE': 5,
        'LP4EE$_{SROpt}$': 4,
        'Orig.LP4EE': 3,
    }

    LP4EE_GROUP = {'LW-LP4EE', 'Orig.LP4EE', 'LP4EE$_{SROpt}$'}


    @staticmethod
    def _finite(values, label):
        x = np.asarray(values, dtype=float).reshape(-1)
        if not len(x) or not np.isfinite(x).all():
            raise ValueError(f'{label} is empty or contains non-finite values')
        return x

    @staticmethod
    def _basis_errors(actual, pred, key):
        actual = np.asarray(actual, float)
        pred = np.maximum(np.asarray(pred, float), 0.01)
        if key == 'ae':
            return np.abs(actual - pred)
        if key == 'abslog':
            return np.abs(ErrorMeasures.log_residuals(actual, pred))

    @staticmethod
    def _scores(actual, pred):
        return {
            'MAE': ErrorMeasures.mae(actual, pred),
            'MdAE': ErrorMeasures.mdae(actual, pred),
            'LSD': ErrorMeasures.lsd(actual, pred),
        }

    @staticmethod
    def _kernel_members():
        members = tuple(part for part in Rq4Eval.KERNEL_SUBSET.split('-') if part)
        if not members:
            raise ValueError(f'Invalid KERNEL_SUBSET {Rq4Eval.KERNEL_SUBSET!r}')
        return members

    @staticmethod
    def _bandwidth_members():
        try:
            members = tuple(int(part) / 10.0 for part in Rq4Eval.BANDWIDTH_SUBSET.split('-'))
        except ValueError as exc:
            raise ValueError(f'Invalid BANDWIDTH_SUBSET {Rq4Eval.BANDWIDTH_SUBSET!r}') from exc
        if not members:
            raise ValueError(f'Invalid BANDWIDTH_SUBSET {Rq4Eval.BANDWIDTH_SUBSET!r}')
        return members

    @staticmethod
    def _display_name(method):
        if method in Rq4Eval.METHOD_NAME_MAP:
            return Rq4Eval.METHOD_NAME_MAP[method]
        if method in Rq4Eval.COMPARE_NAMES:
            return Rq4Eval.COMPARE_NAMES[method]
        return method

    @staticmethod
    def _latex_name(method):
        name = Rq4Eval._display_name(method)
        if '\\' in name or '$' in name:
            return name
        return name.replace('_', r'\_')

    @staticmethod
    def _read_parquet(path, columns, filters=None):
        path = Path(path)
        if filters:
            try:
                return pd.read_parquet(path, engine='pyarrow', columns=columns, filters=filters)
            except Exception:
                frame = pd.read_parquet(path, columns=list(dict.fromkeys(
                    [*columns, *[item[0] for item in filters]])))
                for column, _op, value in filters:
                    frame = frame[frame[column] == value]
                return frame.loc[:, columns].reset_index(drop=True)
        return pd.read_parquet(path, columns=columns)


    @staticmethod
    def _load_fixed_ensemble(dataset, cache_dir):
        kernels = set(Rq4Eval._kernel_members())
        bandwidths = np.asarray(Rq4Eval._bandwidth_members(), float)
        path = Path(cache_dir) / f'{dataset}.parquet'
        frame = pd.read_parquet(path, columns=['index', 'actual', 'kernel', 'bandwidth', 'pred'])
        bandwidth = frame['bandwidth'].to_numpy(float)
        selected = frame['kernel'].isin(kernels) & np.any(
            np.isclose(bandwidth[:, None], bandwidths[None, :], rtol=0, atol=1e-8), axis=1)
        frame = frame.loc[selected]
        expected = len(kernels) * len(bandwidths)
        if frame.empty or not (frame.groupby('index').size() == expected).all():
            raise ValueError(f'Incomplete selected ensemble members in {path}')
        if not (frame.groupby('index')['actual'].nunique(dropna=False) == 1).all():
            raise ValueError(f'Inconsistent ensemble actual values in {path}')
        grouped = frame.groupby('index', sort=True)
        actual = Rq4Eval._finite(grouped['actual'].first().to_numpy(), f'{dataset} actual')
        pred = np.maximum(Rq4Eval._finite(grouped['pred'].mean().to_numpy(), f'{dataset} ensemble'), 0.0)
        return actual, pred


    @staticmethod
    def _load_baseline(dataset, model, root, expected_actual, cache):
        root = Path(root).resolve()
        key = (str(root), dataset, model)
        if key not in cache:
            column = f'pred__{model}'
            path = root / model / f'{dataset}.parquet'
            if not path.exists():
                raise FileNotFoundError(f'Baseline prediction file not found: {path}')
            frame = pd.read_parquet(path, columns=['actual', column])
            actual = Rq4Eval._finite(frame['actual'], f'{path} actual')
            pred = Rq4Eval._finite(frame[column], f'{path} {column}')
            if actual.shape != expected_actual.shape or not np.allclose(
                    actual, expected_actual, rtol=0, atol=1e-12):
                raise ValueError(f'Actual values are not aligned for {dataset}/{model}')
            cache[key] = pred
        return np.maximum(cache[key], 0.0)

    @staticmethod
    def _baseline_model(key):
        if key in ('Orig', 'LP4EE'):
            return 'LP4EE'
        return key

    @staticmethod
    def _load_methods(dataset, actual, ensemble, models, cache):
        predictions = {'Ensemble': ensemble}
        for key in models:
            if key == 'Ensemble':
                continue
            predictions[key] = Rq4Eval._load_baseline(
                dataset, Rq4Eval._baseline_model(key), Rq4Eval.BASELINE_RESULTS_DIR,
                actual, cache)
        return predictions


    @staticmethod
    def _r_scalar(result, key):
        try:
            value = float(np.asarray(result.rx2(key), dtype=float).reshape(-1)[0])
        except Exception as exc:
            raise RuntimeError(f'R WRS result did not contain numeric {key!r}') from exc
        if not np.isfinite(value):
            raise RuntimeError(f'R WRS returned non-finite {key!r}')
        return value

    @staticmethod
    def _paired_test(x, y, wrs):
        x, y = Rq4Eval._finite(x, 'test x'), Rq4Eval._finite(y, 'test y')
        if x.shape != y.shape:
            raise ValueError(f'Paired arrays differ: {x.shape} vs {y.shape}')
        if len(x) < 2 or np.allclose(x, y, rtol=0, atol=1e-12):
            return 1.0, 0.0
        from rpy2.robjects import numpy2ri
        return (Rq4Eval._r_scalar(wrs.bprm(numpy2ri.py2rpy(x), numpy2ri.py2rpy(y)), 'p.value'),
                Rq4Eval._r_scalar(wrs.cid(numpy2ri.py2rpy(x), numpy2ri.py2rpy(y)), 'd'))

    @staticmethod
    def _raw_rows(dataset, actual, predictions, wrs, cache):
        perf = {}
        keys = list(dict.fromkeys(['ae', 'abslog']))
        for method, pred in predictions.items():
            pred = Rq4Eval._finite(pred, f'{dataset}/{method}')
            if pred.shape != actual.shape:
                raise ValueError(f'Shape mismatch for {dataset}/{method}')
            perf[method] = {
                **{f'ERR_{key}': Rq4Eval._basis_errors(actual, pred, key) for key in keys},
                **Rq4Eval._scores(actual, pred),
            }
        rows = []
        for left, right in itertools.combinations(predictions, 2):
            stats = {}
            for key in keys:
                x, y = perf[left][f'ERR_{key}'], perf[right][f'ERR_{key}']
                cache_key = (dataset, key, left, right, x.tobytes(), y.tobytes())
                if cache_key not in cache:
                    cache[cache_key] = Rq4Eval._paired_test(x, y, wrs)
                stats[key] = cache[cache_key]
            common = {'dataset': dataset}
            for key, (p_value, _delta) in stats.items():
                common[f'p_value_{key}'] = p_value
            for method_a, method_b, sign in ((left, right, 1.0), (right, left, -1.0)):
                rows.append({
                    **common, 'method_a': method_a, 'method_b': method_b,
                    **{f'cliff_delta_{key}': sign * delta for key, (_p, delta) in stats.items()},
                    **{f'{_METRIC_KEY[m]}_a': perf[method_a][m] for m in Rq4Eval.METRICS},
                    **{f'{_METRIC_KEY[m]}_b': perf[method_b][m] for m in Rq4Eval.METRICS},
                })
        return rows

    @staticmethod
    def _holm(p_values):
        p = np.asarray(p_values, dtype=float)
        m = len(p)
        if m == 0:
            return p
        order = np.argsort(p)
        ranked = p[order] * (m - np.arange(m))
        ranked = np.maximum.accumulate(ranked)
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(ranked, 1.0)
        return adjusted


    @staticmethod
    def _adjusted_p(out, key):
        column = f'p_value_{key}'
        pair = [tuple(sorted(item)) for item in zip(out['method_a'], out['method_b'])]
        frame = pd.DataFrame({'dataset': out['dataset'].to_numpy(), 'pair': pair,
                              'p': out[column].to_numpy()}, index=out.index)
        unique = frame.drop_duplicates(['dataset', 'pair']).copy()
        unique['p_adj'] = unique.groupby('dataset')['p'].transform(
            lambda s: pd.Series(Rq4Eval._holm(s.to_numpy()), index=s.index))
        lookup = unique.set_index(['dataset', 'pair'])['p_adj']
        return pd.Series(lookup.loc[list(zip(frame['dataset'], frame['pair']))].to_numpy(),
                         index=out.index)

    @staticmethod
    def _test_key(metric):
        if metric == 'LSD':
            return 'abslog'
        return 'ae'

    @staticmethod
    def _required_columns():
        keys = {Rq4Eval._test_key(m) for m in Rq4Eval.METRICS}
        return {f'{kind}_{key}' for key in keys for kind in ('p_value', 'cliff_delta')}

    @staticmethod
    def apply_wtl(raw):
        required = Rq4Eval._required_columns()
        out = raw.copy()
        keys = sorted({Rq4Eval._test_key(m) for m in Rq4Eval.METRICS})
        for key in keys:
            out[f'p_adj_{key}'] = Rq4Eval._adjusted_p(out, key)
        for metric in Rq4Eval.METRICS:
            key = Rq4Eval._test_key(metric)
            short = _METRIC_KEY[metric]
            delta = out[f'cliff_delta_{key}']
            retained = ((out[f'p_adj_{key}'] <= Rq4Eval.P_THRESHOLD)
                        & (delta.abs() >= Rq4Eval.DELTA_THRESHOLD))
            better = out[f'{short}_a'] < out[f'{short}_b']
            worse = out[f'{short}_a'] > out[f'{short}_b']
            win = retained & better
            loss = retained & worse
            out[f'w_{short}'] = win.astype(int)
            out[f'l_{short}'] = loss.astype(int)
            out[f't_{short}'] = (~(win | loss)).astype(int)
            out[f'wl_{short}'] = out[f'w_{short}'] - out[f'l_{short}']
        return out

    @staticmethod
    def summarize(wtl):
        shorts = [_METRIC_KEY[m] for m in Rq4Eval.METRICS]
        agg = {f'{m}_{k}': (f'{k}_{m}', 'sum') for m in shorts for k in ('w', 't', 'l', 'wl')}
        overall = wtl.groupby('method_a').agg(**agg).sort_values(
            [f'{m}_wl' for m in shorts], ascending=False)
        by_dataset = wtl.groupby(['dataset', 'method_a']).agg(**agg).sort_index()
        return overall, by_dataset

    @staticmethod
    def _rank_with_ties(frame, axis=1, ascending=True, decimals=2, method='min'):
        return frame.round(decimals).rank(axis=axis, ascending=ascending, method=method)

    @staticmethod
    def _wtl_table_csv(overall):
        shorts = [_METRIC_KEY[m] for m in Rq4Eval.METRICS]
        labels = {'w': 'Wins', 'l': 'Loss', 'wl': 'Net Wins'}
        table = pd.DataFrame(index=overall.index)
        for metric, short in zip(Rq4Eval.METRICS, shorts):
            for kind, label in labels.items():
                table[f'{metric} {label}'] = overall[f'{short}_{kind}'].astype(int)
        table['Total Net Wins'] = overall[[f'{s}_wl' for s in shorts]].sum(axis=1).astype(int)
        table = table.sort_values('Total Net Wins', ascending=False, kind='stable')
        table.index = [Rq4Eval._display_name(m) for m in table.index]
        table.index.name = 'Method'
        return table



    @staticmethod
    def _raw_value_positions(table):
        if table.empty or table.index.has_duplicates:
            raise ValueError('Raw-value table must have unique, nonempty datasets')
        methods = list(Rq4Eval.COMPARE_NAMES)
        metrics = list(Rq4Eval.METRICS)
        if not methods or not metrics:
            raise ValueError('Raw-value plot needs methods and metrics')
        missing_markers = set(metrics) - set(Rq4Eval.RAW_VALUE_MARKERS)
        if missing_markers:
            raise ValueError(f'No raw-value markers for {sorted(missing_markers)}')
        columns = [f'{method}_{metric}' for metric in metrics for method in methods]
        missing = sorted(set(columns) - set(table.columns))
        if missing:
            raise ValueError(f'Raw-value table missing columns: {missing}')
        rows = []
        for dataset, row in table.iterrows():
            for metric in metrics:
                values = Rq4Eval._finite(
                    [row[f'{method}_{metric}'] for method in methods],
                    f'{dataset}/{metric} raw errors')
                low, high = float(values.min()), float(values.max())
                positions = (100.0 * ((values - low) / (high - low))
                             if high > low else np.full(len(values), 50.0))
                for method, value, position in zip(methods, values, positions):
                    rows.append({
                        'dataset': dataset, 'metric': metric, 'method': method,
                        'display_name': Rq4Eval._display_name(method),
                        'raw_value': value, 'minimum': low, 'maximum': high,
                        'range_position_pct': position,
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def _plot_raw_value_figure(table, plot_dir=None, show=False):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.text import Text

        names = Rq4Eval.RAW_VALUE_DATASET_NAMES
        order = {name: i for i, name in enumerate(dict.fromkeys(names.values()))}
        datasets = sorted(table.index,
                          key=lambda dataset: order.get(names.get(dataset, dataset), len(order)))
        table = table.loc[datasets]
        data = Rq4Eval._raw_value_positions(table)
        metrics = list(Rq4Eval.METRICS)
        methods = list(Rq4Eval.COMPARE_NAMES)
        highlighted = [m for m in methods
                       if Rq4Eval._display_name(m) in Rq4Eval.RAW_VALUE_COLORS]
        ncols = min(2, len(table))
        nrows = (len(table) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(10 if ncols == 2 else 9, max(4.8, 1.15 * nrows + 2.4)),
            squeeze=False)
        decimals = {'MAE': 2, 'MdAE': 2, 'LSD': 3}
        for panel, dataset in enumerate(table.index):
            column, row = divmod(panel, nrows)
            ax = axes[row, column]
            subset = data[data['dataset'] == dataset]
            for lane, metric in enumerate(metrics):
                group = subset[subset['metric'] == metric].set_index('method')
                marker = Rq4Eval.RAW_VALUE_MARKERS[metric]
                ax.hlines(lane, 0, 100, color='#bbbbbb', linewidth=1, zorder=1)
                for method in methods:
                    name = Rq4Eval._display_name(method)
                    colored = method in highlighted
                    ax.scatter(
                        group.loc[method, 'range_position_pct'],
                        lane, marker=marker,
                        s=110 if colored else 65,
                        color=Rq4Eval.RAW_VALUE_COLORS.get(name, '#a0a0a0'),
                        edgecolors='white', linewidths=0.7,
                        zorder=Rq4Eval.RAW_VALUE_ZORDERS.get(name, 2))
                places = decimals[metric]
                for x, field, align in ((-5, 'minimum', 'right'), (105, 'maximum', 'left')):
                    ax.text(x, lane, f'{group.iloc[0][field]:.{places}f}',
                            ha=align, va='center', fontsize=11, color='#555555')
            ax.set_xlim(-35, 135)
            ax.set_ylim(len(metrics) - 0.45, -0.45)
            ax.set_yticks(range(len(metrics)), metrics, fontsize=13, va='center')
            ax.set_title(names.get(dataset, str(dataset)), loc='left', pad=8,
                         fontsize=14, fontweight='bold')
            ax.set_xticks([0, 25, 50, 75, 100])
            bottom_panel = panel == min((column + 1) * nrows, len(table)) - 1
            ax.tick_params(axis='x', labelbottom=bottom_panel,
                           length=0, labelsize=12)
            ax.tick_params(axis='y', length=0)
            ax.grid(axis='x', color='#eeeeee', linewidth=0.7)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if bottom_panel:
                ax.set_xlabel('Raw-error range position (%) · lower is better', fontsize=13)
        for unused in range(len(table), nrows * ncols):
            column, row = divmod(unused, nrows)
            fig.delaxes(axes[row, column])
        handles = [Line2D([], [], marker='o', linestyle='none', markersize=10,
                          color=Rq4Eval.RAW_VALUE_COLORS[Rq4Eval._display_name(m)],
                          label=Rq4Eval._display_name(m)) for m in highlighted]
        handles.append(Line2D([], [], marker='o', linestyle='none', markersize=9, color='#a0a0a0',
                              label='Other methods'))
        metric_handles = [Line2D([], [], marker=Rq4Eval.RAW_VALUE_MARKERS[m],
                                 linestyle='none', markersize=9, color='#333333', label=m)
                          for m in metrics]
        handles.insert(0, Line2D([], [], linestyle='none', label='Method:'))
        metric_handles.insert(0, Line2D([], [], linestyle='none', label='Metric:'))
        legend_columns = max(len(handles), len(metric_handles))
        legend_handles = []
        for column in range(legend_columns):
            for row_handles in (handles, metric_handles):
                legend_handles.append(row_handles[column] if column < len(row_handles)
                                      else Line2D([], [], linestyle='none', label=''))
        fig.suptitle('Raw prediction errors by dataset', y=1.00, fontsize=19, fontweight='bold')
        legend = fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.52, -0.00),
                            ncol=legend_columns, frameon=True, framealpha=1, fontsize=12,
                            labelspacing=0.8, borderpad=0.7, handleheight=2.2)
        for text in legend.get_texts():
            if text.get_text() in ('Method:', 'Metric:'):
                text.set_fontsize(13)
                text.set_fontweight('bold')

        for text in fig.findobj(match=Text):
            text.set_zorder(6)
        fig.subplots_adjust(left=0.065, right=0.98, top=0.93, bottom=0.16,
                            hspace=0.65, wspace=0.20)
        root = Path(Rq4Eval.STAT_PLOT_DIR if plot_dir is None else plot_dir)
        root.mkdir(parents=True, exist_ok=True)
        paths = {}
        try:
            path = root / f'rq4_raw_error_range_by_dataset.eps'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f'Raw-value plot saved to {path}')
            if show:
                plt.show()
        finally:
            plt.close(fig)
        return paths


    @staticmethod
    def _ensemble_metrics_frame():
        frame = Rq4Eval._read_parquet(
            Rq4Eval.FIXED_ENSEMBLE_PATH,
            ['dataset', 'kernel_subset', 'bandwidth_subset', *Rq4Eval.METRICS],
            [('kernel_subset', '==', Rq4Eval.KERNEL_SUBSET),
             ('bandwidth_subset', '==', Rq4Eval.BANDWIDTH_SUBSET)])
        if frame['dataset'].duplicated().any():
            duplicates = sorted(frame.loc[frame['dataset'].duplicated(), 'dataset'].unique())
            raise ValueError(f'Ensemble result has duplicate datasets: {duplicates}')
        if frame.empty:
            raise ValueError('No ensemble rows selected')
        return frame.set_index('dataset')[Rq4Eval.METRICS]

    @staticmethod
    def _iter_ensembles(datasets):
        for dataset in datasets:
            yield dataset, Rq4Eval._load_fixed_ensemble(dataset, Rq4Eval.BW_CACHE_DIR)

    @staticmethod
    def _check_ensemble_metrics(dataset, actual, pred, reference):
        scores = Rq4Eval._scores(actual, pred)
        for metric in Rq4Eval.METRICS:
            if not np.isclose(scores[metric], reference[metric], rtol=0, atol=1e-6):
                raise ValueError(
                    f'{dataset} {metric} from configured predictions '
                    f'({scores[metric]}) != ensemble file ({reference[metric]})')
        return scores

    @staticmethod
    def run():
        reference = Rq4Eval._ensemble_metrics_frame()
        datasets = list(reference.index)
        models = list(Rq4Eval.COMPARE_NAMES)
        stat_models = list(Rq4Eval.METHOD_NAME_MAP)
        pred_cache = {}
        loaded = {}
        rows = []
        for dataset, (actual, ensemble) in Rq4Eval._iter_ensembles(datasets):
            predictions = Rq4Eval._load_methods(dataset, actual, ensemble, stat_models, pred_cache)
            loaded[dataset] = (actual, predictions)
            scores = {key: Rq4Eval._scores(actual, predictions[key]) for key in models}
            ensemble_scores = scores.get('Ensemble', Rq4Eval._scores(actual, ensemble))
            rival = Rq4Eval._scores(actual, predictions[Rq4Eval.RIVAL])
            row = {
                'dataset': dataset,
                'ensemble_wins': sum(ensemble_scores[m] < rival[m] for m in Rq4Eval.METRICS),
            }
            for key in models:
                for metric in Rq4Eval.METRICS:
                    row[f'{key}_{metric}'] = scores[key][metric]
            rows.append(row)

        table = pd.DataFrame(rows).set_index('dataset')

        raw_value_paths = Rq4Eval._plot_raw_value_figure(table, plot_dir=Rq4Eval.STAT_PLOT_DIR)
        table.attrs['plot_raw_values'] = raw_value_paths
        stat_results = Rq4Eval._run_stat_tests(loaded, table)
        stat_results['plot_raw_values'] = raw_value_paths
        return table, stat_results

    @staticmethod
    def _run_stat_tests(loaded, table):
        if Rq4Eval.VERBOSE:
            print('\n' + '=' * 100)
            print('WTL statistical comparison (deterministic single analysis)')
            print('=' * 100)
        try:
            from rpy2.robjects.packages import importr
        except ImportError as exc:
            raise RuntimeError('rpy2 is required for WRS bprm/cid tests') from exc
        wrs = importr('WRS')
        rows, test_cache = [], {}
        for i, dataset in enumerate(loaded, 1):
            actual, predictions = loaded[dataset]
            compared = {key: predictions[key] for key in Rq4Eval.METHOD_NAME_MAP}
            rows.extend(Rq4Eval._raw_rows(dataset, actual, compared, wrs, test_cache))
            print(f'[{i}/{len(loaded)}] {dataset}: {len(compared)} methods, '
                  f'{len(compared) * (len(compared) - 1)} directed pairs')
        raw = pd.DataFrame(rows)
        wtl = Rq4Eval.apply_wtl(raw)
        overall, _ = Rq4Eval.summarize(wtl)
        wtl_table = Rq4Eval._wtl_table_csv(overall)
        out_dir = Path(Configurations.TABLE_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        wtl_table.to_csv(out_dir / 'rq4_wtl_summary.csv')
        if Rq4Eval.VERBOSE:
            print('\nWTL summary (wins, loss, net wins):')
            print(wtl_table.to_string())
        return {'wtl_table': wtl_table}
