# =============================================================================
# evaluations/plots.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : Reporting for RQ1-RQ3, computed from the cache-backed experiments.
# =============================================================================

import re
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from configs import Configurations
from error_measures import ErrorMeasures


def _apply_dataset_ylim(ax, n_datasets=None):
    """Y range of the trend panels: axis break at n/2, top at n (datasets)."""
    if n_datasets is None:
        n_datasets = len(Configurations.DATASET_NAMES)
    y_break = n_datasets / 2
    ax.set_ylim(y_break - 0.5, n_datasets + 0.2)
    ax.set_yticks(np.arange(np.ceil(y_break), n_datasets + 1, 1))


class Lp4eeReference:

    @staticmethod
    def load_metrics(datasets=None):
        path_dir = f'{Configurations.BASELINE_PATH}LP4EE/'
        if datasets is None:
            datasets = sorted(f[:-len('.parquet')] for f in os.listdir(path_dir) if f.endswith('.parquet'))
        rows = []
        for dataset in datasets:
            frame = pd.read_parquet(f'{path_dir}{dataset}.parquet', columns=['actual', 'pred__LP4EE'])
            actual = frame['actual'].to_numpy(float)
            pred = np.maximum(frame['pred__LP4EE'].to_numpy(float), 0.01)
            rows.append({
                'dataset': dataset,
                'MAE': ErrorMeasures.mae(actual, pred),
                'MdAE': ErrorMeasures.mdae(actual, pred),
                'LSD': ErrorMeasures.lsd(actual, pred),
            })
        return pd.DataFrame(rows).set_index('dataset')[list(Configurations.METRICS)]


class Rq1Eval:

    METRICS = Configurations.METRICS
    KERNELS = Configurations.KERNELS

    @staticmethod
    def run(lw_path=Configurations.EXP1_FILE,
            verbose=True):
        lw = pd.read_parquet(lw_path)
        lp = Lp4eeReference.load_metrics().reset_index()[['dataset'] + Rq1Eval.METRICS]
        lp = lp.rename(columns={m: f'{m}_orig' for m in Rq1Eval.METRICS})

        df = lw.merge(lp, on='dataset', how='left')

        detailed_mask = pd.concat([df[m] < df[f'{m}_orig'] for m in Rq1Eval.METRICS], axis=1)
        detailed_mask.columns = Rq1Eval.METRICS

        def _ordered(t):
            return t.reindex([k for k in Rq1Eval.KERNELS if k in t.index])

        sub_df = df[['kernel', 'bandwidth']].join(detailed_mask)
        per_metric = {
            metric: _ordered(
                sub_df.groupby(['kernel', 'bandwidth'])[metric].sum().unstack(level=-1)
            )
            for metric in Rq1Eval.METRICS
        }

        index = per_metric[Rq1Eval.METRICS[0]].index
        columns = per_metric[Rq1Eval.METRICS[0]].columns
        detailed_table = pd.DataFrame(index=index, columns=columns)
        for kernel in detailed_table.index:
            for bw in detailed_table.columns:
                detailed_table.loc[kernel, bw] = '/'.join(
                    str(int(per_metric[metric].loc[kernel, bw])) for metric in Rq1Eval.METRICS
                )

        total_df = per_metric[Rq1Eval.METRICS[0]].copy()
        for metric in Rq1Eval.METRICS[1:]:
            total_df = total_df.add(per_metric[metric])

        average_df = total_df / len(Rq1Eval.METRICS)
        row_averages = average_df.mean(axis=1).map(lambda v: f'{v:.2f}')
        col_averages = average_df.mean(axis=0).map(lambda v: f'{v:.2f}')

        detailed_table['Average'] = row_averages.reindex(detailed_table.index)
        averages_row = col_averages.reindex(columns)
        averages_row['Average'] = ''
        detailed_table.loc['Average'] = averages_row

        if verbose:
            pd.set_option('display.width', 200)
            pd.set_option('display.max_columns', None)
            print(detailed_table)

        return detailed_table


class Rq2Eval:

    METRICS = Configurations.METRICS
    KERNELS = Configurations.KERNELS
    HIGHLIGHT_PALETTE = (
        '#0072B2', '#009E73', '#D55E00',
        '#8C564B', '#9467BD', '#CC79A7',
    )

    @staticmethod
    def run(cv_path=Configurations.EXP2_FILE, verbose=True):
        lw = pd.read_parquet(cv_path)

        lp = Lp4eeReference.load_metrics().reset_index()[['dataset'] + Rq2Eval.METRICS]
        lp = lp.rename(columns={m: f'{m}_orig' for m in Rq2Eval.METRICS})
        df = lw.merge(lp, on='dataset', how='left')
        sub_df = df.copy()
        mask = pd.concat([df[m] < df[f'{m}_orig'] for m in Rq2Eval.METRICS], axis=1).sum(axis=1)
        detailed_mask = pd.concat([df[m] < df[f'{m}_orig'] for m in Rq2Eval.METRICS], axis=1)
        detailed_mask.columns = ['MAE', 'MdAE', 'LSD']
        df['mask'] = mask > 2


        grouped = df.groupby(['kernel', 'subset'])['mask'].agg(mask='sum').reset_index()
        grouped['subset_size'] = grouped['subset'].str.split('-').apply(len)
        grouped = grouped.groupby(['kernel', 'subset_size'])['mask'].agg("max")
        table = grouped.unstack(level=-1)

        sub_df = sub_df.drop(['MAE', 'MdAE', 'LSD', 'MAE_orig', 'MdAE_orig', 'LSD_orig'], axis=1)
        sub_df[['MAE', 'MdAE', 'LSD']] = detailed_mask
        detailed_grouped = sub_df.groupby(['kernel', 'subset'])[['MAE', 'MdAE', 'LSD']].sum().sum(axis=1).reset_index()
        detailed_grouped['subset_size'] = detailed_grouped['subset'].str.split('-').apply(len)
        detailed_grouped = detailed_grouped.groupby(['kernel', 'subset_size'])[0].agg("max")
        detailed_table = detailed_grouped.unstack(level=-1)

        combined_table = pd.DataFrame(index=table.index, columns=table.columns, dtype=object)
        for kernel in combined_table.index:
            for subset_size in combined_table.columns:
                x = table.loc[kernel, subset_size]
                y = detailed_table.loc[kernel, subset_size]
                if pd.isna(x) or pd.isna(y):
                    combined_table.loc[kernel, subset_size] = 'NA'
                else:
                    combined_table.loc[kernel, subset_size] = f"{int(x)} ({int(y)})"

        def _y(cell):
            return int(cell.split('(')[-1][:-1]) if isinstance(cell, str) and cell != 'NA' else 0

        combined_table.loc['Total'] = combined_table.map(_y).sum(axis=0)
        combined_table['Total'] = combined_table.map(
            lambda s: _y(s) if isinstance(s, str) else s
        ).sum(axis=1)

        if verbose:
            pd.set_option('display.width', 200)
            pd.set_option('display.max_columns', None)
            print(combined_table)

        return combined_table

    @staticmethod
    def _kernel_rank_scores(combined_table):

        if 'Total' in combined_table.columns:
            scores = pd.to_numeric(
                combined_table['Total'].drop(index='Total', errors='ignore'),
                errors='coerce',
            )
            if scores.notna().any():
                return scores.sort_values(ascending=False)

        _, metric_scores = Rq2Eval._performance_matrices(combined_table)
        return metric_scores.mean(axis=1).sort_values(ascending=False)

    @staticmethod
    def _configure_highlight_colors(combined_table, n_top=3, n_bottom=2):

        ranked = Rq2Eval._kernel_rank_scores(combined_table)
        strongest = ranked.index[:n_top].tolist()
        remaining = ranked.index[n_top:]
        weakest = remaining[-n_bottom:].tolist() if len(remaining) else []
        highlighted = strongest + weakest
        Rq2Eval.HIGHLIGHT_COLORS = dict(
            zip(highlighted, Rq2Eval.HIGHLIGHT_PALETTE[:len(highlighted)])
        )
        return strongest, weakest

    @staticmethod
    def _performance_matrices(combined_table):

        body = combined_table.drop(index='Total', errors='ignore').drop(
            columns='Total', errors='ignore'
        )
        all_three_wins = pd.DataFrame(index=body.index, columns=body.columns, dtype=float)
        metric_scores = pd.DataFrame(index=body.index, columns=body.columns, dtype=float)

        for kernel in body.index:
            for subset_size in body.columns:
                match = re.fullmatch(r'\s*(\d+)\s*\(\s*(\d+)\s*\)\s*', str(body.loc[kernel, subset_size]))
                if match is None:
                    raise ValueError(
                        f"Cell ({kernel!r}, {subset_size!r}) must use the 'x (y)' format"
                    )
                all_three_wins.loc[kernel, subset_size] = int(match.group(1))
                metric_scores.loc[kernel, subset_size] = int(match.group(2))

        ordered_columns = sorted(metric_scores.columns, key=int)
        return all_three_wins[ordered_columns], metric_scores[ordered_columns]

    @staticmethod
    def plot_performance(combined_table, output_path=None, show=False):

        strongest, weakest = Rq2Eval._configure_highlight_colors(combined_table)
        _, metric_scores = Rq2Eval._performance_matrices(combined_table)
        average_wins = metric_scores / len(Rq2Eval.METRICS)
        subset_sizes = np.asarray(average_wins.columns, dtype=int)
        kernel_averages = average_wins.mean(axis=1).sort_values(ascending=False)
        kernel_stds = average_wins.std(axis=1, ddof=1)
        highlighted = strongest + weakest
        monotone_kernels = [
            kernel for kernel in average_wins.index
            if average_wins.loc[kernel].is_monotonic_decreasing
        ]
        endpoint_declines = [
            kernel for kernel in average_wins.index
            if average_wins.loc[kernel].iloc[-1] < average_wins.loc[kernel].iloc[0]
        ]

        fig, (ax_trend, ax_total) = plt.subplots(
            1, 2, figsize=(11,  5.5),
            gridspec_kw={'width_ratios': (2, 1)},
            constrained_layout=True,
        )
        fig.patch.set_facecolor('white')

        for kernel in average_wins.index:
            if kernel in highlighted:
                continue
            ax_trend.plot(subset_sizes, average_wins.loc[kernel], color='#c7c7c7',
                          linestyle='--', linewidth=2, alpha=0.8, zorder=1)
            ax_trend.scatter(
                subset_sizes, average_wins.loc[kernel], s=48,
                color='#c7c7c7', edgecolor='white', linewidth=0.6,
                alpha=0.85, zorder=2,
            )

        line_handles = []
        for kernel in highlighted:
            color = Rq2Eval.HIGHLIGHT_COLORS[kernel]
            is_weak = kernel in weakest
            line, = ax_trend.plot(
                subset_sizes, average_wins.loc[kernel],
                color=color, linewidth=3 if not is_weak else 2,
                linestyle='--' if is_weak else '--', label=kernel,
                zorder=3,
            )
            ax_trend.scatter(
                subset_sizes, average_wins.loc[kernel], s=48,
                color=color, edgecolor='white', linewidth=0.9, zorder=4,
            )
            line_handles.append(line)
        else_handle, = ax_trend.plot(
            [], [], color='#c7c7c7', linestyle='--' , linewidth=2, label='else', zorder=1,
        )
        line_handles.append(else_handle)

        ax_trend.set_title('(a) Highest achievable improvement\ncount by subset size',
                           loc='center', fontsize=16.5, fontweight='bold', pad=12)
        ax_trend.set_xlabel('Bandwidth subset size', fontsize=15, labelpad=8)
        ax_trend.set_ylabel('Mean joint improvement count', fontsize=15, labelpad=10)
        ax_trend.set_xticks(subset_sizes)
        _apply_dataset_ylim(ax_trend)
        ax_trend.tick_params(axis='both', labelsize=15)
        ax_trend.grid(axis='y', color='#e5e5e5', linewidth=0.8)
        ax_trend.spines[['top', 'right']].set_visible(False)
        break_style = {
            'transform': ax_trend.transAxes,
            'color': '#333333',
            'clip_on': False,
            'linewidth': 1.2,
        }
        ax_trend.plot((-0.012, 0.012), (-0.018, 0.018), **break_style)
        ax_trend.plot((-0.012, 0.012), (0.018, 0.054), **break_style)

        ax_trend.legend(
            handles=line_handles, title='Kernel', loc='upper right',
            ncol=2, frameon=False, fontsize=14, title_fontsize=15,
        )

        ranked = kernel_averages.sort_values(ascending=True)
        ranked_stds = kernel_stds.reindex(ranked.index)
        bar_colors = [
            Rq2Eval.HIGHLIGHT_COLORS.get(kernel, '#c7c7c7')
            for kernel in ranked.index
        ]
        bars = ax_total.barh(ranked.index, ranked.values, color=bar_colors,
                             edgecolor='white', linewidth=0.7,
                             xerr=ranked_stds.values,
                             error_kw={'ecolor': '#4d4d4d', 'elinewidth': 1.2,
                                       'capsize': 3, 'capthick': 1.2})
        for bar, (kernel, value), std in zip(bars, ranked.items(), ranked_stds):
            ax_total.text(value + std + 0.18, bar.get_y() + bar.get_height() / 2,
                          rf'{value:.2f} $\pm$ {std:.2f}', va='center', fontsize=12,
                          fontweight='bold' if kernel in highlighted else 'normal')
        ax_total.set_title('(b) Kernel ranking\n by mean of the per-size\n best counts',
                           loc='center', fontsize=16, fontweight='bold', pad=14)
        ax_total.set_xlabel('Mean of the per-size \n best joint improvement counts',
                            fontsize=15, labelpad=8)
        ax_total.set_xlim(0, 14)
        ax_total.set_xticks(np.arange(0, 15, 2))
        ax_total.tick_params(axis='x', labelsize=15)
        ax_total.tick_params(axis='y', labelsize=13)
        ax_total.grid(axis='x', color='#e5e5e5', linewidth=0.8)
        ax_total.set_axisbelow(True)
        ax_total.spines[['top', 'right', 'left']].set_visible(False)
        ax_total.tick_params(axis='y', length=0)
        for label in ax_total.get_yticklabels():
            kernel = label.get_text()
            if kernel in highlighted:
                label.set_color(Rq2Eval.HIGHLIGHT_COLORS[kernel])
                label.set_fontweight('bold')


        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        summary = {
            'average_wins': average_wins,
            'kernel_averages': kernel_averages,
            'kernel_stds': kernel_stds,
            'strongest': strongest,
            'weakest': weakest,
            'monotone_kernels': monotone_kernels,
            'endpoint_declines': endpoint_declines,
        }
        return fig, (ax_trend, ax_total), summary


class Rq3Eval:

    METRICS = Configurations.METRICS
    KERNELS = Configurations.KERNELS
    HIGHLIGHT_COLORS = ('#0072B2', '#009E73', '#D55E00', '#8C564B', '#9467BD')

    @staticmethod
    def _per_subset_scores(ensemble_path=Configurations.EXP3_FILE):

        ens = pd.read_parquet(ensemble_path)
        lp = Lp4eeReference.load_metrics().reset_index()[['dataset'] + Rq3Eval.METRICS]
        lp = lp.rename(columns={m: f'{m}_orig' for m in Rq3Eval.METRICS})
        df = ens.merge(lp, on='dataset', how='left')

        detailed_mask = pd.concat([df[m] < df[f'{m}_orig'] for m in Rq3Eval.METRICS], axis=1)
        detailed_mask.columns = Rq3Eval.METRICS
        df['better'] = detailed_mask.sum(axis=1) > 2

        per_subset = df.groupby(['kernel_subset', 'bandwidth_subset'])['better'].agg(mask='sum').reset_index()

        sub_df = df[['kernel_subset', 'bandwidth_subset']].join(detailed_mask)
        metric_score = (
            sub_df.groupby(['kernel_subset', 'bandwidth_subset'])[Rq3Eval.METRICS]
            .sum().sum(axis=1).rename('metric_score').reset_index()
        )

        per_subset = per_subset.merge(metric_score, on=['kernel_subset', 'bandwidth_subset'])
        per_subset['kernel_subset_size'] = per_subset['kernel_subset'].str.split('-').apply(len)
        per_subset['bandwidth_subset_size'] = per_subset['bandwidth_subset'].str.split('-').apply(len)
        return per_subset

    @staticmethod
    def run(ensemble_path=Configurations.EXP3_FILE,
            lp_path=None,
            verbose=True):
        per_subset = Rq3Eval._per_subset_scores(ensemble_path)

        table = per_subset.groupby(['kernel_subset_size', 'bandwidth_subset_size'])['mask'] \
            .agg(['max']).unstack(level=-1)
        detailed_table = per_subset.groupby(['kernel_subset_size', 'bandwidth_subset_size'])['metric_score'] \
            .agg(['max']).unstack(level=-1)

        combined_table = pd.DataFrame(index=table.index, columns=table.columns, dtype=object)
        for k_size in combined_table.index:
            for b_size in combined_table.columns:
                x = table.loc[k_size, b_size]
                y = detailed_table.loc[k_size, b_size]
                if pd.isna(x) or pd.isna(y):
                    combined_table.loc[k_size, b_size] = 'NA'
                else:
                    combined_table.loc[k_size, b_size] = f"{int(x)} ({int(y)})"

        def _y(cell):
            return int(cell.split('(')[-1][:-1]) if isinstance(cell, str) and cell != 'NA' else 0

        combined_table.loc['Total'] = combined_table.map(_y).sum(axis=0)
        combined_table['Total'] = combined_table.map(
            lambda s: _y(s) if isinstance(s, str) else s
        ).sum(axis=1)

        if verbose:
            print(combined_table.to_markdown())

        max_table = (
            per_subset.groupby(['kernel_subset_size', 'bandwidth_subset_size'])['metric_score']
            .max().unstack(level=-1) / len(Rq3Eval.METRICS)
        ).round(2)

        if verbose:
            print(max_table.to_markdown())

        return table, detailed_table, combined_table, max_table

    @staticmethod
    def _metric_score_matrix(combined_table):
        body = combined_table.drop(index='Total', errors='ignore')
        score_columns = [
            column for column in body.columns
            if (column[0] if isinstance(column, tuple) else column) != 'Total'
        ]
        metric_scores = pd.DataFrame(index=body.index, dtype=float)

        for column in score_columns:
            bandwidth_size = column[-1] if isinstance(column, tuple) else column
            bandwidth_size_int = int(str(bandwidth_size))
            values = []
            for kernel_size in body.index:
                match = re.fullmatch(
                    r'\s*(\d+)\s*\(\s*(\d+)\s*\)\s*',
                    str(body.loc[kernel_size, column]),
                )
                if match is None:
                    raise ValueError(
                        f"Cell ({kernel_size!r}, {bandwidth_size_int!r}) must use "
                        "the 'x (y)' format"
                    )
                values.append(int(match.group(2)))
            metric_scores[bandwidth_size_int] = values

        metric_scores.index = pd.Index(
            [int(size) for size in metric_scores.index],
            name='kernel_subset_size',
        )
        return metric_scores.reindex(sorted(metric_scores.columns), axis=1).sort_index()

    @staticmethod
    def _plot_trend_panel(ax, average_improvements, x_values, x_label,
                           legend_title, panel_label, panel_title):

        score_totals = (average_improvements * len(Rq3Eval.METRICS)).sum(axis=1)
        descending_order = score_totals.sort_values(ascending=False, kind='stable').index
        subset_averages = average_improvements.mean(axis=1).reindex(descending_order)
        strongest = subset_averages.index[:3].tolist()
        weakest = subset_averages.index[-2:].tolist()
        highlighted = strongest + weakest
        highlight_colors = dict(zip(highlighted, Rq3Eval.HIGHLIGHT_COLORS))

        for size in average_improvements.index:
            if size in highlighted:
                continue
            ax.plot(x_values, average_improvements.loc[size], color='#c7c7c7',
                     linestyle='--', linewidth=2, alpha=0.8, zorder=1)
            ax.scatter(x_values, average_improvements.loc[size], s=48, color='#c7c7c7',
                        edgecolor='white', linewidth=0.6, alpha=0.85, zorder=2)

        line_handles = []
        for size in highlighted:
            color = highlight_colors[size]
            is_weak = size in weakest
            line, = ax.plot(
                x_values, average_improvements.loc[size], color=color,
                linewidth=3.0 if not is_weak else 2.0,
                linestyle='--', label=str(size), zorder=3,
            )
            ax.scatter(x_values, average_improvements.loc[size], s=48, color=color,
                        edgecolor='white', linewidth=0.9, zorder=4)
            line_handles.append(line)

        else_handle, = ax.plot(
            [], [], color='#c7c7c7', linestyle='--', linewidth=2, label='else', zorder=1,
        )
        line_handles.append(else_handle)

        ax.set_title(panel_title, loc='center', fontsize=13.5, fontweight='bold', pad=12,)
        ax.set_xlabel(x_label, fontsize=12, labelpad=8)
        ax.set_ylabel('Mean joint improvement count', fontsize=12, labelpad=10)
        ax.set_xticks(x_values)
        ax.set_ylim(5.5, 14)
        ax.set_yticks(np.arange(7, 15, 1))
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(axis='y', color='#e5e5e5', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        break_style = {'transform': ax.transAxes, 'color': '#333333',
                       'clip_on': False, 'linewidth': 1.2}
        ax.plot((-0.012, 0.012), (-0.018, 0.018), **break_style)
        ax.plot((-0.012, 0.012), (0.018, 0.054), **break_style)
        ax.legend(handles=line_handles, title=legend_title, loc='lower left',
                  ncol=2, frameon=False, fontsize=13, title_fontsize=14)


    @staticmethod
    def _draw_joint_heatmap(ax, combined_table, cmap='PuBu', annotate=True,
                             title='(b) Optimal (kernel, bandwidth)\nsubset-size region',
                             highlight_threshold=12.3):

        metric_scores = Rq3Eval._metric_score_matrix(combined_table)
        average_improvements = metric_scores / len(Rq3Eval.METRICS)
        kernel_sizes = average_improvements.index.to_numpy()
        bandwidth_sizes = average_improvements.columns.to_numpy()
        values = average_improvements.to_numpy()

        best_value = np.nanmax(values)
        best_cells = list(zip(*np.where(values == best_value)))

        mesh = ax.imshow(
            values, cmap=cmap, aspect='auto', origin='lower',
            vmin=np.nanmin(values), vmax=np.nanmax(values),
        )
        ax.set_xticks(np.arange(len(bandwidth_sizes)))
        ax.set_xticklabels(bandwidth_sizes, fontsize=12)
        ax.set_yticks(np.arange(len(kernel_sizes)))
        ax.set_yticklabels(kernel_sizes, fontsize=12)
        ax.set_xlabel('Bandwidth subset size', fontsize=12, labelpad=8)
        ax.set_ylabel('Kernel subset size', fontsize=12, labelpad=8)
        ax.set_title(title, loc='center', fontsize=14, fontweight='bold', pad=12)

        if annotate:
            threshold = np.nanmin(values) + (np.nanmax(values) - np.nanmin(values)) * 0.6
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    if np.isnan(values[i, j]):
                        continue
                    ax.text(
                        j, i, f'{values[i, j]:.1f}', ha='center', va='center',
                        fontsize=11.5,
                        color='#1a1a1a' if values[i, j] < threshold else 'white',
                    )

        mask = np.nan_to_num(values, nan=-np.inf) >= highlight_threshold
        if mask.any():
            scale = 20
            big = np.kron(mask.astype(float), np.ones((scale, scale)))
            big = np.pad(big, scale, mode='constant')
            xs = (np.arange(big.shape[1]) - scale + 0.5) / scale - 0.5
            ys = (np.arange(big.shape[0]) - scale + 0.5) / scale - 0.5
            ax.contour(xs, ys, big, levels=[0.5], colors='#D55E00',
                       linewidths=2.5, zorder=5)
            ax.set_xlim(-0.5, values.shape[1] - 0.5)
            ax.set_ylim(-0.5, values.shape[0] - 0.5)

        cbar = ax.figure.colorbar(mesh, ax=ax, location='bottom', pad=0.03, shrink=0.9)
        cbar.set_label('Mean joint improvement count', fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        best_kernel_sizes = sorted({int(kernel_sizes[i]) for i, _ in best_cells})
        best_bandwidth_sizes = sorted({int(bandwidth_sizes[j]) for _, j in best_cells})

        return {
            'average_improvements': average_improvements,
            'best_value': float(best_value),
            'best_kernel_sizes': best_kernel_sizes,
            'best_bandwidth_sizes': best_bandwidth_sizes,
        }

    @staticmethod
    def plot_performance_and_joint_region(combined_table, output_path=None, show=False, cmap='YlGnBu'):

        metric_scores = Rq3Eval._metric_score_matrix(combined_table)
        average_improvements = metric_scores / len(Rq3Eval.METRICS)
        bandwidth_sizes = np.asarray(average_improvements.columns, dtype=int)

        fig, (ax_trend, ax_heat) = plt.subplots(
            1, 2, figsize=(11, 6), gridspec_kw={'width_ratios': (1, 1)},
            constrained_layout=True,
        )
        fig.patch.set_facecolor('white')

        Rq3Eval._plot_trend_panel(
            ax_trend, average_improvements, bandwidth_sizes, 'Bandwidth subset size',
            'Kernel subset size', 'a',
            '(a) Highest achievable ensemble\nimprovement count by subset size',
        )
        summary = Rq3Eval._draw_joint_heatmap(ax_heat, combined_table, cmap=cmap)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        return fig, (ax_trend, ax_heat), summary

    KERNEL_SIZE_BUCKETS = (3, 4, 5, None)
    KERNEL_SIZE_BUCKET_COLORS = ('#0072B2', '#009E73', '#D55E00', '#8C564B')

    @staticmethod
    def _kernel_size_bucket_label(bucket):
        return f'{bucket}-subset' if bucket is not None else 'else'


class Rq3_2Eval:

    KERNEL_ORDER = Configurations.KERNELS

    @staticmethod
    def _kernel_label(kernel_subset):
        if isinstance(kernel_subset, str):
            kernel_subset = (kernel_subset,)
        order = {k: i for i, k in enumerate(Rq3_2Eval.KERNEL_ORDER)}
        kernel_subset = sorted(kernel_subset, key=lambda k: order.get(k, len(order)))
        return '-'.join(kernel_subset)

    @staticmethod
    def _bandwidth_label(bandwidth_subset):
        if isinstance(bandwidth_subset, (int, float)):
            bandwidth_subset = (bandwidth_subset,)
        return '-'.join(str(int(round(b * 10))) for b in bandwidth_subset)

    @staticmethod
    def compare(kernel_subsets, bandwidth_subsets,
                ensemble_path=Configurations.EXP3_FILE,
                verbose=True):

        per_subset = Rq3Eval._per_subset_scores(ensemble_path)
        lookup = per_subset.set_index(['kernel_subset', 'bandwidth_subset'])[['mask', 'metric_score']]

        k_names = list(kernel_subsets.keys())
        b_names = list(bandwidth_subsets.keys())

        table = pd.DataFrame(index=k_names, columns=b_names, dtype=object)
        average_table = pd.DataFrame(index=k_names, columns=b_names, dtype=object)
        for k_name in k_names:
            k_label = Rq3_2Eval._kernel_label(kernel_subsets[k_name])
            for b_name in b_names:
                b_label = Rq3_2Eval._bandwidth_label(bandwidth_subsets[b_name])
                if (k_label, b_label) in lookup.index:
                    row = lookup.loc[(k_label, b_label)]
                    table.loc[k_name, b_name] = f"{int(row['mask'])} ({int(row['metric_score'])})"
                    average_table.loc[k_name, b_name] = round(row['metric_score'] / len(Rq3Eval.METRICS), 2)
                else:
                    table.loc[k_name, b_name] = 'NA'
                    average_table.loc[k_name, b_name] = 'NA'

        def _score(s):
            if isinstance(s, str):
                return 0 if s == 'NA' else int(s.split('(')[-1][:-1])
            return s

        table.loc['Total'] = table.map(_score).sum(axis=0)
        table['Total'] = table.map(_score).sum(axis=1)

        if verbose:
            pd.set_option('display.width', 200)
            pd.set_option('display.max_columns', None)
            print(table)
            print(average_table)

        return table, average_table


    @staticmethod
    def _row_family(name):
        if name == 'All' or '_' not in name:
            return name
        return name.rsplit('_', 1)[0]

    DEFAULT_FONTS = {
        'xtick': 24,
        'xtick_linespacing': 0.8,
        'ytick': 24,
        'xlabel': 27,
        'ylabel': 27,
        'title': 29,
        'cell': 23,
        'cbar_label': 27,
        'cbar_tick': 24,
    }

    @staticmethod
    def _draw_heatmap_only(ax_heat, values, k_names, b_names, cmap, norm,
                            panel_title, value_fmt, best='max',
                            show_row_labels=True, aspect='auto',
                            fonts=None, col_separators=None, highlight_threshold=None):

        fonts = {**Rq3_2Eval.DEFAULT_FONTS, **(fonts or {})}

        row_mean = np.nanmean(values, axis=1)
        col_mean = np.nanmean(values, axis=0)

        best_value = np.nanmax(values) if best == 'max' else np.nanmin(values)

        mesh = ax_heat.imshow(values, cmap=cmap, norm=norm, aspect=aspect, origin='upper')
        ax_heat.set_xticks(np.arange(len(b_names)))
        ax_heat.set_xticklabels([b.replace('+', '\n+\n') for b in b_names],
                                fontsize=fonts['xtick'], rotation=0, ha='center',
                                linespacing=fonts['xtick_linespacing'])
        ax_heat.set_yticks(np.arange(len(k_names)))
        ax_heat.set_yticklabels(k_names if show_row_labels else [], fontsize=fonts['ytick'])
        ax_heat.set_xlabel('Bandwidth subset', fontsize=fonts['xlabel'], labelpad=18)
        if show_row_labels:
            ax_heat.set_ylabel('Kernel subset', fontsize=fonts['ylabel'], labelpad=18)
        ax_heat.set_title(panel_title, loc='center', fontsize=fonts['title'], fontweight='bold', pad=30)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isnan(values[i, j]):
                    continue
                r, g, b, _ = mesh.cmap(mesh.norm(values[i, j]))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                ax_heat.text(
                    j, i, value_fmt(values[i, j]), ha='center', va='center',
                    fontsize=fonts['cell'], color='#1a1a1a' if luminance > 0.6 else 'white',
                )

        families = [Rq3_2Eval._row_family(n) for n in k_names]
        for i in range(1, len(families)):
            if families[i] != families[i - 1]:
                ax_heat.axhline(i - 0.5, color='#000000', linewidth=2, linestyle='--')

        for name in (col_separators or []):
            if name in b_names:
                ax_heat.axvline(b_names.index(name) + 0.5, color='#000000',
                                linewidth=2, linestyle='--')

        if highlight_threshold is not None:
            mask = np.nan_to_num(values, nan=-np.inf) >= highlight_threshold
            if mask.any():
                scale = 20
                big = np.kron(mask.astype(float), np.ones((scale, scale)))
                big = np.pad(big, scale, mode='constant')
                xs = (np.arange(big.shape[1]) - scale + 0.5) / scale - 0.5
                ys = (np.arange(big.shape[0]) - scale + 0.5) / scale - 0.5
                ax_heat.contour(xs, ys, big, levels=[0.5], colors='#D55E00',
                                linewidths=4, zorder=5)
                ax_heat.set_xlim(-0.5, values.shape[1] - 0.5)
                ax_heat.set_ylim(values.shape[0] - 0.5, -0.5)

        summary = {
            'best_value': float(best_value),
            'row_mean': pd.Series(row_mean, index=k_names),
            'col_mean': pd.Series(col_mean, index=b_names),
        }
        return mesh, summary


    @staticmethod
    def plot_joint_heatmap(kernel_subsets, bandwidth_subsets,
                            ensemble_path=Configurations.EXP3_FILE,
                            cmap='YlGnBu', output_path=None, show=False,
                            fonts=None,
                            col_separators=Configurations.NAMED_COL_SEPARATORS,
                            highlight_threshold=12):

        _, average_table = Rq3_2Eval.compare(
            kernel_subsets, bandwidth_subsets,
            ensemble_path=ensemble_path, verbose=False,
        )
        average_table = average_table.drop(index='Total', errors='ignore') \
                                       .drop(columns='Total', errors='ignore')
        values = average_table.apply(pd.to_numeric, errors='coerce').to_numpy()

        k_names = list(average_table.index)
        b_names = list(average_table.columns)

        fig = plt.figure(figsize=(16, 10), layout='constrained')
        fig.patch.set_facecolor('white')
        ax_heat = fig.add_subplot(111)

        norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
        mesh, summary = Rq3_2Eval._draw_heatmap_only(
            ax_heat, values, k_names, b_names, cmap=cmap, norm=norm,
            panel_title='Optimal (named kernel subset, named bandwidth subset) region',
            value_fmt=lambda v: f'{v:.1f}', best='max',
            fonts=fonts, col_separators=col_separators,
            highlight_threshold=highlight_threshold,
        )

        fonts = {**Rq3_2Eval.DEFAULT_FONTS, **(fonts or {})}
        cbar = fig.colorbar(mesh, ax=ax_heat, location='right', pad=0.075, shrink=0.85)
        cbar.set_label('Mean joint improvement count', fontsize=fonts['cbar_label'])
        cbar.ax.tick_params(labelsize=fonts['cbar_tick'])

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        summary['average_table'] = average_table
        return fig, summary
