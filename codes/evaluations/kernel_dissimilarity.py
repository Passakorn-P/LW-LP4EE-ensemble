# =============================================================================
# evaluations/kernel_dissimilarity.py
# -----------------------------------------------------------------------------
# Project  : LW-LP4EE Ensemble: A Practical Approach to Locally Weighted
#            Linear Programming for Software Effort Estimation
# Purpose  : Kernel analysis that backs the kernel subsets of RQ3.
# =============================================================================

import itertools
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from configs import Configurations
from models.lw_lp4ee import LW_LP4EE


class KernelDissimilarity:

    SIMILAR_3 = ('epanechnikov', 'biweight', 'triweight')
    SIZES = range(3, 7)
    TIE_TOL = 1e-9
    EQUATIONS = {
        'rectangular': r'$K(h)=\frac{1}{2},\ 0\leq h<1$',
        'epanechnikov': r'$K(h)=\frac{3}{4}(1-h^2),\ 0\leq h<1$',
        'tricube': r'$K(h)=\frac{70}{81}(1-h^3)^3,\ 0\leq h<1$',
        'gaussian': r'$K(h)=\frac{1}{\sqrt{2\pi}}e^{-h^2/2}$',
        'triangle': r'$K(h)=1-h,\ 0\leq h<1$',
        'triweight': r'$K(h)=\frac{35}{32}(1-h^2)^3,\ 0\leq h<1$',
        'biweight': r'$K(h)=\frac{15}{16}(1-h^2)^2,\ 0\leq h<1$',
        'cosine': r'$K(h)=\frac{\pi}{4}\cos\!\left(\frac{\pi h}{2}\right),\ 0\leq h<1$',
        'logistic': r'$K(h)=\frac{1}{e^h+2+e^{-h}}$',
        'sigmoid': r'$K(h)=\frac{2}{\pi(e^h+e^{-h})}$',
    }

    @staticmethod
    def kernel(name, h):
        a = abs(h)
        inside = a < 1.0
        if name == 'rectangular':
            return 0.5 if inside else 0.0
        if name == 'epanechnikov':
            return 0.75 * (1 - a ** 2) if inside else 0.0
        if name == 'tricube':
            return (70 / 81) * (1 - a ** 3) ** 3 if inside else 0.0
        if name == 'gaussian':
            return math.exp(-0.5 * a ** 2) / math.sqrt(2 * math.pi)
        if name == 'triangle':
            return 1 - a if inside else 0.0
        if name == 'triweight':
            return (35 / 32) * (1 - a ** 2) ** 3 if inside else 0.0
        if name == 'biweight':
            return (15 / 16) * (1 - a ** 2) ** 2 if inside else 0.0
        if name == 'cosine':
            return (math.pi / 4) * math.cos(math.pi / 2 * a) if inside else 0.0
        if name == 'logistic':
            return 0.0 if a > 700 else 1 / (math.exp(a) + 2 + math.exp(-a))
        if name == 'sigmoid':
            return 0.0 if a > 700 else (2 / math.pi) / (math.exp(a) + math.exp(-a))
        raise ValueError(name)

    @staticmethod
    def half_integral(f, upper, points=()):
        kw = dict(limit=500, epsabs=1e-13, epsrel=1e-11)
        if math.isinf(upper):
            split = max([1.0, *points])
            inner = quad(f, 0, split, points=[p for p in points if 0 < p < split] or None, **kw)[0]
            return 2 * (inner + quad(f, split, np.inf, **kw)[0])
        return 2 * quad(f, 0, upper, points=[p for p in points if 0 < p < upper] or None, **kw)[0]

    @staticmethod
    def scale_factors():
        scale = {}
        for name in Configurations.KERNELS:
            upper = 1.0 if name in Configurations.COMPACT_KERNELS else np.inf
            m0 = KernelDissimilarity.half_integral(lambda h: KernelDissimilarity.kernel(name, h), upper)
            m2 = KernelDissimilarity.half_integral(lambda h: h * h * KernelDissimilarity.kernel(name, h), upper)
            scale[name] = math.sqrt(m2 / m0)
        return scale

    @staticmethod
    def l2_dissimilarity(p, q, scale):
        def rescaled(name, h):
            return scale[name] * KernelDissimilarity.kernel(name, scale[name] * h)

        def support_end(name):
            return 1 / scale[name] if name in Configurations.COMPACT_KERNELS else math.inf

        ends = [support_end(k) for k in (p, q) if k in Configurations.COMPACT_KERNELS]
        upper = math.inf if math.isinf(support_end(p)) or math.isinf(support_end(q)) else max(ends)
        f = lambda h: (rescaled(p, h) - rescaled(q, h)) ** 2
        return math.sqrt(KernelDissimilarity.half_integral(f, upper, points=ends))

    @staticmethod
    def pairwise(scale):
        d = {}
        for p, q in itertools.combinations(Configurations.KERNELS, 2):
            d[p, q] = d[q, p] = KernelDissimilarity.l2_dissimilarity(p, q, scale)
        return d

    @staticmethod
    def most_distinct(d, m):
        best, best_val, n_ties = None, -1.0, 0
        for combo in itertools.combinations(Configurations.KERNELS, m):
            v = min(d[p, q] for p, q in itertools.combinations(combo, 2))
            if v > best_val + KernelDissimilarity.TIE_TOL:
                best, best_val, n_ties = combo, v, 1
            elif abs(v - best_val) <= KernelDissimilarity.TIE_TOL:
                n_ties += 1
        return best, best_val, n_ties

    @staticmethod
    def report():
        scale = KernelDissimilarity.scale_factors()
        d = KernelDissimilarity.pairwise(scale)
        kernels = Configurations.KERNELS
        width = max(map(len, kernels))

        lines = ['Dissimilarity of rescaled kernels', ' ' * width + ' ' + ' '.join(k[:6].rjust(7) for k in kernels)]
        for p in kernels:
            lines.append(p.ljust(width) + ' ' + ' '.join(('-' if p == q else f'{d[p, q]:.3f}').rjust(7) for q in kernels))
        for m in KernelDissimilarity.SIZES:
            combo, val, ties = KernelDissimilarity.most_distinct(d, m)
            tie_note = 'unique' if ties == 1 else f'{ties} tied optima'
            lines.append(f'size {m}: {", ".join(combo)}  min={val:.3f}  ({tie_note})')
        pairs = {f'{p}-{q}': round(d[p, q], 3) for p, q in itertools.combinations(KernelDissimilarity.SIMILAR_3, 2)}
        lines.append(f'Similar_3 pairwise: {pairs}')

        text = '\n'.join(lines)
        os.makedirs(Configurations.TABLE_PATH, exist_ok=True)
        with open(f'{Configurations.TABLE_PATH}kernel_dissimilarity.txt', 'w') as f:
            f.write(text + '\n')
        return d

    @staticmethod
    def plot_grid(save_path=None):
        save_path = f'{Configurations.PLOT_PATH}related_work_kernels_grid.eps' if save_path is None else save_path
        h = np.linspace(0.0, 3.0, 1201)
        fig, axes = plt.subplots(2, 5, figsize=(12.0, 5.0), sharex=True, sharey=True)

        for ax, name in zip(axes.flat, Configurations.KERNELS):
            weights = LW_LP4EE.kernel_weights(name, h, 1.0)
            ax.plot(h, weights, color='#1f4e79', linewidth=1.8)
            ax.fill_between(h, 0, weights, color='#e9edf2', alpha=0.10)
            if name in Configurations.COMPACT_KERNELS:
                ax.axvline(1.0, color='#777777', linewidth=0.8, linestyle='--')
            ax.set_title(f'{name}\n{KernelDissimilarity.EQUATIONS[name]}', fontsize=11, pad=6)
            ax.set_xlim(0.0, 3.0)
            ax.set_ylim(0.0, 1.15)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(labelsize=11)
            ax.grid(axis='y', color='#dddddd', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.supxlabel(r'Scaled distance, $h$', fontsize=13)
        fig.supylabel(r'Raw kernel weight, $K(h)$', fontsize=13)
        fig.tight_layout(pad=1.0, w_pad=0.8, h_pad=1.2)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path
