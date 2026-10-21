"""图 7：闸门 vs 记账——X 使用率的三组对照，以及发现 g 的能量账。

左图读两处 trace 目录（正式跑量与反事实试跑），右图是把 probe3/probe4 的能量
分解画成条形。数字与 multidataset_report.md 第 6 节逐位对应。
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / 'multidataset_report' / 'fig7_gate_vs_accounting.png'
OFFICIAL = REPO / 'temp' / 'md_pt20' / 'multidataset_pt20'
PILOT = HERE / 'multidataset_runs' / 'xdim_pilot'

XTOK = re.compile(r'(?<![A-Za-z_])X(?![A-Za-z_])')


def cp(k, n, conf=0.95):
    lo = 0.0 if k == 0 else beta.ppf((1 - conf) / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - (1 - conf) / 2, k + 1, n - k)
    return lo, hi


def x_rate(tracedir, scenario):
    n = k = 0
    for fn in sorted(Path(tracedir).glob('trace_%s_*.dat' % scenario)):
        with open(fn) as f:
            for line in f:
                rec = json.loads(line)
                n += 1
                k += bool(XTOK.search(rec[4]))
    return k, n


def main():
    rows = []
    for label, d, sc in [
            ('official\nX dim NOT declared\nX = g', OFFICIAL, 'md_candG'),
            ('official\nX dim NOT declared\nX = junk', OFFICIAL, 'md_junkX'),
            ('pilot\nX dim declared\nX = g', PILOT, 'md_candG'),
            ('pilot\nX dim declared\nX = junk', PILOT, 'md_junkX')]:
        k, n = x_rate(d, sc)
        lo, hi = cp(k, n)
        rows.append((label, k, n, 100 * k / max(n, 1), 100 * lo, 100 * hi))
        print('%-34s %5d/%6d = %6.3f%%  CP95 [%.3f, %.3f]'
              % (label.replace('\n', ' | '), k, n, 100 * k / max(n, 1),
                 100 * lo, 100 * hi))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ax = axes[0]
    xs = np.arange(len(rows))
    vals = [r[3] for r in rows]
    err = [[r[3] - r[4] for r in rows], [r[5] - r[3] for r in rows]]
    cols = ['tab:orange', 'tab:green', 'tab:orange', 'tab:green']
    hatches = ['///', '///', '', '']
    bars = ax.bar(xs, vals, yerr=err, capsize=4, color=cols, edgecolor='k',
                  linewidth=.6)
    for b, h, r in zip(bars, hatches, rows):
        b.set_hatch(h)
        ax.text(b.get_x() + b.get_width() / 2, r[5] + .22,
                '%d/%d' % (r[1], r[2]), ha='center', fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.6)
    ax.set_ylabel('samples whose formula uses X (%)')
    ax.set_ylim(0, 7.6)
    ax.set_title('(a) The gate, not the objective function, made it exactly 0\n'
                 'hatched = official run (hard-blocked), solid = pilot '
                 '(one line added: dimensions["X"])', fontsize=9.5)
    ax.axvline(1.5, color='k', ls=':', lw=1)
    ax.text(0.5, 6.9, 'both exactly 0:\nthe gate, not the data',
            ha='center', fontsize=8.5)
    ax.text(2.5, 6.9, 'Fisher exact p = 0.869\n(g vs junk: no difference)',
            ha='center', fontsize=8.5)

    ax = axes[1]
    # probe3: X written into the formula, constants still fitted per dataset.
    # probe4: constants shared across datasets, honest pooled BIC accounting.
    labels = ['now:\nper-dataset fit\nk counts symbols',
              'with shared-constant fit\n+ honest k']
    dEB = [0.0, -8.075]
    dEP = [6.0022, 6.0022]
    xs = np.arange(len(labels))
    ax.bar(xs, dEB, width=.5, label=r'$\Delta E_B$ (data / BIC)',
           color='tab:blue', edgecolor='k', linewidth=.6)
    ax.bar(xs, dEP, width=.5, bottom=dEB, label=r'$\Delta E_P$ (prior)',
           color='tab:red', edgecolor='k', linewidth=.6)
    net = [a + b for a, b in zip(dEB, dEP)]
    for xi, nv in zip(xs, net):
        ax.plot([xi - .3, xi + .3], [nv, nv], 'k--', lw=1.2)
        ax.text(xi + .33, nv, 'net %+.2f nat\nodds 1:%.0f' % (nv, np.exp(nv))
                if nv > 0 else 'net %+.2f nat\nodds %.1f:1' % (nv, np.exp(-nv)),
                va='center', fontsize=8)
    ax.axhline(0, color='k', lw=.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r'$\Delta E$ of the X-using model vs MAP (nat)')
    ax.set_xlim(-.6, 2.4)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('(b) The reward for discovering g is worth 22.8 BIC nat,\n'
                 'but only a shared-constant fit can collect it', fontsize=9.5)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print('saved', OUT)


if __name__ == '__main__':
    main()
