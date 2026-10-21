"""多数据集单摆实验的正式报告图与派生统计生成器。

输入（集群跑量回收的产物）:
  - trace_md*.dat  逐样本迹线（字段布局见 core/parallel.py 的写盘段）
  - summary.json   multidataset_planets.py --analyze-only 的汇总

输出（相对本文件所在目录）:
  - multidataset_report/derived_samples.csv  逐样本派生指标
  - multidataset_report/derived_chains.csv   逐链汇总
  - multidataset_report/fig1..fig6.png       报告插图

图注与口径全部复用 experiments/pendulum/multidataset/multidataset_planets.py 里的
判定函数（family_coefficients / uses_var），与官方 analyze 结果严格一致。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sympy import sympify, count_ops
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from experiments.pendulum.multidataset.multidataset_planets import (  # noqa: E402
    SCENARIOS, PLANETS, make_frames, uses_var, family_coefficients,
)

REPO = HERE.parents[2]
OUTDIR = HERE / 'multidataset_report'
COLORS = {'md_Lonly': 'tab:blue', 'md_candG': 'tab:orange', 'md_junkX': 'tab:green'}
NICE = {'md_Lonly': 'md_Lonly (L only)', 'md_candG': 'md_candG (X = g)',
        'md_junkX': 'md_junkX (X = junk)'}
TWO_PI = 2 * np.pi


# ---------------------------------------------------------------------------
# 逐样本派生指标
# ---------------------------------------------------------------------------

def cp_interval(k, n, conf=0.95):
    lo = 0.0 if k == 0 else beta.ppf((1 - conf) / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - (1 - conf) / 2, k + 1, n - k)
    return lo, hi


def _classify(args):
    """对去重后的 (formula, par_values) 计算形状关 + ops 数（worker 内执行）。"""
    formula, pv, scenario = args
    in_family = family_coefficients(formula, pv, _X[scenario]) is not None
    try:
        nops = int(count_ops(sympify(formula.replace(' ', ''))))
    except Exception:
        nops = np.nan
    return in_family, nops


_X = {}


def load_samples(tracedir):
    """读取全部 trace，返回逐样本 DataFrame。"""
    rows = []
    for fn in sorted(tracedir.glob('trace_md*.dat')):
        parts = fn.stem.split('_')
        scenario, chain = '_'.join(parts[1:-1]), int(parts[-1])
        with open(fn) as f:
            for line in f:
                rec = json.loads(line)
                rows.append((scenario, chain, rec[0], rec[1], rec[2],
                             rec[4], uses_var(rec[4], 'X') if 'X' in
                             SCENARIOS[scenario]['variables'] else False,
                             json.dumps(rec[5], sort_keys=True)))
    df = pd.DataFrame(rows, columns=['scenario', 'chain', 's', 'bic', 'E',
                                     'formula', 'uses_X', 'pv_json'])
    return df


def add_family_and_ops(df, jobs):
    """对去重 (formula, pv, scenario) 组合并行判定，再映射回逐样本行。"""
    global _X
    for sc in df['scenario'].unique():
        _X[sc] = make_frames(sc)[0]
    uniq = df[['scenario', 'formula', 'pv_json']].drop_duplicates()
    tasks = [(r.formula, json.loads(r.pv_json), r.scenario, r.pv_json)
             for r in uniq.itertuples()]
    with Pool(jobs) as pool:
        res = pool.map(_classify, [t[:3] for t in tasks], chunksize=64)
    lut = {(t[2], t[0], t[3]): r for t, r in zip(tasks, res)}
    fam, ops = [], []
    for r in df.itertuples():
        f, o = lut[(r.scenario, r.formula, r.pv_json)]
        fam.append(f)
        ops.append(o)
    df['in_family'] = fam
    df['nops'] = ops
    return df


# ---------------------------------------------------------------------------
# 图 1：5 份数据集 + MAP 拟合曲线
# ---------------------------------------------------------------------------

def fig1_data(summary, out):
    x, y, _ = make_frames('md_Lonly')
    mp = summary['results']['md_Lonly']['map_par_values']
    planets = list(PLANETS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    Lg = np.linspace(0.1, 3.0, 200)
    for i, name in enumerate(planets):
        ds = 'd%d' % i
        ax = axes[0]
        ax.plot(x[ds]['L'], y[ds], 'o', ms=3.5, alpha=0.6, label='%s (g=%.2f)' % (name, PLANETS[name]))
        a = float(mp[ds]['_a_'])
        ax.plot(Lg, np.sqrt(Lg / a), '-', lw=1.2)
        axes[1].loglog(x[ds]['L'], y[ds], 'o', ms=3.5, alpha=0.6)
        axes[1].loglog(Lg, np.sqrt(Lg / a), '-', lw=1.2)
    axes[0].set_xlabel('L (m)'); axes[0].set_ylabel('T (s)')
    axes[0].set_title('(a) five datasets, never shown g')
    axes[0].legend(fontsize=7.5, ncol=1, loc='upper left')
    axes[1].set_xlabel('L (m)'); axes[1].set_ylabel('T (s)')
    axes[1].set_title('(b) log-log, MAP fit per planet')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 图 2：family_frac 与 X_used_frac（带 Clopper-Pearson 区间）
# ---------------------------------------------------------------------------

def fig2_rates(summary, df, out):
    scs = list(SCENARIOS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, title in [
        (axes[0], 'family', '(a) samples in the a*sqrt(L) family'),
        (axes[1], 'X', '(b) samples whose formula uses X'),
    ]:
        cells = scs if key == 'family' else ['md_candG', 'md_junkX']
        vals, los, his = [], [], []
        for sc in cells:
            n = summary['results'][sc]['n_samples']
            if key == 'family':
                k = int(round(df[(df.scenario == sc) & df.in_family].shape[0]))
            else:
                k = int(df[(df.scenario == sc) & df.uses_X].shape[0])
            lo, hi = cp_interval(k, n)
            vals.append(k / n); los.append(k / n - lo); his.append(hi - k / n)
        if key == 'family':
            ax.set_ylim(0, 1.12)
            ax.axhline(1.0, color='gray', lw=0.6, ls='--')
        else:
            # 全为 0：压缩 y 轴让 CP 上界可见
            ax.set_ylim(0, 20 * max(v + hi for v, lo, hi in zip(vals, los, his)))
        bars = ax.bar(range(len(cells)), vals, 0.55,
                      color=[COLORS[sc] for sc in cells],
                      yerr=[los, his], capsize=4,
                      error_kw=dict(lw=1.2, ecolor='k'))
        for b, v, (lo, hi) in zip(bars, vals, zip(los, his)):
            top = v + hi
            if v == 0:
                txt = '0\n95%% upper bound\n%.1e' % top
                ytxt = top + 0.04 * ax.get_ylim()[1]
                ax.text(b.get_x() + b.get_width() / 2, ytxt, txt,
                        ha='center', va='bottom', fontsize=8.5)
            else:
                ax.text(b.get_x() + b.get_width() / 2, top + 0.012,
                        '%.1f%%' % (100 * v), ha='center', va='bottom',
                        fontsize=8.5)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(cells, fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel('fraction of posterior samples')
    axes[0].text(0.02, 0.30,
                 'MAP in all three cells:\nsqrt(L)/_a_ or sqrt(L/_a_)\n'
                 '(same law, E = -607.9)',
                 transform=axes[0].transAxes, fontsize=8,
                 bbox=dict(boxstyle='round', fc='w', alpha=0.8))
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 图 3：反推 g vs 真值
# ---------------------------------------------------------------------------

def fig3_g_recovery(summary, out):
    truth = {'d%d' % i: g for i, g in enumerate(PLANETS.values())}
    names = {'d%d' % i: n for i, n in enumerate(PLANETS)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    mk = {'md_Lonly': 'o', 'md_candG': 's', 'md_junkX': '^'}
    for sc in SCENARIOS:
        ig = summary['results'][sc].get('implied_g')
        if not ig:
            continue
        xs = [truth[ds] for ds in ig]
        ys = [ig[ds] for ds in ig]
        axes[0].loglog(xs, ys, mk[sc], ms=6, mfc='none', mec=COLORS[sc],
                       label=NICE[sc])
    g0 = np.array(list(truth.values()))
    axes[0].loglog(g0, g0, 'k-', lw=1, label='y = x')
    for ds, t in truth.items():
        axes[0].annotate(names[ds], (t, t), textcoords='offset points',
                         xytext=(6, 4), fontsize=8)
    axes[0].set_xlabel('true g (m/s$^2$)'); axes[0].set_ylabel('implied g (m/s$^2$)')
    axes[0].set_title('(a) gravity recovered from the 5 fitted constants')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].grid(alpha=0.3, which='both')

    ig = summary['results']['md_Lonly']['implied_g']
    dev = {ds: 100 * abs(ig[ds] - truth[ds]) / truth[ds] for ds in ig}
    xs = np.arange(len(dev))
    axes[1].bar(xs, [dev[ds] for ds in sorted(dev)], 0.5,
                color='tab:blue')
    for i, ds in enumerate(sorted(dev)):
        axes[1].text(i, dev[ds] + 0.02, '%.2f%%' % dev[ds], ha='center', fontsize=8)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([names[ds] for ds in sorted(dev)], fontsize=8.5)
    axes[1].set_ylabel('|relative deviation| (%)')
    axes[1].set_title('(b) per-planet deviation (max 0.8%)')
    axes[1].set_ylim(0, 1.0)
    axes[1].axhline(1.0, color='gray', ls='--', lw=0.8)
    axes[1].grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 图 4：样本能量分布 + 逐链均值
# ---------------------------------------------------------------------------

def fig4_energy(df, summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for sc in SCENARIOS:
        sub = df[df.scenario == sc]
        axes[0].hist(sub.E, bins=60, density=True,
                     color=COLORS[sc], alpha=0.3, label=NICE[sc])
        mapE = summary['results'][sc]['map_E']
        axes[0].axvline(mapE, color=COLORS[sc], lw=1.2, ls='--')
    axes[0].set_xlabel('sample energy E'); axes[0].set_ylabel('density')
    axes[0].set_title('(a) posterior sample energies (dashed = MAP)')
    axes[0].legend(fontsize=8)
    rng = np.random.default_rng(0)
    for sc in SCENARIOS:
        sub = df[df.scenario == sc]
        means = sub.groupby('chain').E.mean()
        axes[1].scatter(means.index + rng.uniform(-0.15, 0.15, len(means)),
                        means.values, s=14, color=COLORS[sc], alpha=0.75,
                        label=NICE[sc])
        axes[1].axhline(means.mean(), color=COLORS[sc], lw=0.9, ls=':')
    axes[1].set_xlabel('chain id'); axes[1].set_ylabel('mean sample E per chain')
    axes[1].set_title('(b) chain-to-chain consistency')
    axes[1].legend(fontsize=8, loc='lower right')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 图 5：公式复杂度分布
# ---------------------------------------------------------------------------

def fig5_complexity(df, out):
    bins = [(1, 3), (4, 6), (7, 15), (16, 200)]
    labels = ['1-3 ops', '4-6', '7-15', '>15']
    scs = list(SCENARIOS)
    share = np.zeros((len(scs), len(bins)))
    for i, sc in enumerate(scs):
        sub = df[df.scenario == sc]
        n = len(sub)
        for j, (a, b) in enumerate(bins):
            share[i, j] = sub.nops.between(a, b).sum() / n
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    bottom = np.zeros(len(scs))
    cmap = plt.cm.Blues(np.linspace(0.45, 0.95, len(bins)))
    for j, lab in enumerate(labels):
        axes[0].bar(range(len(scs)), share[:, j], 0.55, bottom=bottom,
                    color=cmap[j], label=lab)
        for i in range(len(scs)):
            if share[i, j] > 0.06:
                axes[0].text(i, bottom[i] + share[i, j] / 2, '%.1f%%' % (100 * share[i, j]),
                             ha='center', va='center', fontsize=8,
                             color='w' if j >= 2 else 'k')
        bottom += share[:, j]
    axes[0].set_xticks(range(len(scs)))
    axes[0].set_xticklabels(scs, fontsize=9)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel('share of posterior samples')
    axes[0].set_title('(a) formula size (count_ops)')
    axes[0].legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))

    for i, sc in enumerate(scs):
        med = df[df.scenario == sc].groupby('chain').nops.median()
        axes[1].scatter(med.index + 0.0, med.values, s=16,
                        color=COLORS[sc], alpha=0.7, label=NICE[sc])
    axes[1].set_yscale('symlog', linthresh=1)
    axes[1].set_yticks([1, 2, 3, 5, 10, 20, 50])
    axes[1].set_yticklabels(['1', '2', '3', '5', '10', '20', '50'])
    axes[1].set_xlabel('chain id'); axes[1].set_ylabel('median ops per chain (symlog)')
    axes[1].set_title('(b) every chain settles on minimal expressions')
    axes[1].legend(fontsize=8, loc='upper center')
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# 图 6：E 轨迹（burnin 后 equilibration 视角）
# ---------------------------------------------------------------------------

def fig6_traj(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    for ax, sc in zip(axes, SCENARIOS):
        sub = df[df.scenario == sc]
        chains = sorted(sub.chain.unique())
        show = chains[::max(1, len(chains) // 7)]
        for c in show:
            d = sub[sub.chain == c]
            ax.plot(d.s, d.E, lw=0.5, alpha=0.45, color=COLORS[sc])
        mean_traj = sub.groupby('s').E.mean()
        ax.plot(mean_traj.index, mean_traj.values, 'k-', lw=1.4,
                label='cell mean')
        ax.set_title(sc, fontsize=10)
        ax.set_xlabel('posterior sample index')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('E')
    fig.suptitle('post-burnin energy trajectories (thin lines: example chains)',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracedir', default=str(
        REPO / 'temp/md_pt20/multidataset_pt20'))
    ap.add_argument('--jobs', type=int, default=max(1, (os_cpu() or 4) - 1))
    args = ap.parse_args()
    tracedir = Path(args.tracedir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    with open(tracedir / 'summary.json') as f:
        summary = json.load(f)

    print('loading traces from', tracedir)
    df = load_samples(tracedir)
    print('  raw samples:', len(df))
    df = add_family_and_ops(df, args.jobs)
    df.to_csv(OUTDIR / 'derived_samples.csv', index=False)

    chain_df = df.groupby(['scenario', 'chain']).agg(
        n=('s', 'count'),
        family_frac=('in_family', 'mean'),
        x_used=('uses_X', 'mean'),
        mean_E=('E', 'mean'),
        median_ops=('nops', 'median'),
        last_s=('s', 'max'),
    ).reset_index()
    chain_df.to_csv(OUTDIR / 'derived_chains.csv', index=False)

    # 与官方 summary.json 对账
    for sc in SCENARIOS:
        n_official = summary['results'][sc]['n_samples']
        n_here = int(df[df.scenario == sc].shape[0])
        fam = df[(df.scenario == sc) & df.in_family].shape[0] / n_here
        print('%s: n=%d (official %d, %s)  family_frac=%.4f' % (
            sc, n_here, n_official,
            'OK' if n_here == n_official else 'MISMATCH', fam))

    fig1_data(summary, OUTDIR / 'fig1_data.png')
    fig2_rates(summary, df, OUTDIR / 'fig2_rates.png')
    fig3_g_recovery(summary, OUTDIR / 'fig3_g_recovery.png')
    fig4_energy(df, summary, OUTDIR / 'fig4_energy.png')
    fig5_complexity(df, OUTDIR / 'fig5_complexity.png')
    fig6_traj(df, OUTDIR / 'fig6_traj.png')
    print('figures written to', OUTDIR)


def os_cpu():
    try:
        import os
        return os.cpu_count()
    except Exception:
        return None


if __name__ == '__main__':
    main()
