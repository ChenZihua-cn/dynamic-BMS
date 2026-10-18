"""先验消融对照：读 pendulum_runs/<tag>/summary.json，出对照表与图。

用法::

    .venv/bin/python experiments/pendulum/prior_pendulum_compare.py \
        flat1.0 bms_wiki_nv13_np13 bms_wiki_nv13_np13_x0.5 ...

不传参数时自动收集 ``pendulum_runs`` 下所有含 summary.json 的目录，
按"先验力度"（真公式在该先验下的 EP，越小越弱）排序。

对照的核心问题（见 log/ops_complete_prior_design_2026-10-16.md §7-3）：
先验力度加大能压掉"怪物外衣"，但也可能压掉真公式本身。
因此同时看三条曲线：命中率、MAP 复杂度、常数精度。
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNROOT = HERE / 'pendulum_runs'
OUT_PNG = HERE / 'pendulum_prior_ablation.png'
OUT_MD = HERE / 'pendulum_prior_ablation.md'

TWO_PI = 2 * np.pi
SCEN = [('main', '5 planets, L,g', 'tab:blue'),
        ('ctrlB_Lg', 'Earth only, L,g', 'tab:orange'),
        ('ctrlA_Lonly', 'Earth only, L', 'tab:red')]

#: 展示名 -> 说明。未列出的直接用目录名。
LABELS = {
    'flat1.0': 'flat 1.0 (hand-set, 2026-10-16 run)',
    'bms_wiki_nv13_np13_x0.25': 'wiki x0.25',
    'bms_wiki_nv13_np13_x0.5': 'wiki x0.5',
    'bms_wiki_nv13_np13': 'wiki x1 (paper values)',
    'bms_wiki_nv13_np13_x2': 'wiki x2',
    'bms_wiki_nv13_np13_linear': 'wiki linear (no quadratic)',
    'flat_default': 'flat 10.0 (core 默认)',
    'flat_zero': 'flat 0.0 (纯 BIC)',
}


def load(tag):
    p = RUNROOT / tag / 'summary.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d['_tag'] = tag
    return d


def prior_strength(d):
    """先验力度的标量：真公式在该先验下的 EP（nat）。越小 = 越弱。"""
    ref = d.get('reference', {}).get('main')
    if ref is None:
        return float('nan')
    return float(ref['EP'])


def fmt(v, spec='{:.3f}'):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return '—'
        return spec.format(float(v))
    except Exception:
        return str(v)


def collect(tags=None):
    if not tags:
        tags = sorted([p.parent.name for p in RUNROOT.glob('*/summary.json')])
    runs = [load(t) for t in tags]
    runs = [r for r in runs if r is not None]
    runs.sort(key=prior_strength)
    return runs


def table_md(runs):
    lines = []
    for scen, scen_label, _ in SCEN:
        lines.append(f'\n### {scen_label}（{scen}）\n')
        lines.append('| 先验 | 真公式EP | 占有率 | 命中链 | c\\*（2π=6.2832） | '
                     'MAP ops | 最优BIC ops | 迹线ops均值 | 平均EP | mixing |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for d in runs:
            r = d.get('results', {}).get(scen)
            if r is None:
                continue
            c = r.get('c_median')
            c_txt = fmt(c, '{:.4f}') if c is not None and np.isfinite(c) else '—'
            lines.append(
                '| {label} | {ep} | {occ:.1%} | {hit}/{n} | {c} | {mapops} | '
                '{bicops} | {sz} | {epm} | {mv} |'.format(
                    label=LABELS.get(d['_tag'], d['_tag']),
                    ep=fmt(prior_strength(d), '{:.1f}'),
                    occ=r['occ_true'], hit=r['n_chains_hit'], n=r['n_chains'],
                    c=c_txt,
                    mapops=fmt(r.get('map_nops'), '{:.0f}'),
                    bicops=fmt(r.get('best_bic_nops'), '{:.0f}'),
                    sz=fmt(r.get('size_mean'), '{:.1f}'),
                    epm=fmt(r.get('ep_mean'), '{:.1f}'),
                    mv=fmt(r.get('move_frac'), '{:.2f}'),
                ))
    return '\n'.join(lines)


def make_figure(runs, out_path=OUT_PNG):
    tags = [d['_tag'] for d in runs]
    xs = np.arange(len(tags))
    strength = [prior_strength(d) for d in runs]
    short = [LABELS.get(t, t) for t in tags]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    ax = axes[0, 0]
    for scen, label, color in SCEN:
        ys = [d.get('results', {}).get(scen, {}).get('occ_true', np.nan)
              for d in runs]
        ax.plot(xs, np.array(ys) * 100., 'o-', color=color, label=label)
    ax.set_xticks(xs); ax.set_xticklabels(short, rotation=30, ha='right',
                                          fontsize=8)
    ax.set_ylabel('True-family occupancy (%)')
    ax.set_title('(a) Recovery vs prior strength')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for scen, label, color in SCEN:
        ys, hits = [], []
        for d in runs:
            r = d.get('results', {}).get(scen, {})
            ys.append(r.get('map_nops', np.nan))
            hits.append(r.get('n_chains_hit', 0) > 0)
        ax.plot(xs, ys, '-', color=color, lw=1, alpha=0.6)
        # 空心点 = 该配置没有任何链找到真公式，MAP 的"简洁"是平庸式而非真公式
        ax.plot([x for x, h in zip(xs, hits) if h],
                [y for y, h in zip(ys, hits) if h], 'o', color=color,
                label=label)
        ax.plot([x for x, h in zip(xs, hits) if not h],
                [y for y, h in zip(ys, hits) if not h], 'o', color=color,
                mfc='white', mew=1.4)
    ax.plot([], [], 'o', color='gray', mfc='white', mew=1.4,
            label='hollow = no chain found it')
    ax.set_xticks(xs); ax.set_xticklabels(short, rotation=30, ha='right',
                                          fontsize=8)
    ax.set_ylabel('MAP formula complexity (sympy ops)')
    ax.set_title('(b) Is the MAP still a "monster"?')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for scen, label, color in SCEN:
        ys = [d.get('results', {}).get(scen, {}).get('size_mean', np.nan)
              for d in runs]
        ax.plot(xs, ys, 'o-', color=color, label=label)
    ax.set_xticks(xs); ax.set_xticklabels(short, rotation=30, ha='right',
                                          fontsize=8)
    ax.set_ylabel('mean ops along the trace')
    ax.set_title('(c) How big do sampled formulas get?')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    mv = [d.get('results', {}).get('main', {}).get('move_frac', np.nan)
          for d in runs]
    ax.plot(xs, np.array(mv) * 100., 'o-', color='tab:purple',
            label='main: fraction of changed formulas')
    ax.set_xticks(xs); ax.set_xticklabels(short, rotation=30, ha='right',
                                          fontsize=8)
    ax.set_ylabel('% of consecutive samples that differ')
    ax.set_title('(d) Mixing collapses as the prior tightens')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle('Prior ablation: pendulum, identical data & seeds. '
                 'x-axis ordered by prior strength '
                 '(EP of the true formula, nat): ' +
                 ', '.join(f'{s:.1f}' for s in strength), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)


def main(argv=None):
    args = [a for a in (argv or sys.argv[1:])]
    runs = collect(args if args else None)
    if not runs:
        print('no runs found under', RUNROOT)
        return
    print(f'{len(runs)} runs: ' + ', '.join(d['_tag'] for d in runs))
    md = ['# 先验消融对照（钟摆，同数据同种子）\n',
          'x 轴按先验力度排序（真公式 _a_*sqrt(L/g) 在该先验下的 EP，nat）。\n']
    md.append(table_md(runs))
    OUT_MD.write_text('\n'.join(md))
    make_figure(runs)
    print('\n'.join(md))
    print(f'\n表已保存: {OUT_MD}')
    print(f'图已保存: {OUT_PNG}')


if __name__ == '__main__':
    main()
