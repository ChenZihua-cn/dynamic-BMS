"""多数据集单摆实验：不给 g，5 份独立数据能否逼出一条共同公式？

与 known_g/pendulum.py 的区别
--------------------------------
`experiments/pendulum/known_g/pendulum.py` 把 5 颗星球合并成一张 300 行的表，并把 g 当作一列输入。那样
"只有一条公式"是喂数据时规定的，不是发现的；它证明的是"这条公式必须含 g"。

本脚本走 `Tree` 的多数据集通道：``x`` 是 ``{'d0'...d4}`` 的字典，每份只有
(L, T)，**g 从不作为输入出现**（除了候选列那一格）。表达式只有一条，常数按
数据集各自拟合。于是"是否存在共同公式"变成 BMS 自己要付账的假设。

三格设计
--------
1. ``md_Lonly``  变量 ['L']。基准：能否收敛到 a·√L 一条式子，以及 5 个拟合
   常数 a_i 是否各自等于 2*pi/sqrt(g_i)（即反推出各星球的重力加速度）。
2. ``md_candG``  变量 ['L', 'X']，X = 该星球的 g，每份数据内恒定、跨数据变化，
   **不告诉它 X 是什么**。这是"能否发现隐藏变量"那一问。
3. ``md_junkX``  变量 ['L', 'X']，X = 每星球一个随机数，同样份内恒定。
   负对照：X 与物理无关，且恒定值可以被该数据集的自由常数完全吸收。

判读要点（决定 2 能不能算"发现"）
--------------------------------
`core/energy.py` 里 ``k = 1 + len(parameters)`` 在 ``for ds in self.y`` 之外，
所以 **5 个数据集各自拟合一个常数，与 1 个数据集的复杂度惩罚相同**。把 X 用进
公式、让同一个常数跨星球复现，并不会减少 k。因此 BIC 对"用 X 把 5 个常数塌成
1 个"这件事**没有任何奖励**。

推论：md_candG 里 X 的使用率若与 md_junkX 的无差别，说明 BMS 不是"没看出 g"，
而是这套记账方式根本不区分二者 —— 隐藏变量的发现没有目标函数支撑。两格必须
一起跑，单跑 md_candG 得到 0% 是用不出结论的。

采样器与 known_g/pendulum.py 同口径：默认单温度，``--tempering N`` 走 core 的并行回火。
"""

import sys
import os
import re
import json
import time
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool

import scipy.special
from sympy import sympify, lambdify, count_ops

from core import Tree, Parallel
from experts.constitution.dimensional import DimensionalConstitution
from prior.registry import get_prior_par, DEFAULT_PRIOR

HERE = Path(__file__).resolve().parent
RUNROOT = HERE / 'multidataset_runs'

DATA_SEED = 20261018
PLANETS = {'Moon': 1.62, 'Mars': 3.71, 'Earth': 9.81,
           'Neptune': 11.15, 'Jupiter': 24.79}
N_PER = 60
L_RANGE = (0.1, 3.0)
DL = 0.01
DT = 0.1
TWO_PI = 2 * np.pi

DIMS = {'L': (1, 0, 0, 0, 0, 0, 0)}

SCENARIOS = {
    'md_Lonly':  dict(variables=['L'],        offer='none'),
    'md_candG':  dict(variables=['L', 'X'],   offer='g'),
    'md_junkX':  dict(variables=['L', 'X'],   offer='junk'),
}
LABELS = {
    'md_Lonly': '基准：5 份数据，只给 L',
    'md_candG': '候选列：X = 该星球的 g（不说明）',
    'md_junkX': '负对照：X = 每星球一个随机数',
}


def ladder(n):
    return [1] + [1.04 ** k for k in range(1, n)]


# ---------------------------------------------------------------------------
# 数据：5 个独立数据集，仪器分辨率模型同 known_g/pendulum.py
# ---------------------------------------------------------------------------

def make_frames(scenario, seed=DATA_SEED):
    """返回 (x, y, meta)。x/y 是按 d0..d4 索引的字典。"""
    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    junk = rng.uniform(0.5, 5.0, len(PLANETS))
    x, y, meta = {}, {}, {}
    for i, (name, gv) in enumerate(PLANETS.items()):
        L_true = rng.uniform(*L_RANGE, N_PER)
        L_obs = np.round(L_true / DL) * DL
        T_obs = np.round(TWO_PI * np.sqrt(L_true / gv) / DT) * DT
        row = {'L': L_obs}
        if cfg['offer'] == 'g':
            row['X'] = np.full(N_PER, gv)
        elif cfg['offer'] == 'junk':
            row['X'] = np.full(N_PER, junk[i])
        x['d%d' % i] = pd.DataFrame(row)
        y['d%d' % i] = pd.Series(T_obs, name='T')
        meta['d%d' % i] = {'planet': name, 'g': gv}
    return x, y, meta


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------

def run_chain(task):
    seed, scenario, chain_id, prior_par, rundir, cfg = task
    warnings.filterwarnings('ignore')
    np.random.seed(seed)
    import random
    random.seed(seed)

    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)
    trace_fn = str(rundir / f'trace_{scenario}_{chain_id}.dat')
    progress_fn = str(rundir / f'progress_{scenario}_{chain_id}.dat')

    x, y, _ = make_frames(scenario)
    variables = SCENARIOS[scenario]['variables']
    tree_kwargs = dict(
        variables=variables, parameters=['a'],
        dimensions=DIMS, constitutions=[DimensionalConstitution()],
        x=x, y=y, prior_par=prior_par, max_size=cfg['max_size'],
    )
    extra = {}
    if cfg['Ts']:
        sampler = Parallel(cfg['Ts'], **tree_kwargs)
        extra['anneal_after'] = cfg['anneal_after']
    else:
        sampler = Tree(**tree_kwargs)
    n_annealed = sampler.mcmc(tracefn=trace_fn, progressfn=progress_fn,
                              burnin=cfg['burnin'], thin=cfg['thin'],
                              samples=cfg['samples'], progress=False, **extra)
    return {'trace_fn': trace_fn, 'scenario': scenario, 'chain_id': chain_id,
            'annealed': int(n_annealed or 0)}


# ---------------------------------------------------------------------------
# 分析：一条式子是否成立、5 个常数是否各自反推出 g、候选列用没用
# ---------------------------------------------------------------------------

def uses_var(formula, name):
    return re.search(r'(?<![A-Za-z_])%s(?![A-Za-z_])' % re.escape(name),
                     formula) is not None


def sqrtL_coefficients(formula, par_values, x, rtol=1e-3):
    """若公式在每份数据上都正比于 sqrt(L)，返回 {ds: 有效系数 c}，否则 None。

    必须先过这一关再谈"反推 g"：把 (2*pi/a)^2 套到一个不是 a*sqrt(L) 的式子上
    只会得到量级荒谬的假数。
    """
    try:
        ex = sympify(formula.replace(' ', ''))
    except Exception:
        return None
    atomd = {a.name: a for a in ex.atoms() if a.is_Symbol}
    syms = list(atomd.values())
    try:
        flam = lambdify(syms, ex, [{'fac': scipy.special.factorial}, 'numpy'])
    except Exception:
        return None
    out = {}
    for ds, df in x.items():
        pv = (par_values or {}).get(ds, {})
        args = [df[v.name].values if v.name in df.columns
                else float(pv.get(v.name, 1.0)) for v in syms]
        try:
            with np.errstate(all='ignore'):
                pred = np.asarray(flam(*args), dtype=float)
        except Exception:
            return None
        base = np.sqrt(df['L'].values)
        if pred.shape != base.shape or not np.all(np.isfinite(pred)):
            return None
        with np.errstate(all='ignore'):
            ratio = pred / base
        if not np.all(np.isfinite(ratio)):
            return None
        m = ratio.mean()
        if m == 0 or abs(ratio.std() / m) > rtol:
            return None
        out[ds] = float(m)
    return out or None


def implied_g(coefficients):
    """由每份数据的有效系数 c 反推 g = (2*pi/c)^2 —— 仅在 c 来自 a*sqrt(L) 形状时有意义。"""
    return {ds: (TWO_PI / c) ** 2 for ds, c in coefficients.items() if c > 0}


_coef_cache = {}


def family_coefficients(formula, pv, x):
    key = (formula, json.dumps(pv, sort_keys=True, default=str))
    if key not in _coef_cache:
        _coef_cache[key] = sqrtL_coefficients(formula, pv, x)
    return _coef_cache[key]


def analyze(traces, scenario, x):
    variables = SCENARIOS[scenario]['variables']
    has_X = 'X' in variables
    n_total = 0
    n_with_X = 0
    n_family = 0
    sizes = []
    best_E = (np.inf, None, None)
    for fn in traces:
        with open(fn) as f:
            for line in f:
                rec = json.loads(line)
                E, formula, pv = rec[2], rec[4], rec[5]
                n_total += 1
                if has_X and uses_var(formula, 'X'):
                    n_with_X += 1
                if family_coefficients(formula, pv, x) is not None:
                    n_family += 1
                try:
                    sizes.append(int(count_ops(sympify(formula.replace(' ', '')))))
                except Exception:
                    pass
                if E < best_E[0]:
                    best_E = (E, formula, pv)
    E_best, mapf, mappv = best_E
    out = {
        'scenario': scenario,
        'n_chains': len(traces),
        'n_samples': n_total,
        'family_frac': n_family / max(n_total, 1),
        'map_E': float(E_best) if np.isfinite(E_best) else None,
        'map_formula': mapf,
        'map_nops': int(count_ops(sympify(mapf.replace(' ', '')))) if mapf else None,
        'map_par_values': mappv,
        'size_median': float(np.median(sizes)) if sizes else None,
    }
    if has_X:
        out['X_used_frac'] = n_with_X / max(n_total, 1)
        out['X_used_in_MAP'] = uses_var(mapf or '', 'X')
    else:
        out['X_used_frac'] = None
    # 只有 MAP 自己过了形状关，反推 g 才是有内容的数
    coef = family_coefficients(mapf, mappv, x) if mapf else None
    out['map_in_family'] = coef is not None
    if coef:
        ig = implied_g(coef)
        order = list(PLANETS)
        truth = {'d%d' % i: PLANETS[n] for i, n in enumerate(order)}
        out['implied_g'] = ig
        rel = [abs(ig[ds] - truth[ds]) / truth[ds] for ds in ig if ds in truth]
        out['implied_g_max_reldev'] = float(max(rel)) if rel else None
    return out


def print_results(res):
    print('\n' + '=' * 74)
    print('多数据集单摆实验结果')
    print('=' * 74)
    for sc in SCENARIOS:
        r = res.get(sc)
        if not r:
            continue
        print(f'\n[{LABELS[sc]}]')
        print(f'  链数/样本数    : {r["n_chains"]} / {r["n_samples"]}')
        print(f'  共同公式族占比 : {r["family_frac"]:.1%}'
              f'   （逐样本判 a*sqrt(L) 形状，各数据集常数可不同）')
        print(f'  MAP 公式       : ops={r["map_nops"]}, E={r["map_E"]:.1f}'
              f'{"  [过形状关]" if r.get("map_in_family") else "  [未过形状关，不反推 g]"}')
        print(f'    {(r["map_formula"] or "")[:92]}')
        if r.get('X_used_frac') is not None:
            print(f'  用到 X 的样本  : {r["X_used_frac"]:.1%}'
                  f'   MAP 含 X: {r["X_used_in_MAP"]}')
        if r.get('implied_g'):
            order = list(PLANETS)
            print(f'  由各数据集常数反推的 g（真值见括号）：')
            for ds, gv in sorted(r['implied_g'].items()):
                tv = PLANETS[order[int(ds[1:])]]
                print(f'    {ds}: {gv:7.3f}  ({tv:.2f}，偏差 '
                      f'{abs(gv - tv) / tv:.1%})')
            dev = r.get('implied_g_max_reldev')
            if dev is not None:
                print(f'  最大相对偏差   : {dev:.1%}')


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------

def make_figure(res, out_path, tag):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    x, y, _ = make_frames('md_Lonly')
    for i, (name, gv) in enumerate(PLANETS.items()):
        ax.plot(x['d%d' % i]['L'], y['d%d' % i], 'o', ms=3, label=f'{name}')
    ax.set_xlabel('L (m)'); ax.set_ylabel('T (s)')
    ax.set_title('5 independent datasets, g never shown')
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    sc = [s for s in SCENARIOS if s in res]
    vals = [res[s].get('X_used_frac') for s in sc]
    ok = [(s, v) for s, v in zip(sc, vals) if v is not None]
    if ok:
        bars = ax.bar([s for s, _ in ok], [v for _, v in ok], width=0.5,
                      color=['tab:blue', 'tab:orange'][:len(ok)])
        for b, (_, v) in zip(bars, ok):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.0%}',
                    ha='center', fontsize=10)
    ax.set_xticks(range(len(ok)))
    ax.set_xticklabels(['X = g\n(unexplained)', 'X = junk\n(negative control)']
                       [:len(ok)], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('samples whose formula uses X')
    ax.set_title('Does BMS reach for the hidden variable?')

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='多数据集单摆实验')
    p.add_argument('--prior', default=DEFAULT_PRIOR)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--burnin', type=int, default=4000)
    p.add_argument('--thin', type=int, default=10)
    p.add_argument('--samples', type=int, default=400)
    p.add_argument('--chains', type=int, default=16)
    p.add_argument('--max-size', type=int, default=50)
    p.add_argument('--tempering', type=int, default=20)
    p.add_argument('--anneal-after', type=int, default=0)
    p.add_argument('--scenarios', default=','.join(SCENARIOS))
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--analyze-only', action='store_true')
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument('--rundir', default=None, help='覆盖输出目录（集群上用绝对路径）')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prior_par = dict(get_prior_par(args.prior, scale=args.scale))
    Ts = ladder(args.tempering) if args.tempering else []
    tag = ('%s%s' % (args.prior, '' if args.scale == 1 else '_x%g' % args.scale)
           + ('_pt%d' % args.tempering if args.tempering else '_singleT'))
    rundir = Path(args.rundir) if args.rundir else RUNROOT / tag
    rundir.mkdir(parents=True, exist_ok=True)

    scen = [s for s in args.scenarios.split(',') if s in SCENARIOS]
    cfg = {'burnin': args.burnin, 'thin': args.thin, 'samples': args.samples,
           'max_size': args.max_size, 'Ts': Ts,
           'anneal_after': args.anneal_after}

    print('=' * 74)
    print('多数据集单摆实验：5 份独立数据，g 不作为输入')
    print('=' * 74)
    print(f'采样器 : {args.tempering or "单温度"}  先验 : {args.prior} '
          f'  链数/格 : {args.chains}  目录 : {rundir}')

    if not args.analyze_only:
        tasks = []
        for k, sc in enumerate(scen):
            for i in range(args.chains):
                tasks.append((5000 + 1000 * k + args.seed_offset + i, sc,
                              i + args.seed_offset, prior_par, str(rundir), cfg))
        t0 = time.time()
        with Pool(min(len(tasks), args.jobs)) as pool:
            for j, done in enumerate(pool.imap_unordered(run_chain, tasks)):
                print('  [%d/%d] %s chain %02d  %.0fs'
                      % (j + 1, len(tasks), done['scenario'], done['chain_id'],
                         time.time() - t0), flush=True)

    res = {}
    for sc in scen:
        fns = sorted(str(f) for f in rundir.glob(f'trace_{sc}_*.dat'))
        if fns:
            x_sc, _, _ = make_frames(sc)
            res[sc] = analyze(fns, sc, x_sc)
    print_results(res)

    with open(rundir / 'summary.json', 'w') as f:
        json.dump({'prior': args.prior, 'scale': args.scale, 'config': cfg,
                   'planets': PLANETS, 'data_seed': DATA_SEED,
                   'results': res}, f, indent=2, default=str)
    fig_path = rundir / f'multidataset_{tag}.png'
    if res:
        make_figure(res, fig_path, tag)
        print(f'\n图已保存: {fig_path}')
    print(f'汇总已保存: {rundir / "summary.json"}')


if __name__ == '__main__':
    main()
