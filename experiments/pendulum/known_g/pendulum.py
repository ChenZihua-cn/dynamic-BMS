"""多星球单摆实验：仪器分辨率限制下能否还原 T = 2*pi*sqrt(L/g)？

实验设定
--------
在 5 颗重力加速度不同的星球（月球/火星/地球/海王星/木星）上测量单摆：
  - 摆长 L：米尺测量，分度值 0.01 m（读数按厘米舍入）
  - 周期 T：秒表测量，分度值 0.1 s（读数按 0.1 s 舍入）
每个星球测 60 个摆长，全部数据合并为单一数据集（L, g 为输入变量，
g 当作已知行星参数，无误差），使拟合常数跨星球共享。交给
BMS + 量纲宪法搜索解析表达式。

结构先验可配置
--------------
``--prior`` 从 ``prior/registry.py`` 的登记表里取一组超参数，注入
``Tree(prior_par=...)``，走 core 的 ``EP`` 通道（含 ``Nopi2_*`` 二次项）
进入 MCMC 接受率。可用名字见 ``prior.list_priors()``，其中

* ``bms_wiki_nv13_np13``  —— [BMS2020] 论文所用先验（wiki 语料校准，
  一阶 18 项 + 乘法二次项）。**本轮实验的默认与主角**。
* ``bms_wiki_nv13_np13_linear`` —— 同上但去掉二次项，用于消融。
* ``flat_default`` / ``flat_zero`` —— 未校准平值（10 nat）/ 纯 BIC。
* ``flat1.0`` —— 本实验 2026-10-16 那轮用的手工平值（1 nat/算子），
  不在登记表里（它是实验脚本内的历史设定），分析旧结果时作为目录名出现。

``--scale`` 对整组参数做乘性缩放（先验调温），用于力度消融。

对照实验
--------
A. 单星球（地球）且只给变量 L —— 即本文件原始版本的逻辑。
   注意：量纲宪法的 sqrt 偶数性检查要求根号内量纲指数为偶数，
   sqrt(L)（量纲 L^1）会被硬拒绝，因此该设定在原理上就不可能
   找到真公式 a*sqrt(L)。
B. 单星球（地球）但提供变量 L, g —— 检验没有 g 的变化时，
   结构是否仍能被找到（此时 g 依赖性本身不可识别，但宪法 +
   提供的 g 列可能依然把搜索引向 a*sqrt(L/g)）。

判定标准
--------
把采样公式代入数据点，若 expr / sqrt(L/g) 在数值上为常数
（std/mean < 1e-4，使用迹线记录的拟合参数，fac 映射为
scipy.special.factorial），则属于"真公式族"；有效常数
c* = 该常数，与 2*pi = 6.2832 对比。

除"是否找到"外，还报告先验真正影响的三个量：
  * MAP 公式（按后验能量 E = BIC/2 + EP 最小者，而非按 BIC 最小者）；
  * MAP 公式的复杂度（sympy count_ops）；
  * 迹线的 mixing（相邻样本公式变化率）与平均 EP。
按 BIC 取最优会绕过先验，只有按 E 取最优才能看出先验在做什么。
"""

import sys
import os
import json
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

from core import Tree, Node, OPS
from experts.constitution.dimensional import (
    DimensionalConstitution,
    DimExpr,
    DimensionCheckResult,
    DimensionFailure,
    ZERO,
    collect_constraints, solve_constraints,
)
from prior.registry import get_prior_par, list_priors, DEFAULT_PRIOR

# ---------------------------------------------------------------------------
# 实验配置
# ---------------------------------------------------------------------------

DATA_SEED = 20261014

PLANETS = {           # 重力加速度 m/s^2
    'Moon': 1.62,
    'Mars': 3.71,
    'Earth': 9.81,
    'Neptune': 11.15,
    'Jupiter': 24.79,
}
N_PER_PLANET = 60
L_RANGE = (0.1, 3.0)  # m
DL = 0.01             # 米尺分度值
DT = 0.1              # 秒表分度值

DIMS = {'L': (1, 0, 0, 0, 0, 0, 0), 'g': (1, 0, -2, 0, 0, 0, 0)}

BURNIN = 4000
THIN = 10
SAMPLES = 400
CHAINS_MAIN = 8
CHAINS_CTRL = 4

RUNROOT = Path(__file__).resolve().parent / 'pendulum_runs'

TWO_PI = 2 * np.pi
PARAMS = ['_a_']

SCENARIOS = [('main', ['L', 'g']),
             ('ctrlA_Lonly', ['L']),
             ('ctrlB_Lg', ['L', 'g'])]
SCENARIO_LABELS = {
    'main': '主实验：5 星球，变量 L, g',
    'ctrlA_Lonly': '对照 A：仅地球，变量 L（原脚本逻辑）',
    'ctrlB_Lg': '对照 B：仅地球，变量 L, g',
}


def resolve_prior(name, scale=1.0):
    """按名取先验超参数；``flat1.0`` 是历史实验的手工平值（不在登记表里）。"""
    if name == 'flat1.0':
        return dict([('Nopi_%s' % op, 1.0) for op in OPS])
    return get_prior_par(name, scale=scale)


def run_tag(prior_name, scale):
    return prior_name if scale == 1.0 else f'{prior_name}_x{scale:g}'


# ---------------------------------------------------------------------------
# 数据生成：仪器分辨率模型（读数 = 真值按分度值舍入）
# ---------------------------------------------------------------------------

def make_data(planets, n=N_PER_PLANET, seed=DATA_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for name, g_val in planets.items():
        L_true = rng.uniform(*L_RANGE, n)
        L_obs = np.round(L_true / DL) * DL          # 米尺读数
        T_true = TWO_PI * np.sqrt(L_true / g_val)   # 物理作用在真实摆长上
        T_obs = np.round(T_true / DT) * DT          # 秒表读数
        rows.append(pd.DataFrame({
            'planet': name, 'g': g_val, 'L': L_obs, 'T': T_obs,
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# MCMC worker：每条链独立进程
# ---------------------------------------------------------------------------

def run_chain(task):
    seed, df, variables, tag, chain_id, prior_par, rundir, cfg = task
    warnings.filterwarnings('ignore')
    np.random.seed(seed)
    import random
    random.seed(seed)

    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)
    trace_fn = str(rundir / f'trace_{tag}_{chain_id}.dat')
    progress_fn = str(rundir / f'progress_{tag}_{chain_id}.dat')

    t = Tree(
        variables=variables, parameters=['a'],
        dimensions=DIMS, constitutions=[DimensionalConstitution()],
        x=df[variables].reset_index(drop=True),
        y=df['T'].reset_index(drop=True),
        prior_par=prior_par,
    )
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=cfg['burnin'], thin=cfg['thin'], samples=cfg['samples'],
           progress=False)
    return trace_fn


# ---------------------------------------------------------------------------
# 迹线分析：数值判定是否属于真公式族 expr == c * sqrt(L/g)
# ---------------------------------------------------------------------------

def eval_formula(formula, par_values, df, variables):
    """把采样公式（含拟合参数）代入数据点，返回 (预测值, 运算数) 或 (None, None)。"""
    try:
        ex = sympify(formula.replace(' ', ''))
    except Exception:
        return None, None
    try:
        nops = int(count_ops(ex))
    except Exception:
        nops = -1
    atomd = {a.name: a for a in ex.atoms() if a.is_Symbol}
    var_syms = [atomd[v] for v in variables if v in atomd]
    par_syms = [atomd[p] for p in PARAMS if p in atomd]
    try:
        flam = lambdify(var_syms + par_syms, ex,
                        [{'fac': scipy.special.factorial}, 'numpy'])
        args = [df[v].values for v in variables if v in atomd]
        pv = par_values.get('d0', par_values) \
            if isinstance(par_values, dict) else {}
        args += [pv.get(p, 1.0) for p in PARAMS if p in atomd]
        with np.errstate(all='ignore'):
            vals = np.asarray(flam(*args), dtype=float)
        if vals.shape != df['L'].values.shape:
            return None, nops
        return vals, nops
    except Exception:
        return None, nops


def true_family_constant(formula, par_values, df, variables, rtol=1e-4):
    """若 expr 在全部数据点上等于 c*sqrt(L/g)，返回 (c, 运算数)，否则 (None, nops)。"""
    vals, nops = eval_formula(formula, par_values, df, variables)
    if vals is None:
        return None, nops
    target = np.sqrt(df['L'].values / df['g'].values)
    with np.errstate(all='ignore'):
        ratio = vals / target
    if not np.all(np.isfinite(ratio)):
        return None, nops
    m = np.mean(ratio)
    if m == 0 or np.std(ratio) / abs(m) > rtol:
        return None, nops
    return m, nops


def analyze_traces(trace_files, df, variables):
    """真公式族占有率、有效常数、最优 BIC、MAP（最小 E）、复杂度与 mixing。"""
    n_total, n_true = 0, 0
    c_eff, sizes, eps = [], [], []
    best_bic = (np.inf, None, None, None)   # (bic, formula, c, nops)
    best_E = (np.inf, None, None, None)     # (E, formula, c, nops)  <- MAP
    first_hit = {}
    n_changes, n_pairs = 0, 0
    for fn in trace_files:
        prev_formula = None
        with open(fn) as f:
            for line in f:
                rec = json.loads(line)
                sidx, bic, E, formula, pv = rec[0], rec[1], rec[2], rec[4], rec[5]
                n_total += 1
                eps.append(E - bic / 2.)            # EP = E - EB
                if prev_formula is not None:
                    n_pairs += 1
                    n_changes += int(formula != prev_formula)
                prev_formula = formula
                c, nops = true_family_constant(formula, pv, df, variables)
                if nops is not None and nops >= 0:
                    sizes.append(nops)
                if c is not None:
                    n_true += 1
                    c_eff.append(c)
                    if fn not in first_hit:
                        first_hit[fn] = sidx
                if bic < best_bic[0]:
                    best_bic = (bic, formula, c, nops)
                if E < best_E[0]:
                    best_E = (E, formula, c, nops)
    c_arr = np.array(c_eff) if c_eff else np.array([np.nan])
    return {
        'n_total': n_total,
        'n_true': n_true,
        'occ_true': n_true / max(n_total, 1),
        'c_median': float(np.nanmedian(c_arr)),
        'c_p10': float(np.nanpercentile(c_arr, 10)),
        'c_p90': float(np.nanpercentile(c_arr, 90)),
        'n_chains_hit': len(first_hit),
        'n_chains': len(trace_files),
        'first_hit_samples': sorted(first_hit.values()),
        'best_bic': best_bic[0],
        'best_bic_formula': best_bic[1],
        'best_bic_c': best_bic[2],
        'best_bic_nops': best_bic[3],
        'map_E': best_E[0],
        'map_formula': best_E[1],
        'map_c': best_E[2],
        'map_nops': best_E[3],
        'size_mean': float(np.mean(sizes)) if sizes else float('nan'),
        'size_median': float(np.median(sizes)) if sizes else float('nan'),
        'ep_mean': float(np.mean(eps)) if eps else float('nan'),
        'move_frac': n_changes / max(n_pairs, 1),
    }


def reference_tree(df, variables, prior_par):
    """朴素真公式 _a_*sqrt(L/g) 在同一数据上的 BIC、能量与拟合常数。"""
    warnings.filterwarnings('ignore')
    t = Tree(
        variables=variables, parameters=['a'],
        dimensions=DIMS, constitutions=[DimensionalConstitution()],
        x=df[variables].reset_index(drop=True),
        y=df['T'].reset_index(drop=True),
        prior_par=prior_par,
        from_string='(_a_ * sqrt((L / g)))',
    )
    a_fit = t.par_values['d0']['_a_']
    E, EB, EP = t.get_energy(bic=False, reset=True)
    return {'bic': t.bic, 'a': a_fit, 'E': E, 'EP': EP}


def analyze_run(rundir, df_all, df_earth, prior_par=None):
    """分析一个 run 目录下的三个场景，返回 {tag: 指标}。"""
    rundir = Path(rundir)
    results = {}
    for tag, variables in SCENARIOS:
        fns = sorted(rundir.glob(f'trace_{tag}_*.dat'))
        if not fns:
            continue
        dfc = df_all if tag == 'main' else df_earth
        results[tag] = analyze_traces([str(f) for f in fns], dfc, variables)
    return results


def print_results(results, prior_name, prior_par, refs):
    print('\n' + '=' * 72)
    print(f'结果汇总（先验 {prior_name}）')
    print('=' * 72)
    for tag, res in results.items():
        print(f'\n[{SCENARIO_LABELS.get(tag, tag)}]')
        print(f'  真公式族占有率 : {res["n_true"]}/{res["n_total"]}'
              f' = {res["occ_true"]:.1%}'
              f'   （命中过的链: {res["n_chains_hit"]}/{res["n_chains"]}）')
        if res['n_true']:
            print(f'  有效常数 c*    : 中位 {res["c_median"]:.4f}'
                  f'（10-90%: {res["c_p10"]:.4f} ~ {res["c_p90"]:.4f}），'
                  f'2*pi = {TWO_PI:.4f}，偏差 '
                  f'{abs(res["c_median"] - TWO_PI) / TWO_PI:.2%}')
            print(f'  首次命中样本序 : {res["first_hit_samples"]}')
        print(f'  最优 BIC 公式  : ops={res["best_bic_nops"]}, '
              f'BIC={res["best_bic"]:.1f}')
        print(f'    {(res["best_bic_formula"] or "")[:88]}')
        ref = refs.get(tag)
        ref_txt = (f'（参考真公式 E={ref["E"]:.1f}）' if ref else '（无参考）')
        print(f'  MAP(最小E)公式 : ops={res["map_nops"]}, '
              f'E={res["map_E"]:.1f}{ref_txt}')
        print(f'    {(res["map_formula"] or "")[:88]}')
        print(f'  迹线平均复杂度 : ops 均值 {res["size_mean"]:.1f} / '
              f'中位 {res["size_median"]:.0f}；'
              f'平均 EP {res["ep_mean"]:.1f} nat；'
              f'mixing(相邻变化率) {res["move_frac"]:.2f}')


# ---------------------------------------------------------------------------
# 画图
# ---------------------------------------------------------------------------

def make_figure(results, df_all, rundir, prior_name, refs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    for name, g_val in PLANETS.items():
        sub = df_all[df_all['planet'] == name]
        ax.scatter(sub['L'], sub['T'], s=12, label=f'{name} (g={g_val})')
        Lg = np.linspace(*L_RANGE, 100)
        ax.plot(Lg, TWO_PI * np.sqrt(Lg / g_val), '--', lw=0.8, alpha=0.5)
    ax.set_xlabel('L (m)'); ax.set_ylabel('T (s)')
    ax.set_title('(a) Simulated readings (0.01 m / 0.1 s resolution)')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    x_col = np.sqrt(df_all['L'] / df_all['g'])
    ax.scatter(x_col, df_all['T'], s=12, c='tab:blue', label='readings')
    xs = np.linspace(x_col.min(), x_col.max(), 50)
    ax.plot(xs, TWO_PI * xs, 'r-', lw=1.5, label=r'$T = 2\pi\sqrt{L/g}$')
    ax.set_xlabel(r'$\sqrt{L/g}$ (s)'); ax.set_ylabel('T (s)')
    ax.set_title('(b) All planets collapse on the sqrt(L/g) axis')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for pfn in sorted(Path(rundir).glob('progress_main_*.dat')):
        prog = np.loadtxt(pfn)
        ax.plot(prog[:, 0], prog[:, 1], lw=0.8, alpha=0.8,
                label=pfn.stem.replace('progress_main_', 'chain '))
    ax.axhline(refs['main']['E'], color='k', ls=':', lw=1,
               label='E(true formula) reference')
    ax.set_xlabel('sample'); ax.set_ylabel('Energy  E = BIC/2 + EP')
    ax.set_title(f'(c) Energy traces, main experiment (prior={prior_name})')
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    tags = [t for t, _ in SCENARIOS if t in results]
    occ = [results[t]['occ_true'] for t in tags]
    bars = ax.bar(range(len(tags)), occ,
                  color=['tab:green', 'tab:red', 'tab:orange'][:len(tags)])
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(['5 planets\nL,g', 'Earth\nL only', 'Earth\nL,g'][:len(tags)],
                       fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('True-family posterior occupancy')
    ax.set_title('(d) Recovery rate of T = c*sqrt(L/g)')
    for b, o in zip(bars, occ):
        ax.text(b.get_x() + b.get_width() / 2, o + 0.02, f'{o:.0%}',
                ha='center', fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='多星球单摆实验（先验可配置）')
    p.add_argument('--prior', default=DEFAULT_PRIOR,
                   help='先验登记名，或 flat1.0（历史手工平值）')
    p.add_argument('--scale', type=float, default=1.0,
                   help='先验整体缩放（调温）')
    p.add_argument('--burnin', type=int, default=BURNIN)
    p.add_argument('--thin', type=int, default=THIN)
    p.add_argument('--samples', type=int, default=SAMPLES)
    p.add_argument('--chains-main', type=int, default=CHAINS_MAIN)
    p.add_argument('--chains-ctrl', type=int, default=CHAINS_CTRL)
    p.add_argument('--seed-offset', type=int, default=0,
                   help='主实验链种子整体平移，用于在不覆盖已有迹线的前提下'
                        '扩充链数（文件名按 chain_id 编号，注意别撞车）')
    p.add_argument('--analyze-only', action='store_true',
                   help='不跑 MCMC，只重新分析已有 run 目录')
    p.add_argument('--no-fig', action='store_true')
    p.add_argument('--list-priors', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list_priors:
        for n, d in list_priors():
            print(f'{n:28s} {d}')
        print(f'{"flat1.0":28s} 历史实验手工平值 1 nat/算子（实验脚本内）')
        return

    prior_par = resolve_prior(args.prior, args.scale)
    tag = run_tag(args.prior, args.scale)
    rundir = RUNROOT / tag

    df_all = make_data(PLANETS).reset_index(drop=True)
    df_earth = df_all[df_all['planet'] == 'Earth'].reset_index(drop=True)

    print('=' * 72)
    print('多星球单摆实验：米尺 0.01 m / 秒表 0.1 s 分辨率')
    print('=' * 72)
    print(f'先验 : {args.prior}（scale={args.scale:g}）')
    print(f'  一阶项 {len([k for k in prior_par if k.startswith("Nopi_")])} 个，'
          f'二阶项 {len([k for k in prior_par if k.startswith("Nopi2_")])} 个')
    print(f'  取值区间: '
          f'{[round(v, 3) for v in prior_par.values()][:3]} ... '
          f'min={min(prior_par.values()):.3f} max={max(prior_par.values()):.3f}')
    print(f'数据 : {len(df_all)} 行（5 星球 x {N_PER_PLANET}），'
          f'T 范围 [{df_all["T"].min():.1f}, {df_all["T"].max():.1f}] s')
    print(f'舍入噪声: E[dT^2] = {DT**2/12:.4f} s^2，'
          f'最短周期 {df_all["T"].min():.1f} s 处相对精度 ~'
          f'{DT/2/df_all["T"].min():.0%}')

    # 对照 A 的变量集里没有 g，真公式在其空间内不可表达，故无参考值。
    refs = {
        'main': reference_tree(df_all, ['L', 'g'], prior_par),
        'ctrlB_Lg': reference_tree(df_earth, ['L', 'g'], prior_par),
    }
    print(f'\n参考 _a_*sqrt(L/g)：'
          f'5星球 BIC = {refs["main"]["bic"]:.1f} '
          f'(a = {refs["main"]["a"]:.4f}, EP = {refs["main"]["EP"]:.2f} nat, '
          f'E = {refs["main"]["E"]:.1f})；'
          f'仅地球 BIC = {refs["ctrlB_Lg"]["bic"]:.1f} '
          f'(a = {refs["ctrlB_Lg"]["a"]:.4f})；2*pi = {TWO_PI:.4f}')

    if not args.analyze_only:
        cfg = {'burnin': args.burnin, 'thin': args.thin,
               'samples': args.samples}
        tasks = []
        for i in range(args.chains_main):
            # 种子与文件名都按 seed_offset 平移，便于在已有 run 上追加链。
            tasks.append((1000 + args.seed_offset + i, df_all, ['L', 'g'],
                          'main', i + args.seed_offset,
                          prior_par, str(rundir), cfg))
        for i in range(args.chains_ctrl):
            tasks.append((2000 + i, df_earth, ['L'], 'ctrlA_Lonly', i,
                          prior_par, str(rundir), cfg))
        for i in range(args.chains_ctrl):
            tasks.append((3000 + i, df_earth, ['L', 'g'], 'ctrlB_Lg', i,
                          prior_par, str(rundir), cfg))

        print(f'\n并行运行 {len(tasks)} 条链'
              f'（burnin={args.burnin}, thin={args.thin}, '
              f'samples={args.samples}）...')
        with Pool(min(len(tasks), 16)) as pool:
            for k, fn in enumerate(pool.imap_unordered(run_chain, tasks)):
                print(f'  [{k + 1}/{len(tasks)}] 完成: {Path(fn).name}',
                      flush=True)

    results = analyze_run(rundir, df_all, df_earth)
    print_results(results, args.prior, prior_par, refs)

    if not args.no_fig and results:
        fig_path = Path(__file__).resolve().parent / f'pendulum_{tag}.png'
        make_figure(results, df_all, rundir, args.prior, refs, fig_path)
        print(f'\n图已保存: {fig_path}')

    rundir.mkdir(parents=True, exist_ok=True)
    with open(rundir / 'summary.json', 'w') as f:
        json.dump({'prior': args.prior, 'scale': args.scale,
                   'prior_par': prior_par,
                   'config': {'burnin': args.burnin, 'thin': args.thin,
                              'samples': args.samples},
                   'reference': refs, 'results': results},
                  f, indent=2, default=str)
    print(f'汇总已保存: {rundir / "summary.json"}')


if __name__ == '__main__':
    main()
