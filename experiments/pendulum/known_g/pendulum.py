"""多星球单摆实验：仪器分辨率限制下能否还原 T = 2*pi*sqrt(L/g)？

实验设定
--------
在 5 颗重力加速度不同的星球（月球/火星/地球/海王星/木星）上测量单摆：
  - 摆长 L：米尺测量，分度值 0.01 m（读数按厘米舍入）
  - 周期 T：秒表测量，分度值 0.1 s（读数按 0.1 s 舍入）
每个星球测 60 个摆长，全部数据合并为单一数据集（L, g 为输入变量，
g 当作已知行星参数，无误差），使拟合常数跨星球共享。交给
BMS + 量纲宪法搜索解析表达式。

适用范围
--------
这是一个**回收测试**（injection/recovery test）：目标式已知，问的是搜索
走不走得到那儿。它不是发现实验——g 是作为输入列直接给出的，5 颗星球又被
合并成一张表、只允许一条公式，所以"存在一条跨星球的共同公式"是喂数据时
设定的前提，不是这里得到的结论。若要检验"从多份彼此独立的数据里发现共同
形状"，需要走 ``Tree(x=dict(...))`` 的多数据集入口（每份只给 L、T，不给
g），本脚本不做这件事。

结构先验可配置
--------------
``--prior`` 从 ``prior/registry.py`` 的登记表里取一组超参数，注入
``Tree(prior_par=...)``，走 core 的 ``EP`` 通道（含 ``Nopi2_*`` 二次项）
进入 MCMC 接受率。可用名字见 ``prior.list_priors()``，其中

* ``bms_wiki_nv13_np13``  —— [BMS2020] 论文所用先验（wiki 语料校准，
  一阶 18 项 + 乘法二次项）。本实验的默认先验。
* ``bms_wiki_nv13_np13_linear`` —— 同上但去掉二次项，用于消融。
* ``flat_default`` / ``flat_zero`` —— 未校准平值（10 nat）/ 纯 BIC。
* ``flat1.0`` —— 本实验 2026-10-16 那轮用的手工平值（1 nat/算子），
  不在登记表里（它是实验脚本内的历史设定），分析旧结果时作为目录名出现。

``--scale`` 对整组参数做乘性缩放（先验调温），用于力度消融。

对照实验
--------
A. 单星球（地球）且只给变量 L —— 本文件的原始设定。真公式
   ``a*sqrt(L)`` 在宪法档1（删掉 sqrt 偶数性检查与 ``**`` 底数无量纲
   要求）之前不可表达，所以 2026-10-17 那批迹线里对照 A 的 0%
   是"结构不可达"，不是采样失败。档1 之后该式可表达，g 的数值由
   拟合常数吸收。
B. 单星球（地球）但提供变量 L, g —— 60 个点，g 恒为 9.81，
   g 依赖性本身不可识别。
C. 5 星球 x 60 点但 g 一律给 9.81 —— 与主实验同样本量、同样变量集，
   只抽掉"g 在星球之间变化"这一件事。B 与主实验之间同时差着样本量，
   C 与主实验之间只差 g 的变化，加 C 是为了把这两个因素分开。

采样器
------
默认单温度（``Tree.mcmc``），与 ``pendulum_runs/`` 下 2026-10-15 ~
10-17 的历史结果可比。``--tempering N`` 改用 N 温度并行回火
（``Ts = [1] + [1.04**k for k in range(1, N)]``，[BMS2020] README 配方），
结果写进 ``<先验名>_ptN/`` 独立目录，不覆盖单温度产物。

判定标准
--------
把采样公式代入数据点，若 expr / sqrt(L/g) 在数值上为常数
（std/mean < 1e-4，使用迹线记录的拟合参数，fac 映射为
scipy.special.factorial），则属于"真公式族"；有效常数
c* = 该常数，与 2*pi = 6.2832 对比。

命中在链级上是全有全无的：一条链一旦落进真公式盆，其 400 个样本里
98~100% 都在族内，因此样本级占有率 ≈ 命中链数 / 总链数。两个数一起报；
它是重启成功率，不是后验质量，按前者读。

``--tempering`` 下四个场景的还原率都到了 100%，这个量随之饱和，不再适合
作因变量。``uses_g_frac``（后验样本的公式里是否出现 g）还有余量，但它能
说明的范围有限：g 是输入列、公式唯一性是前提，"变了的变量进公式、没变的
被常数吸收"主要由 BIC 的拟合改善决定。把它当作模型选择与复杂度惩罚是否
双向正常的检查来用，不要当作实验结论。

除"是否找到"外，还报告受先验影响的三个量：
  * MAP 公式（按后验能量 E = BIC/2 + EP 最小者，而非按 BIC 最小者）；
  * MAP 公式的复杂度（sympy count_ops）；
  * 迹线的 mixing（相邻样本公式变化率）与平均 EP。
按 BIC 取最优会绕过先验；按 E 取最优才包含先验的作用。
"""

import sys
import os
import re
import json
import argparse
import subprocess
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from multiprocessing import Pool

import scipy.special
from scipy.stats import beta as _beta
from sympy import sympify, lambdify, count_ops

from core import Tree, Node, OPS, Parallel
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
G_EARTH = PLANETS['Earth']
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
             ('ctrlB_Lg', ['L', 'g']),
             ('ctrlC_constG', ['L', 'g'])]
SCENARIO_LABELS = {
    'main': '主实验：5 星球，变量 L, g',
    'ctrlA_Lonly': '对照 A：仅地球，变量 L（原脚本逻辑）',
    'ctrlB_Lg': '对照 B：仅地球，变量 L, g',
    'ctrlC_constG': '对照 C：5 星球 x 60 点，g 恒为 9.81，变量 L, g',
}
# 每个场景喂 main() 里构造的哪一份数据
SCENARIO_FRAME = {'main': 'all', 'ctrlA_Lonly': 'earth',
                  'ctrlB_Lg': 'earth', 'ctrlC_constG': 'constG'}
SCENARIO_SHORT = {'main': '5 planets\nL,g', 'ctrlA_Lonly': 'Earth\nL only',
                  'ctrlB_Lg': 'Earth\nL,g', 'ctrlC_constG': '5 planets\ng fixed'}


def resolve_prior(name, scale=1.0):
    """按名取先验超参数；``flat1.0`` 是历史实验的手工平值（不在登记表里）。"""
    if name == 'flat1.0':
        return dict([('Nopi_%s' % op, 1.0) for op in OPS])
    return get_prior_par(name, scale=scale)


def run_tag(prior_name, scale, n_temps=0):
    tag = prior_name if scale == 1.0 else f'{prior_name}_x{scale:g}'
    return tag if not n_temps else f'{tag}_pt{n_temps}'


def ladder(n_temps):
    """[BMS2020] README 的几何温度阶梯：1, 1.04, 1.04^2, ..."""
    return [1] + [1.04 ** k for k in range(1, n_temps)]


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

    tree_kwargs = dict(
        variables=variables, parameters=['a'],
        dimensions=DIMS, constitutions=[DimensionalConstitution()],
        x=df[variables].reset_index(drop=True),
        y=df['T'].reset_index(drop=True),
        prior_par=prior_par, max_size=cfg['max_size'],
    )
    extra = {}
    if cfg['Ts']:
        sampler = Parallel(cfg['Ts'], **tree_kwargs)
        extra['anneal_after'] = cfg['anneal_after']
    else:
        sampler = Tree(**tree_kwargs)
    n_annealed = sampler.mcmc(tracefn=trace_fn, progressfn=progress_fn,
                              burnin=cfg['burnin'], thin=cfg['thin'],
                              samples=cfg['samples'],
                              progress=False, **extra)
    return {'trace_fn': trace_fn, 'tag': tag, 'chain_id': chain_id,
            'annealed': int(n_annealed or 0)}


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


def binom_ci(k, n, conf=0.95):
    """链级恢复率的 Clopper–Pearson 精确区间。

    与 2026-10-16 审计报告同一口径（那边 0/16 的上界 20.6% 就是精确区间给的），
    别换成 Wilson，两处数字会打架。
    """
    if n == 0:
        return (0.0, 1.0)
    a = (1. - conf) / 2.
    lo = 0.0 if k == 0 else float(_beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1. - a, k + 1, n - k))
    return (lo, hi)


def uses_variable(formula, name):
    """公式里是否出现了变量 name（整词匹配，避免把 grad/gamma 之类算成 g）。"""
    return re.search(r'(?<![A-Za-z_])%s(?![A-Za-z_])' % re.escape(name),
                     formula) is not None


def analyze_traces(trace_files, df, variables):
    """真公式族占有率、有效常数、最优 BIC、MAP（最小 E）、复杂度与 mixing。"""
    n_total, n_true = 0, 0
    n_uses_g = 0
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
                if 'g' in variables and uses_variable(formula, 'g'):
                    n_uses_g += 1
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
        # 还原率饱和后仍有余量的量。受"g 是喂进去的列、公式唯一性是前提"
        # 限制，只当模型选择的检查用，别当结论。
        'uses_g_frac': (n_uses_g / n_total) if 'g' in variables else None,
        'c_median': float(np.nanmedian(c_arr)),
        'c_p10': float(np.nanpercentile(c_arr, 10)),
        'c_p90': float(np.nanpercentile(c_arr, 90)),
        'n_chains_hit': len(first_hit),
        'n_chains': len(trace_files),
        'chain_ci': binom_ci(len(first_hit), len(trace_files)),
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


def reference_tree(df, variables, prior_par, formula='(_a_ * sqrt((L / g)))'):
    """朴素真公式在同一数据上的 BIC、能量与拟合常数。"""
    warnings.filterwarnings('ignore')
    t = Tree(
        variables=variables, parameters=['a'],
        dimensions=DIMS, constitutions=[DimensionalConstitution()],
        x=df[variables].reset_index(drop=True),
        y=df['T'].reset_index(drop=True),
        prior_par=prior_par,
        from_string=formula,
    )
    a_fit = t.par_values['d0']['_a_']
    E, EB, EP = t.get_energy(bic=False, reset=True)
    return {'bic': t.bic, 'a': a_fit, 'E': E, 'EP': EP}


def analyze_run(rundir, frames):
    """分析一个 run 目录下的各场景，返回 {tag: 指标}。"""
    rundir = Path(rundir)
    results = {}
    for tag, variables in SCENARIOS:
        fns = sorted(rundir.glob(f'trace_{tag}_*.dat'))
        if not fns:
            continue
        results[tag] = analyze_traces([str(f) for f in fns],
                                      frames[SCENARIO_FRAME[tag]], variables)
    return results


def print_results(results, prior_name, prior_par, refs):
    print('\n' + '=' * 72)
    print(f'结果汇总（先验 {prior_name}）')
    print('=' * 72)
    for tag, res in results.items():
        print(f'\n[{SCENARIO_LABELS.get(tag, tag)}]')
        lo, hi = res['chain_ci']
        print(f'  链级恢复率     : {res["n_chains_hit"]}/{res["n_chains"]} 条链'
              f' = {res["n_chains_hit"] / max(res["n_chains"], 1):.1%}'
              f'  95% CI [{lo:.1%}, {hi:.1%}]'
              f'   （样本级占有率 {res["n_true"]}/{res["n_total"]}'
              f' = {res["occ_true"]:.1%}）')
        if res.get('uses_g_frac') is not None:
            print(f'  用到 g 的样本  : {res["uses_g_frac"]:.1%}'
                  f'   （g 已给为输入变量；还原率饱和后改看这个量）')
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
    ref_E = float(refs['main']['E'])
    finals = []
    for pfn in sorted(Path(rundir).glob('progress_main_*.dat')):
        prog = np.atleast_2d(np.loadtxt(pfn))
        near = bool(prog[-1, 1] < ref_E + 50)
        finals.append((prog[-1, 1], near))
        ax.plot(prog[:, 0], prog[:, 1], lw=0.7, alpha=0.9 if near else 0.3,
                color='tab:green' if near else 'tab:gray')
    ax.axhline(ref_E, color='k', ls=':', lw=1.2)
    ax.set_xlabel('sample'); ax.set_ylabel('Energy  E = BIC/2 + EP')
    ax.set_title(f'(c) Energy traces, main experiment (prior={prior_name})')
    n_near = sum(near for _, near in finals)
    trapped = [e for e, near in finals if not near]
    handles = [Line2D([], [], color='tab:green', lw=1.4,
                      label=f'{n_near}/{len(finals)} chains end within '
                            f'50 nat of E$_{{true}}$')]
    if trapped:
        handles.append(Line2D([], [], color='tab:gray', lw=1.4, alpha=0.5,
                              label=f'{len(trapped)} end higher, median E '
                                    f'$\\approx$ {np.median(trapped):.0f}'))
    handles.append(Line2D([], [], color='k', ls=':', lw=1.2,
                          label=f'E(true formula) = {ref_E:.0f}'))
    ax.legend(handles=handles, fontsize=8, loc='upper right', framealpha=0.95)

    ax = axes[1, 1]
    tags = [t for t, _ in SCENARIOS if t in results]
    x = np.arange(len(tags), dtype=float)
    frac = np.array([results[t]['n_chains_hit'] / max(results[t]['n_chains'], 1)
                     for t in tags])
    ci = np.array([results[t]['chain_ci'] for t in tags])
    ug = np.array([results[t]['uses_g_frac'] or 0.0 for t in tags])
    has_g = np.array([results[t]['uses_g_frac'] is not None for t in tags])
    ax.bar(x - 0.19, frac, width=0.36,
           yerr=[frac - ci[:, 0], ci[:, 1] - frac],
           capsize=3, ecolor='black', error_kw=dict(lw=1),
           color='tab:green', label='chains recovered (95% CI)')
    ax.bar(x[has_g] + 0.19, ug[has_g], width=0.36, color='tab:blue',
           label='samples whose formula uses g')
    for i, t in enumerate(tags):
        r = results[t]
        ax.text(i - 0.19, min(ci[i, 1] + 0.04, 1.2),
                f'{r["n_chains_hit"]}/{r["n_chains"]}',
                ha='center', fontsize=8)
        if has_g[i]:
            ax.text(i + 0.19, ug[i] + 0.03, f'{ug[i]:.0%}',
                    ha='center', fontsize=8)
        else:
            ax.text(i + 0.19, 0.03, 'g not\noffered', ha='center',
                    fontsize=7, color='grey')
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_SHORT[t] for t in tags], fontsize=9)
    ax.set_ylim(0, 1.42)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('Fraction (per chain / per sample)')
    ax.set_title('(d) Chain-level recovery vs.\n'
                 'whether the formula contains g')
    ax.legend(fontsize=7, loc='upper center', ncol=2, framealpha=0.95)

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
    p.add_argument('--max-size', type=int, default=50,
                   help='表达式最大规模（默认 50，与历史结果一致）')
    p.add_argument('--tempering', type=int, default=0, metavar='N',
                   help='用 N 温度并行回火取代单温度（默认 0 = 单温度，'
                        '保持与 pendulum_runs/ 历史结果可比）')
    p.add_argument('--anneal-after', type=int, default=2000,
                   help='某对温度连续 N 个样本没有接受交换就整体退火'
                        '（仅 --tempering 生效，0 = 关闭）')
    p.add_argument('--seed-offset', type=int, default=0,
                   help='主实验链种子整体平移，用于在不覆盖已有迹线的前提下'
                        '扩充链数（文件名按 chain_id 编号，注意别撞车）')
    p.add_argument('--analyze-only', action='store_true',
                   help='不跑 MCMC，只重新分析已有 run 目录')
    p.add_argument('--no-fig', action='store_true')
    p.add_argument('--list-priors', action='store_true')
    return p.parse_args(argv)


def git_state():
    """记录产物出自哪个代码态 —— 结果能不能用，全看这一条。"""
    root = str(Path(__file__).resolve().parents[3])
    try:
        head = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=root, stderr=subprocess.DEVNULL).decode().splitlines()
        return {'head': head, 'dirty': bool(dirty),
                'dirty_paths': [l.split()[-1] for l in dirty][:20]}
    except Exception:
        return {'head': None, 'dirty': None, 'dirty_paths': []}


def main(argv=None):
    args = parse_args(argv)
    if args.list_priors:
        for n, d in list_priors():
            print(f'{n:28s} {d}')
        print(f'{"flat1.0":28s} 历史实验手工平值 1 nat/算子（实验脚本内）')
        return

    prior_par = resolve_prior(args.prior, args.scale)
    tag = run_tag(args.prior, args.scale, args.tempering)
    rundir = RUNROOT / tag
    Ts = ladder(args.tempering) if args.tempering else []
    cfg = {'burnin': args.burnin, 'thin': args.thin, 'samples': args.samples,
           'max_size': args.max_size, 'Ts': Ts,
           'anneal_after': args.anneal_after}

    df_all = make_data(PLANETS).reset_index(drop=True)
    # 对照 C：同样 5 x 60 个点、同样给 L 和 g，只是 g 不再在星球间变化。
    df_constG = make_data({name: G_EARTH for name in PLANETS}
                          ).reset_index(drop=True)
    df_earth = df_all[df_all['planet'] == 'Earth'].reset_index(drop=True)
    frames = {'all': df_all, 'earth': df_earth, 'constG': df_constG}

    print('=' * 72)
    print('多星球单摆实验：米尺 0.01 m / 秒表 0.1 s 分辨率')
    print('=' * 72)
    print(f'采样器 : ' + (f'{args.tempering} 温度并行回火 '
                          f'(Ts = 1.04^k, k=0..{args.tempering - 1})，'
                          f'停滞 {args.anneal_after} 样本后退火'
                          if Ts else '单温度 Tree.mcmc'))
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

    # 对照 A 只给 L：真律退化为 a*sqrt(L)，a = 2*pi/sqrt(g_earth)。
    refs = {
        'main': reference_tree(df_all, ['L', 'g'], prior_par),
        'ctrlA_Lonly': reference_tree(df_earth, ['L'], prior_par,
                                      formula='(_a_ * sqrt(L))'),
        'ctrlB_Lg': reference_tree(df_earth, ['L', 'g'], prior_par),
        'ctrlC_constG': reference_tree(df_constG, ['L', 'g'], prior_par),
    }
    print(f'\n参考 _a_*sqrt(L/g)：'
          f'5星球 BIC = {refs["main"]["bic"]:.1f} '
          f'(a = {refs["main"]["a"]:.4f}, EP = {refs["main"]["EP"]:.2f} nat, '
          f'E = {refs["main"]["E"]:.1f})；'
          f'仅地球 BIC = {refs["ctrlB_Lg"]["bic"]:.1f} '
          f'(a = {refs["ctrlB_Lg"]["a"]:.4f})；2*pi = {TWO_PI:.4f}')
    print(f'  对照 C（g 恒定，300 点）BIC = {refs["ctrlC_constG"]["bic"]:.1f} '
          f'(a = {refs["ctrlC_constG"]["a"]:.4f}, '
          f'E = {refs["ctrlC_constG"]["E"]:.1f})')
    print(f'  对照 A（只给 L，60 点）参考 a*sqrt(L)：'
          f'BIC = {refs["ctrlA_Lonly"]["bic"]:.1f} '
          f'(a = {refs["ctrlA_Lonly"]["a"]:.4f}，真值 2*pi/sqrt(9.81) = '
          f'{TWO_PI / np.sqrt(G_EARTH):.4f})')

    if not args.analyze_only:
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
        for i in range(args.chains_ctrl):
            tasks.append((4000 + i, df_constG, ['L', 'g'], 'ctrlC_constG', i,
                          prior_par, str(rundir), cfg))

        print(f'\n并行运行 {len(tasks)} 条链'
              f'（burnin={args.burnin}, thin={args.thin}, '
              f'samples={args.samples}, max_size={args.max_size}）...')
        with Pool(min(len(tasks), 16)) as pool:
            for k, done in enumerate(pool.imap_unordered(run_chain, tasks)):
                print(f'  [{k + 1}/{len(tasks)}] 完成: '
                      f'{Path(done["trace_fn"]).name}', flush=True)

    results = analyze_run(rundir, frames)
    print_results(results, args.prior, prior_par, refs)

    if not args.no_fig and results:
        fig_path = Path(__file__).resolve().parent / f'pendulum_{tag}.png'
        make_figure(results, df_all, rundir, args.prior, refs, fig_path)
        print(f'\n图已保存: {fig_path}')

    rundir.mkdir(parents=True, exist_ok=True)
    observed = {t: sorted(int(f.stem.replace(f'trace_{t}_', ''))
                          for f in rundir.glob(f'trace_{t}_*.dat'))
                for t, _ in SCENARIOS}
    with open(rundir / 'summary.json', 'w') as f:
        json.dump({'prior': args.prior, 'scale': args.scale,
                   'prior_par': prior_par,
                   'config': cfg,
                   'provenance': dict(
                       git_state(),
                       # --analyze-only 下的 git 态是"分析时"而非"采样时"，
                       # 别拿它判断这批迹线能不能用。
                       recorded_at='analysis' if args.analyze_only else 'sampling',
                       data_seed=DATA_SEED,
                       tempering=args.tempering or None,
                       Ts=Ts or None,
                       observed_chains=observed,
                       scenario_frames={t: SCENARIO_FRAME[t]
                                        for t, _ in SCENARIOS},
                       n_rows={k: len(v) for k, v in frames.items()},
                   ),
                   'reference': refs, 'results': results},
                  f, indent=2, default=str)
    print(f'汇总已保存: {rundir / "summary.json"}')


if __name__ == '__main__':
    main()
