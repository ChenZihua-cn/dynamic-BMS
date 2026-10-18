"""原仓库 BMS（并行回火 + 论文 prior.dat）单变量钟摆幂律实验。

问题
----
真实物理里学生只测两样东西：绳长 L（米尺）与周期 T（秒表）。重力加速度 g
不当作输入变量、也不告知。问：只用 (L, T)，原仓库的贝叶斯机器科学家能否
还原周期的幂律形式 T = a * L**b，并且 b ~= 0.5？

与 `experiments/pendulum/pendulum.py` 的区别
---------------------------------
本脚本**完全绕开本地 core/**，直接用上游仓库快照的计算代码：

* `mcmc.py`  —— Tree / OPS / 能量 / MCMC（无宪法层、无量纲约束）
* `parallel.py` —— Parallel 类：多温度并行回火 + 停滞时自动退火
* `Prior/fit_prior.py::read_prior_par` —— 读论文的先验超参数 .dat

温度表与先验注入方式均照上游 `README.md` 的教程配方：

    prior_par = read_prior_par('./Prior/final_..._nv13.np13....dat')
    Ts = [1] + [1.04**k for k in range(1, 20)]
    pms = Parallel(Ts, variables=..., parameters=..., x=x, y=y, prior_par=...)

条件（每个条件 = 一种采样配置 × 一种数据分辨率）
------------------------------------------------
* `pt_nv1np2`   —— 20 温度并行回火 + 上游 nv1.np2 先验（单变量语料校准，
                    与本问题的 1 变量 2 参数最匹配）
* `pt_nv13np13` —— 同上，但用 [BMS2020] 正文的 nv13.np13 先验
* `pt_flat`     —— 同上，prior_par={}（上游默认：每个算子 10 nat）
* `singleT`     —— Ts=[1]，只有 T=1 一条链，用来单独量化"并行回火有没有用"

数据分辨率
----------
* `normal` —— 10 个周期计时一次、0.01 s 舍入（中学常用）
* `coarse` —— 单摆一次计时、0.1 s 舍入（只够定性）
* `planets` —— 5 颗星球混合但**不告知 g**，g 的差异变成常数 a 的散差

判定（形式无关，纯数值）
------------------------
把采样公式在 L 网格上求值，做 log-log 线性回归：
斜率 = b^，R^2 高 => 该式就是一个纯幂律。于是"还原幂律"不依赖公式穿哪件
外衣（`a*sqrt(L)`、`sqrt(L)/a`、`L**b` 都算命中），幂律指数 b^ 直接从曲线
读出。真值 b = 0.5，a = 2*pi/sqrt(g) = 2.0061。
"""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get(
    'UPSTREAM_BMS', str(HERE.parents[2] / 'upstream-bms'))).resolve()
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / 'Prior'))

import pandas as pd
from random import seed as pyseed
from mcmc import Tree, OPS
from parallel import Parallel
from fit_prior import read_prior_par
from sympy import sympify, lambdify
import scipy.special

RUNROOT = HERE / 'upstream_runs'

# ---------------------------------------------------------------------------
# 实验常量
# ---------------------------------------------------------------------------
G_EARTH = 9.81
PLANETS = [1.62, 3.71, 9.81, 11.15, 24.79]
N_PER = 40
L_RANGE = (0.15, 2.5)
DATA_SEED = 20261018

TRUE_B = 0.5
TRUE_A = 2 * np.pi / np.sqrt(G_EARTH)

BURNIN = 1200
SAMPLES = 250
THIN = 5
MAX_SIZE = 20
PARAMS = ['_a_', '_b_']

# 上游 README 配方
TS_PT = [1] + [1.04 ** k for k in range(1, 20)]
TS_SINGLE = [1]

ANNEAL_AFTER = 2000      # 某温度连续这么久没接受 swap 就整体退火（0 = 关闭）

PRIORS = {
    'nv1np2': 'final_prior_param_sq.named_equations.nv1.np2.2017-10-18 18:07:35.163458.dat',
    'nv13np13': 'final_prior_param_sq.named_equations.nv13.np13.2016-09-01 17:05:57.196882.dat',
}

def _real(v):
    import numbers
    if isinstance(v, numbers.Complex) and not isinstance(v, numbers.Real):
        return float('nan')
    try:
        return float(v)
    except Exception:
        return float('nan')


CONDITIONS = {
    'pt_nv1np2': dict(Ts=TS_PT, prior='nv1np2'),
    'pt_nv13np13': dict(Ts=TS_PT, prior='nv13np13'),
    'pt_flat': dict(Ts=TS_PT, prior=None),
    'singleT_nv1np2': dict(Ts=TS_SINGLE, prior='nv1np2'),
    'singleT_nv13np13': dict(Ts=TS_SINGLE, prior='nv13np13'),
}

# 判据随分辨率而变：0.1 s 舍入只允许定性结论
TOL_B = {'normal': 0.02, 'coarse': 0.08, 'planets': 0.10}
R2_POWERLAW = 0.999


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def make_data(kind):
    """返回 (x, y)。x 只有一列 L —— g 从不进入输入。"""
    rng = np.random.default_rng(DATA_SEED)
    if kind == 'planets':
        g = np.repeat(np.array(PLANETS), N_PER)
        L = np.round(rng.uniform(*L_RANGE, size=g.size), 2)
        T = np.round(2 * np.pi * np.sqrt(L / g), 2)
    else:
        L = np.round(rng.uniform(*L_RANGE, size=N_PER), 2)
        T = np.round(2 * np.pi * np.sqrt(L / G_EARTH), 2 if kind == 'normal' else 1)
    return pd.DataFrame({'L': L}), pd.Series(T, name='T')


# ---------------------------------------------------------------------------
# 数值分类：形式无关地读幂律指数
# ---------------------------------------------------------------------------
_LGRID = np.linspace(L_RANGE[0], L_RANGE[1], 80)
_lam_cache = {}


_Lsym = sympify('L')


def _callable_for(formula, par_values):
    key = (formula, tuple(sorted(par_values.items())))
    if key in _lam_cache:
        return _lam_cache[key]
    fn = None
    try:
        ex = sympify(formula)
        subs = {s: par_values[s.name] for s in ex.free_symbols
                if s.name in par_values}
        if subs:
            ex = ex.subs(subs)
        names = sorted(str(s) for s in ex.free_symbols)
        if names in ([], ['L']):
            fn = lambdify([_Lsym] if names == ['L'] else [], ex,
                          ['numpy', {'fac': scipy.special.factorial}])
    except Exception:
        fn = None
    _lam_cache[key] = fn
    return fn


def powerlaw_fit(formula, par_values):
    """在 L 网格上把公式当作 a*L**b 拟合，返回 (b, ln a, R^2)；不合法返回 None。"""
    fn = _callable_for(formula, par_values)
    if fn is None:
        return None
    with np.errstate(all='ignore'):
        try:
            raw = fn(_LGRID)
            v = np.asarray(raw, dtype=float)
            if v.ndim == 0:                       # 常数表达式
                v = np.full(_LGRID.shape, float(v))
        except Exception:
            return None
    if v.shape != _LGRID.shape or not np.all(np.isfinite(v)) or np.any(v <= 0):
        return None
    lx, ly = np.log(_LGRID), np.log(v)
    b, c = np.polyfit(lx, ly, 1)
    pred = b * lx + c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(b), float(c), float(r2)


# ---------------------------------------------------------------------------
# 单条链
# ---------------------------------------------------------------------------
def run_chain(task):
    cond, data_kind, chain_id, rundir, args = task
    cfg = CONDITIONS[cond]
    x, y = make_data(data_kind)
    prior_par = ({} if cfg['prior'] is None
                 else read_prior_par(str(UPSTREAM / 'Prior' / PRIORS[cfg['prior']])))
    pyseed(args.seed0 + chain_id)
    np.random.seed(args.seed0 + chain_id)

    if len(cfg['Ts']) == 1:
        # 上游 Parallel.tree_swap 用 randint(0, len(Ts)-2)，单温度会炸
        # ("empty range in randrange")。单温度基线只能直接用 Tree。
        p = None
        t = Tree(variables=['L'], parameters=list(PARAMS), x=x, y=y,
                 prior_par=prior_par, max_size=args.max_size, BT=1., PT=1.)
    else:
        t = None
        p = Parallel(list(cfg['Ts']), variables=['L'], parameters=list(PARAMS),
                     x=x, y=y, prior_par=prior_par, max_size=args.max_size)

    t_start = time.time()
    Ts = p.Ts if p is not None else []
    last_swap = dict([(T, 0) for T in Ts[:-1]]) if len(Ts) > 1 else {}
    rows = []
    step = 0
    crashed = 0

    def do_steps(n):
        nonlocal step, crashed
        for _ in range(n):
            try:
                if p is not None:
                    p.mcmc_step()
                    BT1, BT2 = p.tree_swap()
                    if BT1 is not None and BT1 in last_swap:
                        last_swap[BT1] = step
                else:
                    t.mcmc_step()
            except Exception:
                # 上游 fit 出复数（如 c1**(I*pi)）会抛 ValueError/TypeError，
                # 记录并跳过该步，不让整条链夭折
                crashed += 1
            step += 1

    do_steps(args.burnin)

    annealed = 0
    for s in range(args.samples):
        do_steps(args.thin)
        cold = p.t1 if p is not None else t
        form = str(cold)
        try:
            pars = dict(cold.par_values['d0'])
        except Exception:
            pars = {}
        rec = powerlaw_fit(form, pars)
        rows.append(dict(
            sample=s, formula=form, pars={k: float(v) for k, v in pars.items()},
            E=_real(cold.E), EB=_real(cold.EB), EP=_real(cold.EP),
            b=(None if rec is None else rec[0]),
            loga=(None if rec is None else rec[1]),
            r2=(None if rec is None else rec[2]),
            nops=int(sum(cold.nops.values())),
        ))
        if args.anneal_after and last_swap:
            inactive = max([step - last_swap[T] for T in last_swap])
            if inactive > args.anneal_after:
                p.anneal(n=args.anneal_after * 1, factor=args.anneal_factor)
                last_swap = dict([(T, step) for T in last_swap])
                annealed += 1

    out = Path(rundir) / ('trace_%s_%s_%03d.jsonl' % (cond, data_kind, chain_id))
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    meta = dict(cond=cond, data=data_kind, chain=chain_id,
                wall=time.time() - t_start, steps=step, annealed=annealed,
                crashed=crashed,
                n_temps=len(cfg['Ts']), prior=cfg['prior'])
    with open(Path(rundir) / ('meta_%s_%s_%03d.json' % (cond, data_kind, chain_id)), 'w') as f:
        json.dump(meta, f)
    return meta


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------
def load_chain(rundir, cond, data_kind, cid):
    path = Path(rundir) / ('trace_%s_%s_%03d.jsonl' % (cond, data_kind, cid))
    if not path.exists():
        return None
    return [json.loads(l) for l in open(path)]


def summarize(cond, data_kind, rundir):
    tol = TOL_B[data_kind]
    per_chain, rows = [], []
    cid = 0
    while True:
        c = load_chain(rundir, cond, data_kind, cid)
        if c is None:
            break
        rows.extend(c)
        best = min(c, key=lambda r: r['E'])                  # MAP = 后验能量最小
        bmap = best['b'] if best['r2'] is not None and best['r2'] >= R2_POWERLAW else None
        hit_map = bmap is not None and abs(bmap - TRUE_B) <= tol
        fam = [r for r in c if r['r2'] is not None and r['r2'] >= R2_POWERLAW]
        hit_ever = any(abs(r['b'] - TRUE_B) <= tol for r in fam)
        changed = sum(1 for a, b in zip(c[:-1], c[1:]) if a['formula'] != b['formula'])
        per_chain.append(dict(chain=cid, map_formula=best['formula'], map_E=best['E'],
                              map_b=bmap, hit_map=hit_map, hit_ever=hit_ever,
                              occ_family=len(fam) / len(c),
                              mixing=changed / max(1, len(c) - 1)))
        cid += 1
    if not per_chain:
        return None

    fam = [r for r in rows if r['r2'] is not None and r['r2'] >= R2_POWERLAW]
    bs = np.array([r['b'] for r in fam]) if fam else np.array([])
    return dict(
        cond=cond, data=data_kind, chains=len(per_chain), samples=len(rows),
        occ_powerlaw=len(fam) / len(rows),
        b_median=(float(np.median(bs)) if bs.size else None),
        b_iqr=([float(np.percentile(bs, 25)), float(np.percentile(bs, 75))]
               if bs.size >= 4 else None),
        b_min=(float(bs.min()) if bs.size else None),
        b_max=(float(bs.max()) if bs.size else None),
        map_hit=sum(r['hit_map'] for r in per_chain),
        ever_hit=sum(r['hit_ever'] for r in per_chain),
        mixing_mean=float(np.mean([r['mixing'] for r in per_chain])),
        tol_b=tol, per_chain=per_chain,
    )


def write_report(all_sum, path):
    lines = ['# 原仓库 BMS 单变量钟摆幂律实验\n']
    lines.append('真值 b = %.3f，a = %.4f（= 2*pi/sqrt(9.81)）\n' % (TRUE_B, TRUE_A))
    lines.append('| 条件 | 数据 | 链数 | 幂律族占有率 | b^ 中位 | b^ IQR | MAP 命中 b^ | 曾命中 | mixing |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for s in all_sum:
        iqr = ('[%.3f, %.3f]' % tuple(s['b_iqr'])) if s['b_iqr'] else '—'
        bm = ('%.4f' % s['b_median']) if s['b_median'] is not None else '—'
        lines.append('| %s | %s | %d | %.1f%% | %s | %s | %d/%d | %d/%d | %.2f |' % (
            s['cond'], s['data'], s['chains'], 100 * s['occ_powerlaw'], bm, iqr,
            s['map_hit'], s['chains'], s['ever_hit'], s['chains'], s['mixing_mean']))
    lines.append('')
    for s in all_sum:
        lines.append('## %s / %s（容差 |b^-0.5|<=%.2f）\n' % (s['cond'], s['data'], s['tol_b']))
        for r in s['per_chain']:
            lines.append('- chain %02d MAP `%s` E=%.1f b^=%s 族内占比=%.0f%% mixing=%.2f%s' % (
                r['chain'], r['map_formula'], r['map_E'],
                ('%.4f' % r['map_b']) if r['map_b'] is not None else '—',
                100 * r['occ_family'], r['mixing'],
                '  <-- 命中' if r['hit_map'] else ''))
        lines.append('')
    Path(path).write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', default=','.join(CONDITIONS))
    ap.add_argument('--data', default='normal,coarse')
    ap.add_argument('--chains', type=int, default=12)
    ap.add_argument('--burnin', type=int, default=BURNIN)
    ap.add_argument('--samples', type=int, default=SAMPLES)
    ap.add_argument('--thin', type=int, default=THIN)
    ap.add_argument('--max-size', type=int, default=MAX_SIZE)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--anneal-after', type=int, default=ANNEAL_AFTER)
    ap.add_argument('--anneal-factor', type=float, default=5.0)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--skip-existing', action='store_true',
                    help='已有迹线的 (条件,数据,链号) 不重跑')
    args = ap.parse_args()
    args.seed0 = 7000 + args.seed_offset

    rundir = RUNROOT
    rundir.mkdir(parents=True, exist_ok=True)

    conds = [c for c in args.conditions.split(',') if c]
    datas = [d for d in args.data.split(',') if d]

    if not args.analyze_only:
        from multiprocessing import Pool
        tasks = []
        for cond in conds:
            for dk in datas:
                for cid in range(args.chains):
                    if args.skip_existing and load_chain(
                            str(rundir), cond, dk, cid) is not None:
                        continue
                    tasks.append((cond, dk, cid, str(rundir), args))
        print('>> %d chains to run' % len(tasks), flush=True)
        t0 = time.time()
        with Pool(processes=args.jobs) as pool:
            for i, meta in enumerate(pool.imap_unordered(run_chain, tasks)):
                print('  [%d/%d] %s/%s chain %02d %.0fs' % (
                    i + 1, len(tasks), meta['cond'], meta['data'],
                    meta['chain'], meta['wall']), flush=True)
        print('>> total %.0fs' % (time.time() - t0), flush=True)

    sums = []
    for cond in conds:
        for dk in datas:
            s = summarize(cond, dk, str(rundir))
            if s:
                sums.append(s)
    write_report(sums, rundir / 'report.md')
    with open(rundir / 'summary.json', 'w') as f:
        json.dump(sums, f, indent=1, default=str)

    for s in sums:
        print('%-16s %-8s chains=%2d occ=%5.1f%% b_med=%s MAP命中=%d/%d 曾命中=%d/%d mixing=%.2f' % (
            s['cond'], s['data'], s['chains'], 100 * s['occ_powerlaw'],
            ('%.4f' % s['b_median']) if s['b_median'] is not None else '—',
            s['map_hit'], s['chains'], s['ever_hit'], s['chains'],
            s['mixing_mean']))


if __name__ == '__main__':
    main()
