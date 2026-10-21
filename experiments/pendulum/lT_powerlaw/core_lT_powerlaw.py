"""本地 core（单温度 + 可选量纲宪法）在同样的单变量钟摆数据上的对照。

与 `upstream_lT_powerlaw.py` 共用数据生成与幂律判据，唯一改变的是搜索端：

* `core_const`   —— 现状：量纲宪法开启（`DimensionalConstitution`）
* `core_noconst` —— 关掉宪法，其余（先验、提议、单温度）一字不动

两者的差就是"宪法到底有没有帮忙"的答案；两者与上游 PT 条件的差，则是
"并行回火有没有帮忙"的答案。
"""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import sys
import json
import time
import random
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import upstream_lT_powerlaw as U
from core import Tree, OPS
from experts.constitution.dimensional import DimensionalConstitution
from prior.registry import get_prior_par

RUNROOT = HERE / 'upstream_runs'          # 与上游结果同一目录，前缀 core_
OUT = [RUNROOT]                           # 可被 --rundir 覆盖

L_DIM = {'L': (1, 0, 0, 0, 0, 0, 0)}

CONDITIONS = {
    'core_const': dict(constitution=True),
    'core_noconst': dict(constitution=False),
}


def allow_fractional_dimensions():
    """放开 sqrt 的"量纲指数必须为偶数"限制。

    分数维指数在量纲分析里完全合法：T = a*L^(1/2) 只要求 [a] = s*m^(-1/2)。
    DimExpr 本来就用 Fraction 存指数，这条偶数性检查只是为了把量纲保持在整数
    上，代价是把摆的平方根律本身从搜索空间里删掉了。
    """
    from experts.constitution import dimensional as D
    D.DimExpr.all_even = lambda self: True


def load_prior(spec):
    """登记表名字，或上游 Prior/ 下 .dat 的绝对路径。"""
    if spec.endswith('.dat'):
        from fit_prior import read_prior_par
        return read_prior_par(spec)
    return get_prior_par(spec)


def run_chain(task):
    cond, data_kind, chain_id, args = task
    x, y = U.make_data(data_kind)
    seed = args.seed0 + chain_id
    random.seed(seed)
    np.random.seed(seed)

    t = Tree(ops=OPS, variables=['L'], parameters=['a', 'b'],
             x=x, y=y,
             prior_par=load_prior(args.prior),
             max_size=args.max_size,
             dimensions=(L_DIM if CONDITIONS[cond]['constitution'] else None),
             constitutions=([DimensionalConstitution()]
                            if CONDITIONS[cond]['constitution'] else []))
    if U._real(t.E) != U._real(t.E):
        raise ValueError('non-finite initial energy')

    t_start = time.time()
    rows = []
    crashed = 0

    def step():
        nonlocal crashed
        try:
            t.mcmc_step()
        except Exception:
            crashed += 1

    for i in range(args.burnin):
        step()
    for s in range(args.samples):
        for _ in range(args.thin):
            step()
        form = str(t)
        try:
            pars = dict(t.par_values['d0'])
        except Exception:
            pars = {}
        rec = U.powerlaw_fit(form, pars)
        rows.append(dict(sample=s, formula=form,
                         pars={k: float(v) for k, v in pars.items()},
                         E=U._real(t.E), EB=U._real(t.EB), EP=U._real(t.EP),
                         b=(None if rec is None else rec[0]),
                         loga=(None if rec is None else rec[1]),
                         r2=(None if rec is None else rec[2]),
                         nops=int(sum(t.nops.values()))))

    with open(OUT[0] / ('trace_%s_%s_%03d.jsonl' % (cond, data_kind, chain_id)), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return dict(cond=cond, data=data_kind, chain=chain_id,
                wall=time.time() - t_start, crashed=crashed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', default=','.join(CONDITIONS))
    ap.add_argument('--data', default='normal,coarse,planets')
    ap.add_argument('--chains', type=int, default=12)
    ap.add_argument('--burnin', type=int, default=U.BURNIN)
    ap.add_argument('--samples', type=int, default=U.SAMPLES)
    ap.add_argument('--thin', type=int, default=U.THIN)
    ap.add_argument('--max-size', type=int, default=U.MAX_SIZE)
    ap.add_argument('--prior', default='bms_wiki_nv13_np13')
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--rundir', default=str(RUNROOT))
    args = ap.parse_args()
    args.seed0 = 7000 + args.seed_offset
    OUT[0] = Path(args.rundir); OUT[0].mkdir(parents=True, exist_ok=True)

    conds = [c for c in args.conditions.split(',') if c]
    datas = [d for d in args.data.split(',') if d]

    if not args.analyze_only:
        from multiprocessing import Pool
        tasks = [(c, d, i, args) for c in conds for d in datas
                 for i in range(args.chains)
                 if not (args.skip_existing and
                         U.load_chain(str(OUT[0]), c, d, i) is not None)]
        print('>> %d core chains (single temperature, prior=%s)' % (
            len(tasks), args.prior), flush=True)
        t0 = time.time()
        with Pool(processes=args.jobs) as pool:
            for i, m in enumerate(pool.imap_unordered(run_chain, tasks)):
                print('  [%d/%d] %s/%s chain %02d %.0fs' % (
                    i + 1, len(tasks), m['cond'], m['data'], m['chain'],
                    m['wall']), flush=True)
        print('>> total %.0fs' % (time.time() - t0), flush=True)

    sums = []
    for cond in conds:
        for dk in datas:
            s = U.summarize(cond, dk, str(OUT[0]))
            if s:
                sums.append(s)
    U.write_report(sums, OUT[0] / 'report_core.md')
    with open(OUT[0] / 'summary_core.json', 'w') as f:
        json.dump(sums, f, indent=1, default=str)
    for s in sums:
        print('%-14s %-8s chains=%2d occ=%5.1f%% b_med=%s MAP命中=%d/%d 曾命中=%d/%d mixing=%.2f' % (
            s['cond'], s['data'], s['chains'], 100 * s['occ_powerlaw'],
            ('%.4f' % s['b_median']) if s['b_median'] is not None else '—',
            s['map_hit'], s['chains'], s['ever_hit'], s['chains'], s['mixing_mean']))


if __name__ == '__main__':
    main()
