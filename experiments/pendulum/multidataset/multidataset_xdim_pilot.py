"""对照试跑：给 X 补上量纲声明后，问题 (b) 还成不成立？

`multidataset_planets.py` 的 ``DIMS`` 只声明了 L 的量纲，而三格里的
md_candG / md_junkX 都把 X 当变量喂进去。量纲宪法
(``experts/constitution/dimensional.py``) 对未声明量纲的叶子给出结构性失败，
于是任何含 X 的表达式在提议阶段就被判 ``dE=+inf``、接受概率恒为 0。本脚本只改
这一处（按各格的物理含义声明 X 的量纲），其余数据、先验、采样预算与正式跑
逐位一致，用来分离"闸门堵死"与"BIC 不奖励"两层原因。

    md_candG : X 是该星球的 g，量纲 L·T^-2 = (1,0,-2,0,0,0,0)
    md_junkX : X 是一个随机数，无量纲 (0,0,0,0,0,0,0)

产物写在独立目录，不触碰正式跑量的 trace。
"""

import sys
import json
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from multiprocessing import Pool

from core import Tree, Parallel
from experts.constitution.dimensional import DimensionalConstitution
from prior.registry import get_prior_par, DEFAULT_PRIOR

from multidataset_planets import (DIMS, SCENARIOS, make_frames, ladder,
                                  analyze, print_results, RUNROOT)

XDIMS = {
    'md_candG': (1, 0, -2, 0, 0, 0, 0),
    'md_junkX': (0, 0, 0, 0, 0, 0, 0),
}


def dims_for(scenario):
    d = dict(DIMS)
    if scenario in XDIMS:
        d['X'] = XDIMS[scenario]
    return d


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
    sampler = Parallel(cfg['Ts'],
                       variables=SCENARIOS[scenario]['variables'],
                       parameters=['a'],
                       dimensions=dims_for(scenario),
                       constitutions=[DimensionalConstitution()],
                       x=x, y=y, prior_par=prior_par,
                       max_size=cfg['max_size'])
    sampler.mcmc(tracefn=trace_fn, progressfn=progress_fn,
                 burnin=cfg['burnin'], thin=cfg['thin'],
                 samples=cfg['samples'], progress=False)
    return {'trace_fn': trace_fn, 'scenario': scenario, 'chain_id': chain_id}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='X 量纲声明对照试跑')
    p.add_argument('--prior', default=DEFAULT_PRIOR)
    p.add_argument('--burnin', type=int, default=4000)
    p.add_argument('--thin', type=int, default=10)
    p.add_argument('--samples', type=int, default=400)
    p.add_argument('--chains', type=int, default=8)
    p.add_argument('--max-size', type=int, default=50)
    p.add_argument('--tempering', type=int, default=20)
    p.add_argument('--scenarios', default='md_candG,md_junkX')
    p.add_argument('--seed-offset', type=int, default=90000)
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--analyze-only', action='store_true')
    p.add_argument('--rundir', default=str(RUNROOT / 'xdim_pilot'))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prior_par = dict(get_prior_par(args.prior))
    rundir = Path(args.rundir)
    rundir.mkdir(parents=True, exist_ok=True)
    scen = [s for s in args.scenarios.split(',') if s in XDIMS]
    cfg = {'burnin': args.burnin, 'thin': args.thin, 'samples': args.samples,
           'max_size': args.max_size, 'Ts': ladder(args.tempering)}

    print('=' * 74)
    print('X 量纲声明对照试跑（其余设定与正式跑量逐位一致）')
    print('=' * 74)
    for s in scen:
        print(f'  {s:10s} dims = {dims_for(s)}')

    if not args.analyze_only:
        tasks = []
        for k, sc in enumerate(scen):
            for i in range(args.chains):
                tasks.append((args.seed_offset + 1000 * k + i, sc, i,
                              prior_par, str(rundir), cfg))
        import time
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
        json.dump({'prior': args.prior, 'config': cfg, 'xdims': XDIMS,
                   'note': 'pilot: only dimensions[X] added vs official run',
                   'results': res}, f, indent=2, default=str)
    print(f'\n汇总已保存: {rundir / "summary.json"}')


if __name__ == '__main__':
    main()
