"""单链诊断：量纲宪法开启后，MCMC 还剩多少移动性、幂律式能否进入链条。

对同一份"只有 L 和 T"的数据跑两条本地 core 单温度链：
  * constitution 开启（现状）
  * constitution 关闭（其余一字不动）
统计三件事：公式变化率（mixing）、被宪法拒绝的提议次数、以及采样过程中
是否出现过任何属于幂律族（log-log 拟合 R^2 高）的公式。
"""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys, json, random, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))

import upstream_lT_powerlaw as U
from core import Tree, OPS
from experts.constitution.dimensional import DimensionalConstitution
from prior.registry import get_prior_par

L_DIM = {'L': (1, 0, 0, 0, 0, 0, 0)}
STEPS = int(os.environ.get('DIAG_STEPS', 3000))


def chain(with_const, data_kind='normal', seed=7000):
    x, y = U.make_data(data_kind)
    random.seed(seed); np.random.seed(seed)
    t = Tree(ops=OPS, variables=['L'], parameters=['a', 'b'], x=x, y=y,
             prior_par=get_prior_par('bms_wiki_nv13_np13'), max_size=U.MAX_SIZE,
             dimensions=(L_DIM if with_const else None),
             constitutions=([DimensionalConstitution()] if with_const else []))
    prev, change, recs = str(t), 0, []
    for i in range(STEPS):
        t.mcmc_step()
        if str(t) != prev:
            change += 1
            prev = str(t)
        if i % 10 == 0:
            try:
                pars = dict(t.par_values['d0'])
            except Exception:
                pars = {}
            recs.append((str(t), U.powerlaw_fit(str(t), pars), float(t.E)))
    fam = [r for r in recs if r[1] is not None and r[1][2] >= U.R2_POWERLAW]
    uniq = sorted({r[0] for r in recs})
    has_root_or_pow = sorted({r[0] for r in uniq
                             if 'sqrt' in r[0] or '**' in r[0]})
    return dict(const=with_const, data=data_kind, mixing=change / STEPS,
                n_uniq=len(uniq), occ_family=len(fam) / len(recs),
                b_family=sorted({round(r[1][0], 4) for r in fam}),
                sqrt_pow_forms=has_root_or_pow[:6],
                final=recs[-1][0], final_E=recs[-1][2])


if __name__ == '__main__':
    out = []
    for dk in ['normal', 'coarse']:
        for c in [True, False]:
            r = chain(c, dk)
            out.append(r)
            print(json.dumps(r, ensure_ascii=False, indent=1), flush=True)
    (HERE / 'upstream_runs' / 'diag_constitution.json').write_text(
        json.dumps(out, indent=1))
