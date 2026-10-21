"""迹线的严格判据：这条公式真的**是**幂律，还是只在测量区间里**像**幂律？

`upstream_lT_powerlaw.powerlaw_fit` 只在数据覆盖的 L∈[0.15, 2.5] m 上做
log-log 回归。`log(a*L+b)**c` 这类式子在那段区间里 R^2 也能超过 0.999，
于是会被误当成"还原了幂律"。本脚本把同一批公式放到 0.02–50 m（近三个数量级）
上重测，用两项指标区分：

* `b_wide`   —— 宽域 log-log 斜率（应当仍是 0.5）
* `dev`      —— 宽域上 |f(L)/ (A*L**b) - 1| 的最大值：真单项式恒为 0，
                区间内伪装的 log/加和式会发散

`dev < 1%` 才算"形式上是幂律"（定性成立）；`|b-0.5|` 给半定量精度。
"""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import upstream_lT_powerlaw as U

L_NARROW = np.linspace(U.L_RANGE[0], U.L_RANGE[1], 80)
L_WIDE = np.exp(np.linspace(np.log(0.02), np.log(50.0), 400))


def evaluate(formula, pars):
    fn = U._callable_for(formula, pars)
    if fn is None:
        return None, None
    with np.errstate(all='ignore'):
        try:
            v = np.asarray(fn(L_WIDE), dtype=float)
        except Exception:
            return None, None
    if v.shape != L_WIDE.shape or not np.all(np.isfinite(v)) or np.any(v <= 0):
        return None, None
    return v, None


def classify(formula, pars):
    """返回 dict(b, dev, b_narrow, r2_narrow) 或 None。"""
    v, _ = evaluate(formula, pars)
    if v is None:
        return None
    lx, ly = np.log(L_WIDE), np.log(v)
    b, c = np.polyfit(lx, ly, 1)
    mono = np.exp(c) * L_WIDE ** b
    dev = float(np.max(np.abs(v / mono - 1.0)))
    # 数据区间内的常规判据（与 upstream 脚本一致）
    try:
        vn = np.asarray(U._callable_for(formula, pars)(L_NARROW), dtype=float)
    except Exception:
        return dict(b=float(b), dev=dev, b_narrow=None, r2_narrow=None)
    if not np.all(np.isfinite(vn)) or np.any(vn <= 0):
        return dict(b=float(b), dev=dev, b_narrow=None, r2_narrow=None)
    lxn, lyn = np.log(L_NARROW), np.log(vn)
    bn, cn = np.polyfit(lxn, lyn, 1)
    pred = bn * lxn + cn
    ss_res = float(np.sum((lyn - pred) ** 2))
    ss_tot = float(np.sum((lyn - lyn.mean()) ** 2))
    r2n = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return dict(b=float(b), dev=dev, b_narrow=float(bn), r2_narrow=float(r2n))


DEV_MONO = 0.01


def analyze(cond, data, rundir, tol):
    cid, out = 0, []
    while True:
        rows = U.load_chain(rundir, cond, data, cid)
        if rows is None:
            break
        cls = []
        for r in rows:
            c = classify(r['formula'], r['pars']) if r['r2'] is not None else None
            cls.append(c)
        good = [(r, c) for r, c in zip(rows, cls)
                if c and c['dev'] < DEV_MONO and c['r2_narrow'] is not None
                and c['r2_narrow'] >= U.R2_POWERLAW]
        bs = np.array([c['b'] for _, c in good]) if good else np.array([])
        best = min(rows, key=lambda r: U._real(r['E']))
        cbest = classify(best['formula'], best['pars'])
        hit_map = bool(cbest and cbest['dev'] < DEV_MONO
                       and abs(cbest['b'] - 0.5) <= tol)
        hit_ever = any(abs(c['b'] - 0.5) <= tol for _, c in good)
        changed = sum(1 for a, b_ in zip(rows[:-1], rows[1:])
                      if a['formula'] != b_['formula'])
        out.append(dict(chain=cid, n=len(rows), occ=len(good) / len(rows),
                        b_med=(float(np.median(bs)) if bs.size else None),
                        b_iqr=([float(np.percentile(bs, 25)),
                                float(np.percentile(bs, 75))] if bs.size >= 4 else None),
                        map_formula=best['formula'], map_b=(cbest or {}).get('b'),
                        map_dev=(cbest or {}).get('dev'), hit_map=hit_map,
                        hit_ever=hit_ever, mixing=changed / max(1, len(rows) - 1)))
        cid += 1
    if not out:
        return None
    return dict(cond=cond, data=data, chains=len(out),
                occ=float(np.mean([o['occ'] for o in out])),
                b_med=float(np.median([o['b_med'] for o in out if o['b_med'] is not None]))
                if any(o['b_med'] is not None for o in out) else None,
                hit_map=sum(o['hit_map'] for o in out),
                hit_ever=sum(o['hit_ever'] for o in out),
                mixing=float(np.mean([o['mixing'] for o in out])),
                per_chain=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', default=str(HERE / 'upstream_runs'))
    ap.add_argument('--conds', default='pt_nv1np2,pt_nv13np13,pt_flat,singleT_nv1np2,'
                                       'core_const,core_noconst')
    ap.add_argument('--datas', default='normal,coarse,planets')
    args = ap.parse_args()
    rows = []
    for cond in args.conds.split(','):
        for data in args.datas.split(','):
            tol = U.TOL_B[data]
            r = analyze(cond, data, args.rundir, tol)
            if r:
                r['tol'] = tol
                rows.append(r)
    print('| 条件 | 数据 | 链 | 真单项式占有率 | b^ 中位 | MAP 命中 | 曾命中 | mixing |')
    print('|---|---|---|---|---|---|---|---|')
    for r in rows:
        print('| %s | %s | %d | %.1f%% | %s | %d/%d | %d/%d | %.3f |' % (
            r['cond'], r['data'], r['chains'], 100 * r['occ'],
            ('%.4f' % r['b_med']) if r['b_med'] is not None else '—',
            r['hit_map'], r['chains'], r['hit_ever'], r['chains'], r['mixing']))
    (Path(args.rundir) / 'strict_report.md').write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))
    print('\nMAP 公式抽样：')
    for r in rows:
        pc = r['per_chain'][0]
        print('  %-16s %-8s `%s`  b=%s dev=%s' % (
            r['cond'], r['data'], pc['map_formula'],
            ('%.4f' % pc['map_b']) if pc['map_b'] is not None else '—',
            ('%.3f' % pc['map_dev']) if pc['map_dev'] is not None else '—'))


if __name__ == '__main__':
    main()
