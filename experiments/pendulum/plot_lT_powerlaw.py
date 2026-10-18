"""画钟摆幂律实验的三栏图：数据与还原式 / b^ 分布 / mixing 对照。"""

import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
import json
import glob
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import upstream_lT_powerlaw as U
import analyze_powerlaw as A

L = np.linspace(0.05, 3.0, 200)
CELLS = [
    ('上游 并行回火 (nv1.np2)', 'upstream_runs', 'pt_nv1np2'),
    ('上游 并行回火 (nv13.np13)', 'upstream_runs', 'pt_nv13np13'),
    ('上游 单温度 (nv1.np2)', 'upstream_runs', 'singleT_nv1np2'),
    ('上游 单温度 (nv13.np13)', 'upstream_runs', 'singleT_nv13np13'),
    ('本地 core+宪法 (nv1.np2)', 'upstream_runs/v2', 'core_const'),
    ('本地 core+宪法 (nv13.np13)', 'upstream_runs/v3_paper', 'core_const'),
]


def tdir(root):
    return str((HERE / root).resolve())


def cell_stats(root, cond):
    hits, bs, mix, maps = [], [], [], []
    for f in sorted(glob.glob('%s/trace_%s_coarse_*.jsonl' % (tdir(root), cond))):
        rows = [json.loads(l) for l in open(f)]
        best = min(rows, key=lambda r: U._real(r['E']))
        c = A.classify(best['formula'], best['pars'])
        ok = bool(c and c['dev'] < 0.01 and abs(c['b'] - 0.5) <= U.TOL_B['coarse'])
        hits.append(ok)
        maps.append(best['formula'])
        if ok:
            bs.append(c['b'])
        mix.append(sum(1 for a, b in zip(rows[:-1], rows[1:])
                       if a['formula'] != b['formula']) / max(1, len(rows) - 1))
    return hits, bs, mix, maps


fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

# 面板 1：数据 + 真律 + 各条件还原出的 MAP 曲线
x, y = U.make_data('coarse')
ax[0].scatter(x['L'], y, s=18, c='0.45', label='测量值（0.1 s 秒表）', zorder=3)
ax[0].plot(L, 2 * np.pi * np.sqrt(L / 9.81), 'k-', lw=1.6, label='真实律 T=2π√(L/g)')
colors = plt.cm.tab10(np.linspace(0, 0.9, len(CELLS)))
for i, (label, root, cond) in enumerate(CELLS):
    fs = sorted(glob.glob('%s/trace_%s_coarse_*.jsonl' % (tdir(root), cond)))
    if not fs:
        continue
    rows = [json.loads(l) for l in open(fs[0])]
    best = min(rows, key=lambda r: U._real(r['E']))
    fn = U._callable_for(best['formula'], best['pars'])
    if fn is None:
        continue
    with np.errstate(all='ignore'):
        v = np.asarray(fn(L), dtype=float)
    if np.all(np.isfinite(v)):
        ax[0].plot(L, v, '--', color=colors[i], lw=1.2,
                   label='%s: %s' % (label, best['formula']))
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('L (m)'); ax[0].set_ylabel('T (s)')
ax[0].set_title('只用绳长与周期：还原出的曲线')
ax[0].legend(fontsize=7, loc='upper left')

# 面板 2：各条件 MAP 的幂律指数
allb = []
labels = []
for label, root, cond in CELLS:
    hits, bs, mix, maps = cell_stats(root, cond)
    if not bs:
        bs = [np.nan]
    allb.append(bs)
    labels.append('%s\n%d/%d' % (label, sum(hits), len(hits)))
pos = np.arange(len(allb))
for i, bs in enumerate(allb):
    ax[1].scatter(np.full(len(bs), i), bs, s=26, color=colors[i])
ax[1].axhline(0.5, color='k', lw=1, ls=':')
ax[1].set_xticks(pos)
ax[1].set_xticklabels(labels, fontsize=7, rotation=20, ha='right')
ax[1].set_ylabel('MAP 的幂律指数 b')
ax[1].set_ylim(0, 1)
ax[1].set_title('指数还原：真值 0.5')

# 面板 3：mixing
mixv, lab2 = [], []
for label, root, cond in CELLS:
    hits, bs, mix, maps = cell_stats(root, cond)
    mixv.append(np.mean(mix) if mix else 0)
    lab2.append(label)
ax[2].barh(range(len(mixv)), mixv, color=colors[:len(mixv)])
ax[2].set_yticks(range(len(mixv)))
ax[2].set_yticklabels(lab2, fontsize=7)
ax[2].set_xlabel('相邻样本公式变化率（0.1 s 数据）')
ax[2].set_title('混合速度：单温度 + 论文先验被冻住')
for i, v in enumerate(mixv):
    ax[2].text(v + 0.008, i, '%.2f' % v, va='center', fontsize=7)

fig.tight_layout()
out = HERE / 'pendulum_lT_powerlaw.png'
fig.savefig(out, dpi=130)
print('wrote', out)
