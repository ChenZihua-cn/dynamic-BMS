"""逐链能量迹线，版式照上游仓库 README 的那张 trace 图。

上游配方（upstream-bms/README.md，Images/output_17_0.png）：一张宽图，x 为
MCMC 步、y 为该链走过的能量、标题写这条链到此刻为止的 MDL 表达式。上游一次只
画一条链；这里每条链占一个小面板，同一个格（条件）的链并到一张图里。

y 轴画的是 `E = BIC/2 + 结构先验项`（core/energy.py:164 的 get_energy 返回值），
也就是迹线第 3 列。上游那张图把同一个量标成 "Description length"，而它的 E 也
是 EB + EP（EB = bic/2），所以这个量并不等于描述长度；这里按实际含义标注。

输入：
  - 正式跑量  temp/md_pt20/multidataset_pt20/trace_<格>_<链>.dat
  - 反事实试跑 multidataset_runs/xdim_pilot/trace_<格>_<链>.dat
输出（本目录 multidataset_report/）：
  - fig8_trace_md_Lonly.png / fig9_trace_md_candG.png / fig10_trace_md_junkX.png
  - fig11_trace_xdim_pilot.png  两格并一张，试跑只有 8 链/格

只读迹线，不重跑采样，也不改报告里已有的图。
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUTDIR = HERE / 'multidataset_report'

OFFICIAL = REPO / 'temp' / 'md_pt20' / 'multidataset_pt20'
PILOT = HERE / 'multidataset_runs' / 'xdim_pilot'

# 与 multidataset_report_figs.py 保持一致的配色
COLORS = {'md_Lonly': 'tab:blue', 'md_candG': 'tab:orange', 'md_junkX': 'tab:green'}

TRACE_RE = re.compile(r'^trace_(?P<scenario>.+)_(?P<chain>\d+)\.dat$')


def read_trace(path):
    """一条链的迹线 -> (E 数组, 该链 MDL 的 (E, 公式))。

    迹线行布局见 core/parallel.py 的写盘段：[sample, bic, E, "(E,EB,EP)", 公式, 常数]，
    回火与单温度同布局（只写 BT=1 那条链）。
    """
    energy, best_E, best_f = [], np.inf, None
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            E, formula = float(rec[2]), rec[4]
            energy.append(E)
            if E < best_E:
                best_E, best_f = E, formula
    return np.asarray(energy), (best_E, best_f)


def load_dir(tracedir):
    """目录下所有迹线按格分组，每组内按链号排序。"""
    cells = {}
    for p in sorted(Path(tracedir).glob('trace_*.dat')):
        m = TRACE_RE.match(p.name)
        if not m:
            continue
        cells.setdefault(m.group('scenario'), []).append(
            (int(m.group('chain')), p))
    return {sc: sorted(v) for sc, v in cells.items()}


def wrap_formula(formula, width=26):
    """面板窄，长公式折行。"""
    words = re.split(r'(?<=[+\-*/(),])', formula)
    lines, cur = [], ''
    for w in words:
        if len(cur) + len(w) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur += w
    lines.append(cur)
    return '\n'.join(lines)


def plot_cell(chains, ncols, figsize_per_row, out, ncols_labels=None):
    n = len(chains)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, figsize_per_row * nrows),
                             squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis('off')
            continue
        label, path = chains[k]
        e, (mdl_E, mdl_f) = read_trace(path)
        sc = TRACE_RE.match(path.name).group('scenario')
        color = COLORS.get(sc, 'tab:gray')
        ax.plot(np.arange(len(e)), e, lw=0.8, color=color)
        ax.axhline(mdl_E, ls='--', lw=0.6, color='k', alpha=0.55)
        head = ('chain %d · %s' % (label, sc)) if ncols_labels else ('chain %d' % label)
        ax.set_title('%s · MDL E = %.2f · n = %d\n%s'
                     % (head, mdl_E, len(e), wrap_formula(mdl_f or '?')),
                     fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
        if not k % ncols:
            ax.set_ylabel('E', fontsize=7)
    for ax_row in axes:
        ax_row[0].set_xlabel('post-burnin sample index', fontsize=7)
    fig.suptitle(out.stem.replace('_', ' '), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--tracedir', default=str(OFFICIAL),
                    help='正式跑量迹线目录')
    ap.add_argument('--pilotdir', default=str(PILOT),
                    help='反事实试跑迹线目录')
    ap.add_argument('--outdir', default=str(OUTDIR))
    ap.add_argument('--no-pilot', action='store_true')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    order = [sc for sc in ('md_Lonly', 'md_candG', 'md_junkX')]
    cells = load_dir(args.tracedir)
    for i, sc in enumerate([s for s in order if s in cells] +
                           sorted(set(cells) - set(order))):
        fig_no = 8 + i
        out = outdir / f'fig{fig_no}_trace_{sc}.png'
        plot_cell(cells[sc], ncols=4, figsize_per_row=2.4, out=out)
        print(f'{out.name}: {len(cells[sc])} 链 -> {out}')

    if not args.no_pilot and Path(args.pilotdir).is_dir():
        pc = load_dir(args.pilotdir)
        if pc:
            merged = [(sc, c, p) for sc in sorted(pc) for c, p in pc[sc]]
            # 试跑每格 8 链、两格，排成 8 行 x 2 列，列按格分组
            per = max(len(v) for v in pc.values())
            grid = []
            for row in range(per):
                for sc in sorted(pc):
                    if row < len(pc[sc]):
                        grid.append(pc[sc][row])
            out = outdir / 'fig11_trace_xdim_pilot.png'
            plot_cell(grid, ncols=len(pc), figsize_per_row=2.4, out=out,
                      ncols_labels=True)
            print(f'{out.name}: {len(grid)} 链（{", ".join(sorted(pc))}）-> {out}')


if __name__ == '__main__':
    main()
