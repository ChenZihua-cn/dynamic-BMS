"""pendulum 迹线的规范形转移网络图：每个条件一张单图，含试验条件说明。

输入（只读迹线，不重跑采样，也不改任何实验设定）：
  - known_g/pendulum_runs/bms_wiki_nv13_np13/         单温度，先验 bms_wiki_nv13_np13
  - known_g/pendulum_runs/bms_wiki_nv13_np13_pt20/    20 温度回火，同一份先验
  - temp/md_pt20/multidataset_pt20/                   多数据集三格（一条公式共享 5 份）
  - lT_powerlaw/upstream_runs/trace_<配置>_*.jsonl    只给 L 的幂律实验（上游快照代码）

用法：``.venv/bin/python experiments/pendulum/trace_graph_figs.py [pt20|singleT|md|lT|all]``

输出 trace_graph_figs/：
  - fig_tracegraph_<来源>_<条件>.png（300 dpi）
  - captions.md  每张图的图注与试验条件（跑 all 时写全）

节点 = 规范形（与 core/tree_base.py:214 TreeBase.canonical 同一口径：sympy 化简 +
自由常数按出现顺序改名 c1、c2…，所以结构不同但化简相同的树算同一个节点）。
圆面积 ∝ 该规范形被访问的次数（跨链合计；半径按 sqrt 缩放，面积才与次数成正比）。
箭头 = 相邻两个保留样本之间的结构转移，粗细与深浅 ∝ 次数。

口径限制：每两个保留样本之间隔着 thin 步（回火还叠加每步一次的副本交换），所以边是
粗粒度过渡而非单次提议；被拒绝的提议与被宪法硬拒的候选不出现在图上。要拿真正的单次
转移（带提议类型）需要在 mcmc_step 里另记一份 moves 日志。
"""

import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from sympy import sympify

matplotlib.font_manager.fontManager.addfont(
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({
    'font.family': ['Noto Sans CJK SC', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'savefig.bbox': 'tight',
})

BMS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BMS))
RUNS = BMS / 'experiments/pendulum/known_g/pendulum_runs'
LT_RUNS = BMS / 'experiments/pendulum/lT_powerlaw/upstream_runs'
MD_RUNS = BMS / 'temp/md_pt20/multidataset_pt20'
OUT = BMS / 'experiments/pendulum/trace_graph_figs'


# ---------------------------------------------------------------------------
# 规范形与统计
# ---------------------------------------------------------------------------
def canonical(formula):
    """与 TreeBase.canonical 同一口径。"""
    try:
        cansp = sympify(formula.replace(' ', ''))
        can = str(cansp)
        positions = []
        for p in [str(s) for s in cansp.free_symbols]:
            if p.startswith('_') and p.endswith('_'):
                positions.append((can.find(p), p))
        positions.sort()
        for i, (_, p) in enumerate(positions, start=1):
            can = can.replace(p, 'c%d' % i)
    except Exception:
        can = formula
    return can.replace(' ', '')


def family_members(order, kind, g_constant=False, tol_b=0.02, r2_min=0.999, xvals=None):
    """真公式族成员，数值判定，与各实验自己的命中判据同口径。

    sqrt 族（known_g）：在 L×g 网格上 expr/√(L/g) 必须是同一常数。g 是常数列的格
      （对照 B/C）只在该值上取值，√g 于是被拟合常数吸收。
    power 族（lT）：按 expr = a·L^b 拟合，|b − 0.5| ≤ 0.02 且 R² ≥ 0.999。
    multi 族（multidataset）：对每一份数据（X 取该份的恒定值）expr/√L 各自是常数。
      注意它只要求"每份内成正比"，不要求那份的常数跨份统一——这正是 md_candG 与
      md_junkX 在目标函数上分不开的原因（k 不随份数增长）。
    """
    from sympy import lambdify, symbols
    Ls, gs = symbols('L'), symbols('g')
    Xs = symbols('X')
    Lv = np.linspace(0.2, 3.0, 14)
    # g 恒为 9.81 的格（对照 B/C）只在该值上取值，√g 于是被拟合常数吸收
    Gv = np.array([9.81]) if g_constant else np.linspace(1.62, 24.79, 6)
    LL, GG = np.meshgrid(Lv, Gv, indexing='ij')
    out = set()
    for can in order:
        try:
            ex = sympify(can)
            consts = sorted([s for s in ex.free_symbols if str(s).startswith('c')], key=str)
            cvals = [1.7 ** (i + 1) for i in range(len(consts))]
            # 个别表达式（嵌套 log/Abs）会让 lambdify 往 stdout 打代码片段，这里屏蔽掉
            with contextlib.redirect_stdout(io.StringIO()), np.errstate(all='ignore'):
                if kind == 'sqrt':
                    f = lambdify([Ls, gs] + consts, ex, 'numpy')
                    vals = np.vectorize(lambda a, b: f(a, b, *cvals))(LL, GG)
                    ratio = vals / np.sqrt(LL / GG)
                    if np.all(np.isfinite(ratio)) and ratio.mean() != 0:
                        if np.std(ratio) / abs(ratio.mean()) < 1e-8:
                            out.add(can)
                elif kind == 'multi':
                    has_X = any(str(s) == 'X' for s in ex.free_symbols)
                    f = lambdify(([Ls, Xs] if has_X else [Ls]) + consts, ex, 'numpy')
                    good = True
                    for xv in (xvals if has_X else [None]):
                        v = f(Lv, xv, *cvals) if has_X else f(Lv, *cvals)
                        ratio = v / np.sqrt(Lv)
                        if not (np.all(np.isfinite(ratio)) and ratio.mean() != 0
                                and np.std(ratio) / abs(ratio.mean()) < 1e-8):
                            good = False
                            break
                    if good:
                        out.add(can)
                else:
                    f = lambdify([Ls] + consts, ex, 'numpy')
                    vals = f(Lv, *cvals)
                    ok = np.isfinite(vals) & (vals > 0)
                    if ok.sum() >= 6:
                        b, a = np.polyfit(np.log(Lv[ok]), np.log(vals[ok]), 1)
                        pred = a + b * np.log(Lv[ok])
                        r2 = 1 - np.sum((np.log(vals[ok]) - pred) ** 2) / \
                            np.sum((np.log(vals[ok]) - np.log(vals[ok]).mean()) ** 2)
                        if abs(b - 0.5) <= tol_b and r2 >= r2_min:
                            out.add(can)
        except Exception:
            pass
    return out


def load_known_g(run_dir, tag):
    """trace_<tag>_<链>.dat，字段 [s, bic, E, 能量三元组, 公式, 常数]"""
    fn = Path(run_dir)
    return [[(canonical(r[4]), r[2]) for r in map(json.loads, open(p))]
            for p in sorted(fn.glob(f'trace_{tag}_*.dat'))]


def load_md(tag, samples=400):
    """多数据集那批：同样的字段布局，但每格有 1 条链被中断写满 400 个样本。

    返回 (链, 被剔除的链号)——半条链会让"次数/链"两列把中断链当成完整链计。
    """
    chains, dropped = [], []
    for p in sorted(Path(MD_RUNS).glob(f'trace_{tag}_*.dat')):
        recs = [(canonical(r[4]), r[2]) for r in map(json.loads, open(p))]
        if len(recs) < samples:
            dropped.append((int(p.stem.rsplit('_', 1)[1]), len(recs)))
        else:
            chains.append(recs)
    return chains, dropped


def load_lT(config):
    """trace_<config>_<NNN>.jsonl，字段 formula / E"""
    return [[(canonical(r['formula']), r['E']) for r in map(json.loads, open(p))]
            for p in sorted(LT_RUNS.glob(f'trace_{config}_*.jsonl'))]


def aggregate(chains):
    visits, nchains, esum, edges, loops = {}, {}, {}, {}, {}
    tot = pairs = same = 0
    for recs in chains:
        seen, prev = set(), None
        for c, e in recs:
            visits[c] = visits.get(c, 0) + 1
            esum[c] = esum.get(c, 0.0) + e
            seen.add(c)
            tot += 1
            if prev is not None:
                pairs += 1
                if c == prev:
                    same += 1
                    loops[c] = loops.get(c, 0) + 1
                else:
                    edges[(prev, c)] = edges.get((prev, c), 0) + 1
            prev = c
        for c in seen:
            nchains[c] = nchains.get(c, 0) + 1
    return dict(visits=visits, nchains=nchains, esum=esum, edges=edges,
                loops=loops, tot=tot, pairs=pairs, same=same)


def pick_nodes(agg, max_nodes=12, min_keep=6):
    """访问量前 K：先用 1% 门槛，节点太少就逐档放宽到 0.15%。"""
    order = sorted(agg['visits'], key=lambda c: -agg['visits'][c])
    tot = agg['tot']
    out = []
    for floor in (0.01, 0.005, 0.0025, 0.0015):
        out = [c for c in order if agg['visits'][c] >= floor * tot][:max_nodes]
        if len(out) >= min_keep:
            return out
    return out


# ---------------------------------------------------------------------------
# 布局：确定性力导向 + 防重叠（不依赖 networkx）
# ---------------------------------------------------------------------------
def layout(order, elist, iters=600, seed=11):
    m = len(order)
    idx = {n: i for i, n in enumerate(order)}
    pos = np.random.default_rng(seed).uniform(-1, 1, size=(m, 2))
    if m == 1:
        return {order[0]: np.zeros(2)}
    if m == 2:
        return {order[0]: np.array([-0.5, 0.]), order[1]: np.array([0.5, 0.])}
    k = np.sqrt(4.0 / m)
    emax = max([w for _, _, w in elist] + [1.0])
    eidx = [(idx[a], idx[b], w) for a, b, w in elist]
    for it in range(iters):
        t = 0.20 * (0.004 / 0.20) ** (it / (iters - 1))
        disp = np.zeros_like(pos)
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-9
        np.fill_diagonal(dist, 1e9)
        disp += ((k ** 2 / dist)[..., None] * d / dist[..., None]).sum(axis=1)
        for i, j, w in eidx:
            v = pos[i] - pos[j]
            dd = np.sqrt((v ** 2).sum()) + 1e-9
            u = v / dd * (dd ** 2 / k * np.sqrt(w / emax))
            disp[i] -= u
            disp[j] += u
        l = np.sqrt((disp ** 2).sum(-1)) + 1e-9
        pos += disp / l[..., None] * np.minimum(l, t)[..., None]
        span = pos.max(0) - pos.min(0)
        pos = (pos - pos.mean(0)) / (span.max() / 2.0) * 0.95
    return {n: pos[i] for n, i in idx.items()}


def separate(pos, rad, order, gap=0.50, passes=4000):
    for _ in range(passes):
        moved = False
        for a in range(len(order)):
            for b in range(a + 1, len(order)):
                p, q = pos[order[a]], pos[order[b]]
                need = rad[order[a]] + rad[order[b]] + gap
                v = q - p
                dd = np.hypot(*v)
                if dd < need:
                    u = v / dd if dd > 1e-9 else np.array([1.0, 0.3])
                    p[:] -= u * (need - dd) / 2.0
                    q[:] += u * (need - dd) / 2.0
                    moved = True
        if not moved:
            break
    worst = max(rad[order[a]] + rad[order[b]] + 0.004
                - np.hypot(*(pos[order[b]] - pos[order[a]]))
                for a in range(len(order)) for b in range(a + 1, len(order)))
    if worst > 0:
        print(f'  警告：布局未收敛，最大圆重叠 {worst:.3f}（数据单位）')
    s = 0.92 / max(np.abs(pos[n]).max() + rad[n] for n in order)
    for n in order:
        pos[n] *= s
        rad[n] *= s
    return pos, rad


# ---------------------------------------------------------------------------
# 成图
# ---------------------------------------------------------------------------
def draw_graph(ax, agg, order, min_w, family, rmax=0.14, gap=0.22):
    vis, edges = agg['visits'], agg['edges']
    nodes = set(order)
    elist = [(a, b, w) for (a, b), w in edges.items()
             if a in nodes and b in nodes and w >= min_w]
    pos = layout(order, elist)
    vmax = max(vis[c] for c in order)
    rad = {c: 0.05 + rmax * np.sqrt(vis[c] / vmax) for c in order}
    pos, rad = separate(pos, rad, order, gap=gap)
    emax = max([w for _, _, w in elist] + [1.0])
    mean_e = {c: agg['esum'][c] / vis[c] for c in order}
    lo, hi = min(mean_e.values()), max(mean_e.values())
    span = max(hi - lo, 1e-9)
    cmap = matplotlib.colormaps['viridis']
    eset = set((a, b) for a, b, _ in elist)

    for (a, b, w) in elist:
        p, q = pos[a], pos[b]
        v = q - p
        dd = np.hypot(*v) + 1e-9
        u = v / dd
        curve = (0.20 if a < b else -0.20) if (b, a) in eset else 0.0
        ax.add_patch(FancyArrowPatch(
            tuple(p + u * (rad[a] + 0.012)), tuple(q - u * (rad[b] + 0.022)),
            arrowstyle='-|>', mutation_scale=6.5,
            connectionstyle='arc3,rad=%.3f' % curve,
            color='0.40', alpha=min(0.35 + 0.65 * w / emax, 0.95),
            lw=0.5 + 2.4 * np.sqrt(w / emax), zorder=1))

    num = {c: i for i, c in enumerate(order, start=1)}
    for c in order:
        dark = (mean_e[c] - lo) / span < 0.5
        ax.add_patch(Circle(pos[c], rad[c], facecolor=cmap((mean_e[c] - lo) / span),
                            edgecolor='#c1272d' if c in family else '0.30',
                            linewidth=2.6 if c in family else 0.7, zorder=3))
        ax.text(pos[c][0], pos[c][1], str(num[c]), ha='center', va='center',
                fontsize=max(6.0, min(10.5, rad[c] * 60.0)), fontweight='bold',
                color='w' if dark else '0.12', zorder=4)

    lim = max(np.abs(pos[c]).max() + rad[c] for c in order) + 0.16
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_axis_off()
    return elist, mean_e, num


def _chunks(text):
    """切成不可再分的原子片段：拉丁/数字连串（含 _ / . * 等）整体保留，汉字逐字可断。"""
    return re.findall(r'[A-Za-z0-9_%.*\-/]+\s*|\s+|.', text) or ['']


def _make_measure(fig, fontsize):
    """返回"字符串在该字号下的渲染宽度（pt）"的测量函数。

    TextPath 量的是字形墨迹，空格量不出来、标点也偏窄；这里用渲染器量整行实际占位。
    """
    renderer = fig.canvas.get_renderer()

    def measure(s):
        if not s:
            return 0.0
        t = fig.text(0, 0, s, fontsize=fontsize)
        try:
            return t.get_window_extent(renderer).width * 72.0 / fig.dpi
        finally:
            t.remove()
    return measure


def _wrap_px(text, budget_pt, width_pt, measure):
    """按实测渲染宽度贪心折行——中英混排下字符数不可信。"""
    lines, cur = [], ''
    for ck in _chunks(text):
        cand = cur + ck
        if cur and measure(cand.rstrip()) > budget_pt:
            lines.append(cur.rstrip())
            cur = ck
        else:
            cur = cand
        while cur and measure(cur) > width_pt and len(cur) > 1:   # 单片段超栏宽，只能硬断
            cut = len(cur) - 1
            while cut > 1 and measure(cur[:cut]) > width_pt:
                cut -= 1
            lines.append(cur[:cut])
            cur = cur[cut:]
    if cur.strip():
        lines.append(cur.rstrip())
    return lines or ['']


def make_figure(name, agg, order, min_w, family, ref_can, title, subtitle, cond_lines):
    fig = plt.figure(figsize=(7.2, 8.6))
    ax = fig.add_axes([0.02, 0.44, 0.70, 0.475])
    elist, mean_e, num = draw_graph(ax, agg, order, min_w, family)
    axc = fig.add_axes([0.755, 0.56, 0.022, 0.28])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(
        vmin=min(mean_e.values()), vmax=max(mean_e.values())))
    cb = fig.colorbar(sm, cax=axc)
    cb.set_label('平均能量 E（自然信息单位）', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)

    cov = sum(agg['visits'][c] for c in order) / agg['tot']
    stats = (f'{len(order)} / {len(agg["visits"])} 个规范形入选（覆盖 {cov:.0%} 样本）；'
             f'转移边 {len(elist)} / {len(agg["edges"])} 条（阈值 ≥{min_w} 次）；'
             f'相邻保留样本原地不动 {agg["same"] / max(agg["pairs"], 1):.0%}')
    fam = family & set(order)
    fam_share = sum(agg['visits'][c] for c in fam) / agg['tot']
    stats += (f'；入选节点中 {len(fam)} 个属真公式族，占 {fam_share:.0%} 样本'
              if fam else '；真公式族成员未进入访问量前列，故图中无红圈')
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.985)
    fig.text(0.5, 0.955, subtitle, ha='center', va='center', fontsize=8.6)
    fig.text(0.03, 0.938, stats, fontsize=7.8, ha='left', va='top', color='0.22')

    # 节点表：两组四栏（编号 / 标记 / 规范形 / 次数·链·E），数字右对齐
    axb = fig.add_axes([0.03, 0.035, 0.555, 0.375])
    axb.set_axis_off()
    groups = 2
    per = int(np.ceil(len(order) / groups))
    step = min(0.80 / max(per, 1), 0.115)   # 节点少时不撑出大间距
    gw = 0.505
    y0, xs = 0.895, {'n': 0.0, 'm': 0.038, 'f': 0.068, 'v': 0.478}
    for g in range(groups):
        x0 = g * gw
        axb.text(x0 + xs['n'], 0.945, '编号', fontsize=6.6, ha='left', va='top',
                 fontweight='bold', color='0.25')
        axb.text(x0 + xs['f'], 0.945, '规范形', fontsize=6.6, ha='left', va='top',
                 fontweight='bold', color='0.25')
        axb.text(x0 + xs['v'], 0.945, '次数 / 链 / E', fontsize=6.6, ha='right',
                 va='top', fontweight='bold', color='0.25')
    for i, c in enumerate(order):
        g, r = i // per, i % per
        x0, y = g * gw, y0 - r * step
        axb.text(x0 + xs['n'], y, f'{num[c]}', fontsize=6.6, ha='left', va='top')
        axb.text(x0 + xs['m'], y, '★' if c == ref_can else ('族' if c in family else ''),
                 fontsize=6.6, ha='left', va='top', color='#c1272d')
        axb.text(x0 + xs['f'], y, c, fontsize=6.6, ha='left', va='top')
        axb.text(x0 + xs['v'], y, f'{agg["visits"][c]} / {agg["nchains"][c]} / '
                                  f'{mean_e[c]:.1f}', fontsize=6.6, ha='right', va='top')
    axb.text(0.0, 1.0, '节点（编号即图中圆内数字；次数与链数为跨链合计）',
             fontsize=7.6, ha='left', va='top', fontweight='bold')

    # 试验条件：标签栏 + 正文栏，正文悬挂缩进
    axc2 = fig.add_axes([0.615, 0.035, 0.375, 0.375])
    axc2.set_axis_off()
    axc2.text(0.0, 1.0, '试验条件', fontsize=7.6, ha='left', va='top', fontweight='bold')
    label_x, body_x, line_h, y = 0.0, 0.13, 0.046, 0.925
    col_pt = 0.375 * fig.get_figwidth() * 72.0          # 条件栏总宽（pt）
    budget = (1.0 - body_x) * col_pt - 4.0               # 正文可用宽度
    hard = budget + 6.0                                  # 再宽就得硬断
    measure = _make_measure(fig, 6.3)
    for line in cond_lines:
        head, _, body = line.partition('   ')
        body = body.lstrip()
        for k, seg in enumerate(_wrap_px(body, budget, hard, measure)):
            if k == 0:
                axc2.text(label_x, y, head, fontsize=6.3, ha='left', va='top', color='0.30')
            axc2.text(body_x, y, seg, fontsize=6.3, ha='left', va='top')
            y -= line_h
        y -= 0.004
    y_bot = max(y - line_h * 0.55, 0.0)     # 框只包住写到的地方，不留空底
    axc2.add_patch(FancyBboxPatch((0.0, y_bot), 1.0, 1.0 - y_bot, transform=axc2.transAxes,
                                  boxstyle='round,pad=0.012', fc='0.972', ec='0.62',
                                  lw=0.6, zorder=0))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'fig_tracegraph_{name}.png', dpi=300)
    plt.close(fig)
    return stats
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------------
# 各来源的试验条件（逐条来自 run 目录的 summary.json / 脚本常量，不是估的）
# ---------------------------------------------------------------------------
def known_g_lines(run, tag, nchains, samples, temper):
    data = {'main': '5 颗星球 × 60 个摆长 = 300 行，输入变量 L, g',
            'ctrlA_Lonly': '仅地球 60 行，输入变量只有 L（g 不给）',
            'ctrlB_Lg': '仅地球 60 行，输入变量 L, g（g 恒为 9.81）',
            'ctrlC_constG': '5 颗星球 × 60 行 = 300 行，但 g 一律给 9.81'}[tag]
    return [
        f'数据   {data}',
        '        L∈[0.1,3.0] m 按米尺 0.01 m 舍入，T 按秒表 0.1 s 舍入，随机种子 20261014',
        TARGET[tag],
        '宪法   DimensionalConstitution，拒绝：加减两侧量纲不同；sin/cos/tan/exp/log/'        'sinh/cosh/tanh/fac 的参数带量纲；** 的指数带量纲；未声明量纲也不是参数的叶子',
        '先验   bms_wiki_nv13_np13（[BMS2020] wiki 语料校准，一阶 18 项 + 二阶 1 项），scale = 1',
        f'采样   {temper}，burnin 4000 / thin 10 / 保留 {samples} 样本 / {nchains} 条链 / '
        f'max_size 50',
        '代码   版本 b491f38（工作区含未提交改动；provenance 记录于分析时，'
        '不为采样时的代码态背书）',
        f'来源   pendulum_runs/{run}/trace_{tag}_*.dat',
    ]


LT_LINES_COMMON = [
    '数据   仅地球，40 个摆长 L∈[0.15,2.5] m（0.01 m 舍入）；normal 档 T 按 0.01 s 舍入'
    '（相当于 10 次计时取一），随机种子 20261018',
    '目标   T = a·L^b 且 b ≈ 0.5；输入只有 L，g 从不进入',
    '宪法   无。上游快照的 mcmc.py 里没有量纲检查，只有 max_size 限制树的大小',
    '先验   nv1.np2（[BMS2017] final_prior_param_sq.named_equations.nv1.np2…dat）',
    '采样   burnin 1200 / thin 5 / 保留 250 样本 / 12 条链，max_size = 20',
    '判据   宽域幂律命中：|b̂ − 0.5| ≤ 0.02 且幂律拟合 R² ≥ 0.999',
]


MD_G = [1.62, 3.71, 9.81, 11.15, 24.79]        # Moon / Mars / Earth / Neptune / Jupiter
MD_LABELS = {'md_Lonly': '基准 只给 L', 'md_candG': '候选列 X=该星球 g',
             'md_junkX': '负对照 X=随机数'}
MD_XDESC = {'md_Lonly': '无 X 列，每份只喂 L',
            'md_candG': 'X = 该星球的 g，份内恒定，不告诉它是什么',
            'md_junkX': 'X = 每星球一个随机数，份内恒定'}


def md_lines(tag, nchains, samples, dropped, x_used=0, x_total=0):
    drop_txt = ('无截断链' if not dropped else
                '；'.join(f'链 {i}（只剩 {n}/{samples} 样本）' for i, n in dropped))
    return [
        '数据   5 份独立数据集（Moon 1.62 / Mars 3.71 / Earth 9.81 / Neptune 11.15 /'
        ' Jupiter 24.79），每份 60 个摆长，只喂 L 与 T',
        '        L∈[0.1,3.0] m 按米尺 0.01 m 舍入，T 按秒表 0.1 s 舍入，随机种子 20261018',
        f'X 列   {MD_XDESC[tag]}；迹线里含 X 的样本 {x_used} / {x_total}',
        '目标   一条公式同时贴合 5 份数据，常数按份各自拟合；k = 1 + 参数个数 不随份数'
        '增长，所以"用 X 把 5 个常数塌成 1 个"在 BIC 里没有奖励',
        '宪法   DimensionalConstitution，拒绝：加减两侧量纲不同；sin/cos/tan/exp/log/'
        'sinh/cosh/tanh/fac 的参数带量纲；** 的指数带量纲；未声明量纲也不是参数的叶子'
        '（语料里只给 L 声明了量纲，含 X 的候选在打分前就被硬拒，所以图上看不见这堵墙）',
        '先验   bms_wiki_nv13_np13（[BMS2020] wiki 语料校准，一阶 18 项 + 二阶 1 项），scale = 1',
        f'采样   20 温度并行回火，Ts = 1.04^k (k=0..19)，停滞退火关闭，burnin 4000 /'
        f' thin 10 / 保留 {samples} 样本 / {nchains} 条链 / max_size 50',
        f'剔除   {drop_txt}',
        f'来源   temp/md_pt20/multidataset_pt20/trace_{tag}_*.dat',
    ]


VARS = {'main': 'L, g', 'ctrlA_Lonly': 'L', 'ctrlB_Lg': 'L, g', 'ctrlC_constG': 'L, g'}
VARS_LIST = {k: [v.strip() for v in val.split(',')] for k, val in VARS.items()}
LABELS = {'main': '主实验', 'ctrlA_Lonly': '对照A 只给L', 'ctrlB_Lg': '对照B 仅地球',
          'ctrlC_constG': '对照C g恒定'}
# 参考拟合值取自 known_g/pendulum_runs/_logs/run_pt20_noanneal_2026-10-18.log
TARGET = {
    'main': '目标   T = a·√(L/g)；5 星球参考拟合 a = 6.2839（2π = 6.2832，差 0.011%）',
    'ctrlA_Lonly': '目标   只给 L 时真公式族退化为 T = a·√L；参考 a = 2.0023'
                   '（真值 2π/√9.81 = 2.0061）',
    'ctrlB_Lg': '目标   T = a·√(L/g)；仅地球参考拟合 a = 6.2715',
    'ctrlC_constG': '目标   T = a·√(L/g)，但 g 一律给 9.81；参考 a = 6.2803',
}
REF_FORMULA = {'main': '(_a_ * sqrt((L / g)))', 'ctrlA_Lonly': '(_a_ * sqrt(L))',
               'ctrlB_Lg': '(_a_ * sqrt((L / g)))', 'ctrlC_constG': '(_a_ * sqrt((L / g)))'}
TARGET_SHORT = {'main': 'T = a·√(L/g)', 'ctrlA_Lonly': 'T = a·√L（g 不给）',
                'ctrlB_Lg': 'T = a·√(L/g)', 'ctrlC_constG': 'T = a·√(L/g)'}


def build(which):
    jobs = []
    if which in ('pt20', 'singleT'):
        run = ('bms_wiki_nv13_np13_pt20' if which == 'pt20' else 'bms_wiki_nv13_np13')
        temper = ('20 温度并行回火，Ts = 1.04^k (k=0..19)，停滞退火关闭'
                  if which == 'pt20' else '单温度（无温度阶梯）')
        head = '20 温度回火' if which == 'pt20' else '单温度'
        tags = ['main', 'ctrlA_Lonly', 'ctrlB_Lg'] + (['ctrlC_constG'] if which == 'pt20' else [])
        for tag in tags:
            chains = load_known_g(RUNS / run, tag)
            agg = aggregate(chains)
            order = pick_nodes(agg)
            jobs.append(dict(
                name=f'knownG_{which}_{tag}', agg=agg, order=order,
                min_w=edge_threshold(agg, order),
                family=family_members(order, 'sqrt', g_constant=(tag != 'main')),
                ref_can=canonical(REF_FORMULA[tag]),
                title=f'规范形转移网络｜多星球单摆 · {head} · {LABELS[tag]}',
                subtitle=f'{len(chains)} 条链 × {len(chains[0])} 个保留样本，'
                         f'输入变量 {VARS[tag]}，目标 {TARGET_SHORT[tag]}',
                cond_lines=known_g_lines(run, tag, len(chains), len(chains[0]), temper)))
    elif which == 'md':
        for tag in ['md_Lonly', 'md_candG', 'md_junkX']:
            chains, dropped = load_md(tag)
            x_used = sum(1 for ch in chains for c, _ in ch
                         if re.search(r'(?<![A-Za-z_])X(?![A-Za-z_])', c))
            agg = aggregate(chains)
            order = pick_nodes(agg)
            jobs.append(dict(
                name=f'multiDataset_{tag}', agg=agg, order=order,
                min_w=edge_threshold(agg, order),
                family=family_members(order, 'multi', xvals=MD_G),
                ref_can=canonical('(_a_ * sqrt(L))'),
                title=f'规范形转移网络｜多数据集单摆 · {MD_LABELS[tag]}',
                subtitle=f'{len(chains)} 条链 × {len(chains[0])} 个保留样本，'
                         f'一条公式共享 5 份数据，目标每份内 T ∝ √L',
                cond_lines=md_lines(tag, len(chains), len(chains[0]), dropped,
                                    x_used, sum(len(ch) for ch in chains))))
    else:
        for cfg, label, temper in [
                ('pt_nv1np2_normal', '20 温度回火', 'Ts = 1.04^k (k=0..19)，停滞退火关闭'),
                ('singleT_nv1np2_normal', '单温度', '无温度阶梯')]:
            chains = load_lT(cfg)
            agg = aggregate(chains)
            order = pick_nodes(agg)
            jobs.append(dict(
                name=f'lTpowerlaw_{cfg}', agg=agg, order=order,
                min_w=edge_threshold(agg, order),
                family=family_members(order, 'power'), ref_can=canonical('(_a_ * (L ** _b_))'),
                title=f'规范形转移网络｜只给 L 的幂律还原 · {label}',
                subtitle=f'{len(chains)} 条链 × {len(chains[0])} 个保留样本，'
                         f'输入变量 L，目标 T = a·L^b（b≈0.5）',
                cond_lines=LT_LINES_COMMON + [f'回火   {temper}',
                                        f'来源   upstream_runs/trace_{cfg}_*.jsonl']))
    return jobs


def edge_threshold(agg, order, keep=70):
    nodes = set(order)
    inw = sorted((w for (a, b), w in agg['edges'].items()
                  if a in nodes and b in nodes), reverse=True)
    return max(2, inw[min(keep, len(inw)) - 1] if len(inw) >= 2 else 2)


def main(which):
    sets = ['pt20', 'singleT', 'md', 'lT'] if which == 'all' else [which]
    blocks = []
    for s in sets:
        for job in build(s):
            stats = make_figure(**job)
            print(f"{job['name']:32s} {stats}")
            blocks.append('\n'.join([
                f"## fig_tracegraph_{job['name']}",
                '',
                f"**{job['title']}**  ",
                f"{job['subtitle']}。{stats}。",
                '',
                '试验条件：', ''] + [f'- {l.strip()}' for l in job['cond_lines']] + [
                '',
                '读法：圆面积 ∝ 该规范形被访问的次数（跨链合计），圈内数字对应图上的节点表；'
                '红圈 = 真公式族成员（表内记作 族，★ = 参考式的字面规范形）；填色 = 节点平均能量 E（每张图独立色标）；'
                '箭头粗细与深浅 ∝ 转移次数。',
                '',
                '限制：两个保留样本之间隔着 thin 步（回火还叠加每步一次的副本交换），故为粗粒度过渡'
                '而非单次提议；被拒绝的提议与被宪法硬拒的候选不出现在图上；规范形按 sympy 等价合并。',
                '']))
    if which == 'all':
        (OUT / 'captions.md').write_text(
            '# 规范形转移网络图：图注与试验条件\n\n'
            '本文件由 `experiments/pendulum/trace_graph_figs.py --all` 生成，'
            '每张图的图注与它自己的试验条件。\n\n' + '\n'.join(blocks))
        print('图注 ->', OUT / 'captions.md')
    print('->', OUT)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'pt20')
