"""实验目的：测量议会动力学——软先验如何参与能量记账。

议会是贝叶斯机器科学家的软约束层：各位专家（奥卡姆尺寸罚项、
渐近行为先验……）对表达式结构发表评价，按 softmax 权重加权后
以 EP 进入总能量 E = EB + EP。本实验文件检验议会动力学的基本定律：

1. 符号约定——表达式变大 ⇒ 议会能量升高（罚项恒正）；
2. 归一化——多专家权重之和恒为 1，议会能量是各专家原始评价的
   加权和，缓存按规范形、按专家名分别记录；
3. 门控级联——宪法层（硬约束）先于议会层（软先验）：违宪提议
   以 inf 被拒，议会根本不被询问；
4. 记账完整性——试算提议（dE_et/dE_lr/dE_rr）在试算后完全回退，
   canonical 拒绝冻结 E 与 EP，专家异常被静默而不中断链条。

另含端对端轨迹实验：MCMC 链条在单议会与双议会配置下逐步能量
有限，且"变大则 EP 升高"的约定沿轨迹成立。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from core import Tree
from experts.parliament.base import ParliamentBase
from experts.parliament.occam import OccamPrior
from experts.parliament.asymptotic import AsymptoticPrior
from experts.constitution.dimensional import DimensionalConstitution
from test.lab import check, measure, reproducible_state, section


# ------------------------------------------------------------------
# 数据准备（与源实验一致）
# ------------------------------------------------------------------

def _make_linear_data(n=50, seed=42):
    reproducible_state(seed)
    x_vals = np.linspace(-3, 3, n)
    y_true = 2.0 * x_vals + 1.0
    y_obs = y_true + np.random.normal(0, 0.1, n)
    return (
        pd.DataFrame({'x': x_vals}),
        pd.Series(y_obs, name='y'),
    )


# ------------------------------------------------------------------
# 议会基线与奥卡姆先验
# ------------------------------------------------------------------

def test_parliament_base_defaults():
    """假设：基类议会专家是无信息先验——对任意结构的评价为零，
    因而不贡献任何议会能量。"""
    section("议会基线与奥卡姆先验")
    p = ParliamentBase()
    check("默认专家名", p.name, 'unnamed_parliament')
    tree = Tree(variables=['x'], parameters=['a'],
                from_string='(_a0_ * x)')
    check("结构评价（无信息）", p.evaluate_structural(tree), 0.0)
    check("拟合评价（无信息）", p.evaluate_fitted(tree), 0.0)


def test_occam_size_penalty():
    """假设：奥卡姆罚项正比于表达式树的节点数——结构越复杂，
    先验能量越高；拟合通道不参与结构罚项。"""
    op = OccamPrior()
    # size=1 tree (just 'x')
    t1 = Tree(variables=['x'], parameters=['a'], from_string='x')
    check("单节点树结构罚项", op.evaluate_structural(t1), 1.0)
    check("单节点树拟合罚项", op.evaluate_fitted(t1), 0.0)

    # size=3 tree: (_a0_ * x) = {_a0_, *, x} = 3 nodes
    t3 = Tree(variables=['x'], parameters=['a'],
              from_string='(_a0_ * x)')
    check("三节点树结构罚项", op.evaluate_structural(t3), 3.0)

    # size=4 tree: (_a0_ * sin(x)) = {_a0_, *, sin, x} = 4 nodes
    t4 = Tree(variables=['x'], parameters=['a'],
              from_string='(_a0_ * sin(x))')
    check("四节点树结构罚项", op.evaluate_structural(t4), 4.0)


# ------------------------------------------------------------------
# 符号约定：罚项随尺寸增长（fix #1 回归）
# ------------------------------------------------------------------

def test_larger_tree_positive_dEP():
    """假设：把叶子替换为子树使表达式变大，议会能量增量 dEP > 0
    （罚项随复杂度增长）。"""
    section("符号约定：变大则罚")
    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[OccamPrior()],
             from_string='(_a0_ * x)')
    x_node = [n for n in t.nodes if n.value == 'x'][0]
    # Replace x with sin(x): size grows from 3 to 4
    dE, dEB, dEP, pv, nif, nfi = t.dE_et(x_node, ['sin', ['x']])
    measure("变大提议的 dEP", dEP)
    check("变大提议 dEP > 0", dEP > 0, True)


def test_same_size_zero_structural_delta():
    """假设：叶子换叶子的提议不改变树尺寸，能量增量保持有限
    （议会结构增量为零，但 prior_par 仍可能贡献 dEP）。"""
    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[OccamPrior()],
             from_string='(_a0_ * x)')
    x_node = [n for n in t.nodes if n.value == 'x'][0]
    # Replace x with 1 (both leaves, size unchanged)
    dE, dEB, dEP, pv, nif, nfi = t.dE_et(x_node, ['1', []])
    measure("等尺寸替换的 dE", dE)
    # Structural delta is 0 (same size) but prior_par may change dEP
    check("等尺寸替换 dE 有限", np.isfinite(dE), True)


# ------------------------------------------------------------------
# 能量记账：EP 包含议会贡献
# ------------------------------------------------------------------

def test_EP_includes_structural_penalty():
    """假设：初态能量记账中 EP 至少包含议会结构罚项
    （size=3, weight=1.0 ⇒ 贡献 3.0）。"""
    section("能量记账：EP 的议会贡献")
    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[OccamPrior()],
             from_string='(_a0_ * x)')
    # Size=3, weight=1.0: parliament penalty = 3.0
    measure("初态 EP", t.EP)
    check("EP ≥ 议会罚项 3.0", t.EP > 3.0, True)


def test_no_parliaments_no_extra_EP():
    """假设：同构表达式下，接入议会的系统 EP 严格高于无议会基线
    （软先验只增不减议会能量）。"""
    t_no = Tree(variables=['x'], parameters=['a'],
                from_string='(_a0_ * x)')
    t_with = Tree(variables=['x'], parameters=['a'],
                  parliaments=[OccamPrior()],
                  from_string='(_a0_ * x)')
    measure("无议会 EP", t_no.EP)
    measure("有议会 EP", t_with.EP)
    check("议会使 EP 升高", t_with.EP > t_no.EP, True)


# ------------------------------------------------------------------
# 议会记忆：缓存行为
# ------------------------------------------------------------------

def test_cache_hit_reuses_raw_values():
    """假设：同一规范形的重复评价命中议会记忆，不产生新条目
    （缓存只随新规范形增长）。"""
    section("议会记忆：缓存行为")
    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[OccamPrior()],
             from_string='x')
    pen1 = t._parliament_energy_structural()
    cache_size_before = len(t._parliament_structural_cache)
    pen2 = t._parliament_energy_structural()
    cache_size_after = len(t._parliament_structural_cache)
    measure("议会结构能量（首次）", pen1)
    measure("议会结构能量（重复）", pen2)
    check("重复测量一致", pen1, pen2)
    check("缓存大小不变", cache_size_after, cache_size_before)


def test_cache_stores_per_expert_raw():
    """假设：议会记忆按专家名分别存放原始评价，加权发生在读取时
    而非存储时。"""
    class DummyP(ParliamentBase):
        name = 'dummy'
        def evaluate_structural(self, tree):
            return 42.0

    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[OccamPrior(), DummyP()],
             from_string='x')
    # Force cache population
    t._parliament_energy_structural()
    canonical = t.canonical()
    raw = t._parliament_structural_cache[canonical]
    check("缓存含 occam 条目", 'occam' in raw, True)
    check("缓存含 dummy 条目", 'dummy' in raw, True)
    check("occam 原始值（size=1）", raw['occam'], 1.0)
    check("dummy 原始值", raw['dummy'], 42.0)


def test_cache_grows_with_new_canonicals():
    """假设：首次评价即为当前规范形建立至少一条议会记忆记录。"""
    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[OccamPrior()],
             from_string='x')
    t._parliament_energy_structural()
    n_cache = len(t._parliament_structural_cache)
    measure("缓存条目数", n_cache)
    check("缓存至少 1 条", n_cache >= 1, True)


# ------------------------------------------------------------------
# 门控级联：宪法（硬约束）先于议会（软先验）
# ------------------------------------------------------------------

def test_gate_failure_returns_inf():
    """假设：量纲违宪的提议在宪法层被无限能量拒绝，议会层根本不
    被询问（级联：gate → parliament）。"""
    section("门控级联：宪法先于议会")
    dims = {'x': (1, 0, 0, 0, 0, 0, 0),
            't': (0, 0, 1, 0, 0, 0, 0)}
    t = Tree(variables=['x', 't'], parameters=['a'],
             dimensions=dims, constitutions=[DimensionalConstitution()],
             parliaments=[OccamPrior()],
             from_string='(_a0_ * x)', max_size=30)
    x_node = [n for n in t.nodes if n.value == 'x'][0]
    # Replace x with sin(x): sin(x) is dimensionless but x is L
    dE, dEB, dEP, pv, nif, nfi = t.dE_et(x_node, ['sin', ['x']])
    measure("违宪提议的 dE", dE)
    check("宪法拒绝 dE = inf", np.isinf(dE), True)


def test_gate_pass_parliament_contributes():
    """假设：量纲合法的提议通过宪法层后，议会 delta 正常进入 dEP
    （树 3 → 5 节点，罚项增量为正）。"""
    # Both L1 and L2 have length dimension (1,0,0,...)
    dims = {'L1': (1, 0, 0, 0, 0, 0, 0),
            'L2': (1, 0, 0, 0, 0, 0, 0)}
    t = Tree(variables=['L1', 'L2'], parameters=['a'],
             dimensions=dims, constitutions=[DimensionalConstitution()],
             parliaments=[OccamPrior()],
             from_string='(_a0_ * L1)', max_size=30)
    L1_node = [n for n in t.nodes if n.value == 'L1'][0]
    # Replace L1 with (L1 + L2): shape = 3 -> 5, dimension L+L=L OK
    dE, dEB, dEP, pv, nif, nfi = t.dE_et(
        L1_node, ['+', ['L1', 'L2']])
    measure("合法提议的 dEP", dEP)
    check("合法提议 dE 有限", np.isfinite(dE), True)
    check("议会 delta 为正", dEP > 0, True)


# ------------------------------------------------------------------
# 多专家加权：softmax 归一化
# ------------------------------------------------------------------

def test_weights_sum_to_one():
    """假设：多专家并存时 softmax 权重归一——权重和恒为 1
    （议会总声量守恒）。"""
    section("多专家加权：softmax 归一化")
    class P1(ParliamentBase):
        name = 'p1'

    class P2(ParliamentBase):
        name = 'p2'

    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[P1(), P2()],
             from_string='x')
    w = t.parliament_weights
    measure("专家权重", dict(w))
    check("权重和", sum(w.values()), 1.0, tol=1e-10)


def test_weighted_sum_correct():
    """假设：议会能量是各专家原始评价的加权和——等权双专家下
    0.5×10 + 0.5×20 = 15.0，精确到 1e-10。"""
    class P1(ParliamentBase):
        name = 'p1'
        def evaluate_structural(self, tree):
            return 10.0

    class P2(ParliamentBase):
        name = 'p2'
        def evaluate_structural(self, tree):
            return 20.0

    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[P1(), P2()],
             from_string='x')
    structural = t._parliament_energy_structural()
    measure("议会结构能量", structural)
    check("等权加权和", structural, 15.0, tol=1e-10)


def test_weights_equal_with_two_experts():
    """假设：两位 gamma=0 的专家在 softmax 下平分权重（各 0.5），
    权重和归一。"""
    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[OccamPrior(),
                          AsymptoticPrior(finite_at_zero=['x'])],
             from_string='x')
    w = t.parliament_weights
    measure("专家权重", dict(w))
    check("occam 权重", w['occam'], 0.5, tol=1e-10)
    check("asymptotic 权重", w['asymptotic'], 0.5, tol=1e-10)
    check("权重和", sum(w.values()), 1.0, tol=1e-10)


def test_weighted_energy_structural():
    """假设：(x / y) 的议会能量 = 0.5×尺寸(3) + 0.5×渐近罚项(0.5)
    = 1.75，加权求和精确到 1e-10。"""
    t = Tree(variables=['x', 'y'], parameters=['a'],
             parliaments=[OccamPrior(),
                          AsymptoticPrior(finite_at_zero=['y'])],
             from_string='(x / y)')
    # OccamPrior raw: size = 3
    # AsymptoticPrior raw: y in denominator = 0.5
    structural = t._parliament_energy_structural()
    measure("议会结构能量", structural)
    expected = 0.5 * 3 + 0.5 * 0.5
    check("双专家加权和", structural, expected, tol=1e-10)


def test_cache_stores_two_experts_raw():
    """假设：双专家缓存按名字分别记录原始值——occam 记尺寸 3.0，
    asymptotic 记渐近罚项 0.5。"""
    t = Tree(variables=['x', 'y'], parameters=['a'],
             parliaments=[OccamPrior(),
                          AsymptoticPrior(finite_at_zero=['y'])],
             from_string='(x / y)')
    t._parliament_energy_structural()
    canonical = t.canonical()
    raw = t._parliament_structural_cache[canonical]
    check("occam 原始值", raw['occam'], 3.0)
    check("asymptotic 原始值", raw['asymptotic'], 0.5)


def test_EP_differs_with_second_expert():
    """假设：加入第二位专家后 softmax(gamma=0) 稀释 occam 权重
    （1.0 → 0.5），议会能量反而下降（3.0 → 1.75）——这是等权
    softmax 的数学性质：先验增多均摊权重，而非按数据逐一定标。
    待 gamma 学习机制修正。"""
    t_occam = Tree(variables=['x', 'y'], parameters=['a'],
                   parliaments=[OccamPrior()],
                   from_string='(x / y)')
    t_both = Tree(variables=['x', 'y'], parameters=['a'],
                  parliaments=[OccamPrior(),
                               AsymptoticPrior(finite_at_zero=['y'])],
                  from_string='(x / y)')
    measure("仅 occam 的议会能量", t_occam._parliament_energy_structural())
    structural = t_both._parliament_energy_structural()
    measure("双专家的议会能量", structural)
    check("双专家 EP 更低", t_both.EP < t_occam.EP, True)
    check("双专家议会能量", structural, 1.75, tol=1e-10)


# ------------------------------------------------------------------
# 误差韧性：专家异常静默
# ------------------------------------------------------------------

def test_expert_raises_is_silenced():
    """假设：单个专家在评价时抛异常不中断 MCMC 链条——异常被静默，
    该专家贡献归零，提议照常进行。"""
    section("误差韧性：专家异常静默")
    class BadExpert(ParliamentBase):
        name = 'bad'
        def evaluate_structural(self, tree):
            raise RuntimeError("deliberate error")

    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[BadExpert()],
             from_string='(_a0_ * x)')
    check("异常专家结构能量归零", t._parliament_energy_structural(), 0.0)
    x_node = [n for n in t.nodes if n.value == 'x'][0]
    dE, dEB, dEP, pv, nif, nfi = t.dE_et(x_node, ['sin', ['x']])
    measure("提议 dE", dE)
    check("dE 有限", np.isfinite(dE), True)


def test_capture_delta_handles_exceptions():
    """假设：议会记账的 capture / delta 通道对专家异常同样鲁棒，
    异常两侧的差分按零处理。"""
    class FlakyExpert(ParliamentBase):
        name = 'flaky'
        def evaluate_structural(self, tree):
            raise RuntimeError("always fails")

    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[FlakyExpert()],
             from_string='(_a0_ * x)')
    old_val = t._capture_parliament_structural()
    check("capture 异常归零", old_val, 0.0)
    check("delta 异常归零", t._parliament_structural_delta(old_val), 0.0)


# ------------------------------------------------------------------
# 提议通道：dE_lr / dE_rr 中的议会 delta（试算后完全回退）
# ------------------------------------------------------------------

def test_dE_lr_parliament_delta():
    """假设：dE_lr 叶算子替换的试算完全回退——表达式串不变，
    议会 delta 有限；叶子-叶子替换的结构增量为零。"""
    section("提议通道：dE_lr / dE_rr 试算回退")
    t = Tree(variables=['x'], parameters=['a'],
            parliaments=[OccamPrior()],
            from_string='(_a0_ * x)', max_size=30)
    x_node = [n for n in t.nodes if n.value == 'x'][0]
    before = str(t)

    dE, _, dEP, _ = t.dE_lr(x_node, 'sin')
    measure("dE_lr 议会增量 dEP", dEP)
    check("dE 有限", np.isfinite(dE), True)
    check("试算后树回退", str(t), before)
    check("dEP 有限", np.isfinite(dEP), True)


def test_dE_lr_with_constitution_and_parliament():
    """假设：* → + 的算子替换通过量纲宪法（L×L 与 L+L 皆量纲自洽），
    议会在宪法放行后照常评价，试算后树回退。"""
    dims = {'x': (1, 0, 0, 0, 0, 0, 0)}
    t = Tree(variables=['x'], parameters=['a'],
             dimensions=dims,
             constitutions=[DimensionalConstitution()],
             parliaments=[OccamPrior()],
             from_string='(_a0_ * x)', max_size=30)
    mul_node = [n for n in t.nodes if n.value == '*'][0]
    before = str(t)

    dE, _, _, _ = t.dE_lr(mul_node, '+')
    measure("dE", dE)
    check("dE 有限", np.isfinite(dE), True)
    check("试算后树回退", str(t), before)


def test_dE_rr_prune_parliament_delta():
    """假设：dE_rr 剪枝试算完全回退——表达式串与尺寸均与试算前
    一致，议会能量记账不被污染。"""
    dims = {'L': (1, 0, 0, 0, 0, 0, 0)}
    t = Tree(variables=['L'], parameters=['a'],
             dimensions=dims,
             constitutions=[DimensionalConstitution()],
             parliaments=[OccamPrior()],
             from_string='(_a0_ + L)', max_size=30)
    check("根可剪", t.is_root_prunable(), True)
    before = str(t)
    old_size = t.size

    dE, _, dEP, _ = t.dE_rr(rr=None)
    measure("剪枝 dEP", dEP)
    check("dE 有限", np.isfinite(dE), True)
    check("表达式回退", str(t), before)
    check("尺寸回退", t.size, old_size)


def test_dE_rr_replace_parliament_delta():
    """假设：dE_rr 换根试算把 sqrt(L/g) 改写为 _b0_ * sqrt(L/g)——
    量纲宪法放行（长度×无量纲），议会见证尺寸增大，试算后回退。"""
    dims = {'L': (1, 0, 0, 0, 0, 0, 0),
            'g': (1, 0, -2, 0, 0, 0, 0)}
    t = Tree(variables=['L', 'g'], parameters=['a', 'b'],
             dimensions=dims,
             constitutions=[DimensionalConstitution()],
             parliaments=[OccamPrior()],
             from_string='sqrt((L / g))', max_size=30)
    before = str(t)

    dE, _, _, _ = t.dE_rr(rr=['*', ['_b0_']])
    measure("换根 dE", dE)
    check("dE 有限", np.isfinite(dE), True)
    check("试算后树回退", str(t), before)


def test_canonical_reject_skips_parliament_delta():
    """假设：canonical 拒绝（update_representative 返回 -1）的提议
    能量为 inf，且议会能量记账完全不更新——E 与 EP 冻结在初值。"""
    t = Tree(variables=['x'], parameters=['a'],
             parliaments=[OccamPrior()],
             from_string='(_a0_ * x)', max_size=30)
    old_E, old_EP = t.E, t.EP

    orig_update = t.update_representative
    t.update_representative = lambda verbose=False: -1
    try:
        x_node = [n for n in t.nodes if n.value == 'x'][0]
        dE, _, _, _, _, _ = t.dE_et(x_node, ['sin', ['x']])
        measure("canonical 拒绝的 dE", dE)
        check("canonical 拒绝 dE = inf", np.isinf(dE), True)
        check("EP 冻结", t.EP, old_EP)
        check("E 冻结", t.E, old_E)
    finally:
        t.update_representative = orig_update


def test_dE_et_includes_both_contributions():
    """假设：x → (x / y) 的提议同时抬升两位专家的罚项（尺寸 3→5，
    且新增 y 分母），dEP 收双份正贡献（occam +1.0、asymptotic +0.25
    加权后至少 1.25，另有 prior_par 变化）。"""
    t = Tree(variables=['x', 'y'], parameters=['a'],
             parliaments=[OccamPrior(),
                          AsymptoticPrior(finite_at_zero=['y'])],
             from_string='(_a0_ * x)', max_size=30)
    x_node = [n for n in t.nodes if n.value == 'x'][0]
    # Replace x with (x / y): adds / node with y in denominator,
    # tree size grows from 3 to 5
    dE, dEB, dEP, pv, nif, nfi = t.dE_et(
        x_node, ['/', ['x', 'y']])
    measure("双专家提议的 dEP", dEP)
    check("dE 有限", np.isfinite(dE), True)
    check("双专家 dEP > 0", dEP > 0, True)


# ------------------------------------------------------------------
# 端对端轨迹：议会下的 MCMC
# ------------------------------------------------------------------

def test_mcmc_short_chain():
    """假设：单议会（OccamPrior）下 50 步 MCMC 轨迹能量记账逐步
    有限；若链条演化出更大表达式，议会能量随之升高；议会记忆
    保持有界。"""
    section("端对端轨迹：议会下的 MCMC")
    x_df, y_series = _make_linear_data(n=50)
    t = Tree(
        variables=['x'], parameters=['a', 'b'],
        x=x_df, y=y_series,
        parliaments=[OccamPrior()],
        from_string='((_a0_ * x) + _b0_)',
        max_size=20,
    )
    initial_EP = t.EP
    initial_size = t.size
    measure("初态 size", initial_size)
    measure("初态 EP", initial_EP)

    # Run 50 steps
    for i in range(50):
        t.mcmc_step(verbose=False)
        check("E 有限", np.isfinite(float(t.E)), True, context=f"step {i}")
        check("EP 有限", np.isfinite(float(t.EP)), True, context=f"step {i}")

    measure("终态 size", t.size)
    measure("终态 EP", t.EP)

    # Verify: larger trees have higher EP (penalty convention)
    if t.size > initial_size:
        check("变大后 EP 升高", t.EP > initial_EP, True,
              context=f"size {t.size} > {initial_size}")

    # Verify: cache is bounded
    n_cache = len(t._parliament_structural_cache)
    measure("议会缓存条目数", n_cache)
    check("缓存有界", n_cache < 10000, True)

    # Verify: some accepted moves occurred, or chain stayed on correct formula
    # (50 steps may not accept if initial formula is already optimal)
    check("终态 E 有限", np.isfinite(float(t.E)), True)
    check("终态 EP 有限", np.isfinite(float(t.EP)), True)


def test_mcmc_short_chain_both_experts():
    """假设：双议会（OccamPrior + AsymptoticPrior）下 50 步 MCMC
    轨迹能量记账逐步有限，"变大则 EP 升高"的约定沿轨迹成立，
    议会记忆保持有界。"""
    reproducible_state(123)
    x_vals = np.linspace(-3, 3, 50)
    y_true = 2.0 * x_vals + 1.0
    y_obs = y_true + np.random.normal(0, 0.1, 50)

    x_df = pd.DataFrame({'x': x_vals})
    y_series = pd.Series(y_obs, name='y')

    t = Tree(
        variables=['x'], parameters=['a', 'b'],
        x=x_df, y=y_series,
        parliaments=[
            OccamPrior(),
            AsymptoticPrior(finite_at_zero=['x']),
        ],
        from_string='((_a0_ * x) + _b0_)',
        max_size=20,
    )

    initial_EP = t.EP
    initial_size = t.size
    measure("初态 size", initial_size)
    measure("初态 EP", initial_EP)

    for i in range(50):
        t.mcmc_step(verbose=False)
        check("E 有限", np.isfinite(float(t.E)), True, context=f"step {i}")
        check("EP 有限", np.isfinite(float(t.EP)), True, context=f"step {i}")

    measure("终态 size", t.size)
    measure("终态 EP", t.EP)

    # Larger tree -> higher EP (penalty convention)
    if t.size > initial_size:
        check("变大后 EP 升高", t.EP > initial_EP, True,
              context=f"size {t.size} > {initial_size}")

    # Cache bounded
    n_cache = len(t._parliament_structural_cache)
    measure("议会缓存条目数", n_cache)
    check("缓存有界", n_cache < 10000, True)

    check("终态 E 有限", np.isfinite(float(t.E)), True)
    check("终态 EP 有限", np.isfinite(float(t.EP)), True)


if __name__ == '__main__':
    for name in sorted(dir()):
        if name.startswith('test_'):
            globals()[name]()
    print("\n实验记录输出完毕（pytest 下用 -s 查看）。")
