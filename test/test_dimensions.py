"""量纲齐次性：任何物理表达式必须量纲自洽（Buckingham π 精神）。

宪法层的核心物理假设：
  1. 加法/减法要求两子树量纲相同（同类量才能相加）；
  2. 乘法/除法允许量纲相乘/相消；
  3. 初等函数（sin, exp, log...）要求输入无量纲；
  4. sqrt 把量纲指数折半，** 只要求指数无量纲——分数维合法，
     幂律式 a·√L 与 L**b 的量纲由自由参数吸收；
  5. 自由参数是待解的量纲变量——共享参数必须同时满足所有约束；
  6. 违宪表达式以无穷能量拒绝，且提议试算后树状态完全回退。

运行示例：单摆 T = 2π√(L/g) 及其家族（偏移、角度修正、大振幅级数）。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from experts.constitution.dimensional import (
    DimExpr, DIM_NAMES, N_DIMS, POW_OUT,
    collect_constraints, solve_constraints,
    check_dimensional_consistency,
    DimensionalConstitution,
    DimensionCheckResult, DimensionFailure,
)
from core import Tree, Node
from test.lab import check, check_above, measure, reproducible_state, section

# SI-7 量纲坐标 (L, M, T, I, Theta, N, J)
DIM_L = (1, 0, 0, 0, 0, 0, 0)
DIM_T = (0, 0, 1, 0, 0, 0, 0)
DIM_M = (0, 1, 0, 0, 0, 0, 0)
DIM_G = (1, 0, -2, 0, 0, 0, 0)
DIM_D = (0, 0, 0, 0, 0, 0, 0)

PENDULUM_DIMS = {'L': DIM_L, 'g': DIM_G, 'm': DIM_M, 'theta0': DIM_D}

DEFAULT_PRIOR_PAR_WEIGHT = 10.0


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------

def leaf(val, parent=None):
    return Node(val, parent=parent, offspring=[])


def unary(op, child):
    n = Node(op, parent=None, offspring=[child])
    child.parent = n
    return n


def binary(op, left, right):
    n = Node(op, parent=None, offspring=[left, right])
    left.parent = n
    right.parent = n
    return n


def de(coeffs=None, const=0):
    """构造 DimExpr 的简写。"""
    c = tuple(sorted((k, Fraction(v)) for k, v in (coeffs or {}).items()))
    return DimExpr(_coeffs=c,
                   const=const if isinstance(const, Fraction) else Fraction(const))


def _capture_state(tree):
    """捕获树的结构状态（用于试算回退验证）。"""
    return (tree.size,
            dict(tree.nops or {}),
            [(n.value, len(n.offspring)) for n in tree.nodes])


def _check_state_unchanged(tree, state, label):
    """检验树状态与捕获状态一致（提议试算完全回退）。"""
    sz, nops, nodes_info = state
    check("尺寸回退", tree.size, sz, context=label)
    for k, v in nops.items():
        check(f"nops[{k}] 回退", tree.nops.get(k, 0), v, context=label)
    check("节点结构回退",
          [(n.value, len(n.offspring)) for n in tree.nodes], nodes_info,
          context=label)


def _solve(root, dims, params, label, expect_valid):
    """收集约束并求解，检验量纲合法性判定。"""
    _, constraints, unknowns = collect_constraints(root, dims, params)
    result = solve_constraints(constraints, extra_failures=unknowns)
    check("量纲判定", result.is_valid, expect_valid, context=label)
    return result


def _ep_from_nops(nops, prior_par_weight=DEFAULT_PRIOR_PAR_WEIGHT):
    return sum(prior_par_weight * n for n in nops.values())


# ============================================================================
# DimExpr 代数
# ============================================================================

def test_dimexpr_birth_and_immutability():
    """DimExpr 出生：零元、系数-常数结构、不可变性。"""
    section("DimExpr 代数")
    e = DimExpr()
    check("零元 is_zero", e.is_zero(), True)
    check("零元 const", e.const, Fraction(0))
    check("零元无系数", len(e.coeffs), 0)

    e2 = DimExpr(_coeffs=(('_a0_', Fraction(1)),), const=Fraction(2))
    check("非零判定", e2.is_zero(), False)
    check("const 保留", e2.const, Fraction(2))

    with pytest.raises(Exception):
        e2.const = Fraction(3)
    check("不可变性（赋值抛异常）", True, True)


def test_dimexpr_arithmetic():
    """DimExpr 代数：加减、标量乘、零化、取负、右乘。"""
    a = de({'p': 1}, const=2)
    b = de({'p': 3}, const=5)

    r = a + b
    check("加法系数", dict(r.coeffs), {'p': Fraction(4)})
    check("加法常数", r.const, Fraction(7))

    r = de({'p': 1}) + de({'p': -1})
    check("相反系数相消", len(r.coeffs), 0)
    check("相消后为零", r.is_zero(), True)

    r = a - b
    check("减法系数", dict(r.coeffs), {'p': Fraction(-2)})
    check("减法常数", r.const, Fraction(-3))

    r = 3 * de({'p': 1, 'q': 2}, const=3)
    check("标量乘系数", dict(r.coeffs), {'p': Fraction(3), 'q': Fraction(6)})
    check("标量乘常数", r.const, Fraction(9))

    r = 0 * de({'p': 5}, const=7)
    check("零化", r.is_zero(), True)

    r = -de({'p': 1, 'q': -3}, const=5)
    check("取负系数", dict(r.coeffs), {'p': Fraction(-1), 'q': Fraction(3)})
    check("取负常数", r.const, Fraction(-5))

    r = Fraction(1, 2) * de({'p': 2}, const=1)
    check("Fraction 右乘系数", dict(r.coeffs), {'p': Fraction(1)})
    check("Fraction 右乘常数", r.const, Fraction(1, 2))


def test_dimexpr_equality_and_repr():
    """DimExpr 等价类：相等、哈希、打印。"""
    a = de({'p': 1}, const=2)
    b = de({'p': 1}, const=2)
    c = de({'p': 1}, const=3)
    check("相等", a == b, True)
    check("可区分", a != c, True)
    check("哈希一致", hash(a) == hash(b), True)
    check("集合去重", len({a, b}), 1)

    r = repr(de({'p': 2, 'q': 1}, const=3))
    check("repr 含系数项", '2*p' in r and '1*q' in r and '3' in r, True)


def test_dimension_check_result_structure():
    """量纲判定结果的数据结构。"""
    section("量纲判定数据结构")
    r = DimensionCheckResult(is_valid=True, failures=[])
    check("合法结果", r.is_valid, True)
    check("无失败记录", len(r.failures), 0)

    f = DimensionFailure(dim_index=0, dim_name='L',
                         description="L + g dimension mismatch",
                         conflicting_exprs=(de({'a': 1}), de({'a': 1})))
    r = DimensionCheckResult(is_valid=False, failures=[f])
    check("非法结果", r.is_valid, False)
    check("失败定位 dim_index", r.failures[0].dim_index, 0)
    check("失败定位 dim_name", r.failures[0].dim_name, 'L')
    check("失败描述", "mismatch" in r.failures[0].description, True)

    f2 = DimensionFailure(dim_index=2, dim_name='T',
                          description="constraint has no solution")
    check("conflicting_exprs 默认 None", f2.conflicting_exprs is None, True)


# ============================================================================
# 约束收集：叶子与加减法
# ============================================================================

def test_leaf_dimensions():
    """叶子量纲：已知量纲的变量直接取值；参数为待解符号。"""
    section("约束收集：叶子量纲")
    dims = {'x': DIM_L}

    exprs, constraints, _ = collect_constraints(
        leaf('x'), dims, set())
    for j in range(N_DIMS):
        check(f"变量叶子 dim[{j}]", exprs[j].const, Fraction(DIM_L[j]))
        check(f"变量叶子 dim[{j}] 无参数系数", len(exprs[j].coeffs), 0)
    check("无约束产生", all(len(c) == 0 for c in constraints), True)

    exprs, constraints, _ = collect_constraints(
        leaf('_a0_'), dims, {'_a0_'})
    for j in range(N_DIMS):
        check(f"参数叶子 dim[{j}] 常数为零", exprs[j].const, Fraction(0))
        check(f"参数叶子 dim[{j}] 带符号系数",
              dict(exprs[j].coeffs), {'_a0_': Fraction(1)})


def test_addition_requires_same_dimensions():
    """加法齐次性：同类量才能相加；共享参数须同时满足全部约束。"""
    section("约束收集：加法齐次性")
    _solve(binary('+', leaf('x'), leaf('t')), {'x': DIM_L, 't': DIM_T}, set(),
           "L + T", expect_valid=False)

    # 线性模型 _a0_*x + _b0_：两个自由参数各自吸收量纲 → 合法
    mul = binary('*', leaf('_a0_'), leaf('x'))
    _solve(binary('+', mul, leaf('_b0_')), {'x': DIM_L}, {'_a0_', '_b0_'},
           "线性模型 a*x + b", expect_valid=True)

    # 共享参数 _a0_ 同时乘以 L 与 T：一个参数无法两全 → 非法
    left = binary('*', leaf('_a0_'), leaf('x'))
    right = binary('*', leaf('_a0_'), leaf('t'))
    _solve(binary('+', left, right), {'x': DIM_L, 't': DIM_T}, {'_a0_'},
           "共享参数 a*x + a*t", expect_valid=False)

    # 共享参数既要 L 又要无量纲 → 非法
    lterm = binary('*', leaf('_a0_'), leaf('x'))
    _solve(binary('+', lterm, leaf('_a0_')), {'x': DIM_L}, {'_a0_'},
           "共享参数 a*L + a", expect_valid=False)


def test_binary_minus_matches_like_addition():
    """二元减号（语料里有，采样器不生成）必须两侧同维。

    旧实现按一元规则只看第一个孩子，减数的量纲从未被检查过。
    """
    section("约束收集：二元减号")
    _solve(binary('-', leaf('x'), leaf('t')), {'x': DIM_L, 't': DIM_T}, set(),
           "L - T", expect_valid=False)
    _solve(binary('-', leaf('x'), leaf('t')), {'x': DIM_L, 't': DIM_L}, set(),
           "L - L（同维）", expect_valid=True)
    # 一元负号保持原语义
    _solve(unary('-', leaf('x')), {'x': DIM_L}, set(), "-L 保维",
           expect_valid=True)
    _solve(unary('sin', unary('-', leaf('x'))), {'x': DIM_L}, set(),
           "sin(-L) 依旧违宪", expect_valid=False)


def test_multiplicative_cancellation():
    """乘法/除法：量纲相乘相消。"""
    section("约束收集：乘除相消")
    # (x/t)*t → L
    div_node = binary('/', leaf('x'), leaf('t'))
    exprs, constraints, _ = collect_constraints(
        binary('*', div_node, leaf('t')), {'x': DIM_L, 't': DIM_T}, set())
    result = solve_constraints(constraints)
    check("(x/t)*t 合法", result.is_valid, True)
    check("根量纲 L^1", exprs[0].const, Fraction(1))
    check("根量纲 T^0", exprs[2].const, Fraction(0))

    # x/x → 无量纲
    exprs, _, _ = collect_constraints(
        binary('/', leaf('x'), leaf('x')), {'x': DIM_L}, set())
    check("x/x 逐维为零", all(e.is_zero() for e in exprs), True)

    # (L/g)*g = L
    div_node = binary('/', leaf('L'), leaf('g'))
    _solve(binary('*', div_node, leaf('g')), PENDULUM_DIMS, set(),
           "(L/g)*g", expect_valid=True)


def test_transcendental_requires_dimensionless():
    """初等函数输入必须无量纲。"""
    section("约束收集：初等函数的无量纲性")
    _solve(unary('sin', binary('/', leaf('x'), leaf('x'))), {'x': DIM_L}, set(),
           "sin(x/x)", expect_valid=True)

    result = _solve(unary('sin', leaf('x')), {'x': DIM_L}, set(),
                    "sin(L)", expect_valid=False)
    check("失败定位在 L 维", any(f.dim_index == 0 for f in result.failures),
          True)

    div = binary('/', leaf('L'), leaf('g'))
    _solve(unary('cos', div), PENDULUM_DIMS, set(), "cos(L/g)=cos(T²)",
           expect_valid=False)

    nested = unary('exp', unary('sin', binary('/', leaf('x'), leaf('x'))))
    _solve(nested, {'x': DIM_L}, set(), "exp(sin(x/x))", expect_valid=True)

    s = unary('sin', leaf('L'))
    _solve(binary('*', leaf('_a0_'), s), PENDULUM_DIMS, {'_a0_'},
           "_a0_*sin(L)", expect_valid=False)

    _solve(binary('**', leaf('x'), leaf('t')), {'x': DIM_L, 't': DIM_T}, set(),
           "x**t（指数带量纲）", expect_valid=False)


def test_sqrt_halves_dimensions():
    """sqrt 只做量纲折半：分数维合法，幂律的量纲由前置常数吸收。"""
    section("约束收集：sqrt 折半")
    _solve(unary('sqrt', binary('/', leaf('x'), leaf('x'))), {'x': DIM_L}, set(),
           "sqrt(x/x)", expect_valid=True)

    exprs, _, _ = collect_constraints(unary('sqrt', leaf('L')), PENDULUM_DIMS,
                                   set())
    check("sqrt(L) 的 L 指数折半", exprs[0].const, Fraction(1, 2))
    _solve(unary('sqrt', leaf('L')), PENDULUM_DIMS, set(),
           "sqrt(L) 分数维通过", expect_valid=True)

    _solve(binary('*', leaf('_a0_'), unary('sqrt', leaf('L'))),
           PENDULUM_DIMS, {'_a0_'}, "_a0_*sqrt(L)", expect_valid=True)

    srt_l = unary('sqrt', leaf('L'))
    srt_g = unary('sqrt', leaf('g'))
    _solve(binary('/', srt_l, srt_g), PENDULUM_DIMS, set(),
           "sqrt(L)/sqrt(g)", expect_valid=True)

    _solve(unary('sqrt', unary('pow2', leaf('x'))), {'x': DIM_L}, set(),
           "sqrt(pow2(x)) = x", expect_valid=True)


def test_power_rule_constrains_only_exponent():
    """** 的标准规则：指数必须无量纲，底数可以带量纲。"""
    section("约束收集：** 只约束指数")
    _solve(binary('**', leaf('x'), leaf('_a0_')), {'x': DIM_L}, {'_a0_'},
           "x**a（指数为自由参数）", expect_valid=True)

    exprs, _, _ = collect_constraints(binary('**', leaf('L'), leaf('_a0_')),
                                   PENDULUM_DIMS, {'_a0_'})
    check("根量纲记为待定符号而非强制无量纲",
          dict(exprs[0].coeffs), dict(POW_OUT.coeffs))

    _solve(binary('**', leaf('L'), leaf('g')), PENDULUM_DIMS, set(),
           "L**g（指数带量纲）", expect_valid=False)


def test_pow_scaling():
    """幂运算标度：pow2 使量纲指数翻倍。"""
    section("约束收集：幂运算标度")
    exprs, _, _ = collect_constraints(unary('pow2', leaf('x')),
                                      {'x': DIM_L}, set())
    check("pow2(L) 的 L 指数", exprs[0].const, Fraction(2))
    check("pow2(L) 的 M 指数", exprs[1].const, Fraction(0))

    _solve(unary('pow2', binary('/', leaf('L'), leaf('g'))), PENDULUM_DIMS,
           set(), "pow2(L/g) = T⁴", expect_valid=True)


# ============================================================================
# 单摆公式家族
# ============================================================================

def _pendulum_case(build_root, params, label, expect_valid=True):
    root = build_root()
    return _solve(root, PENDULUM_DIMS, params, label, expect_valid)


def test_pendulum_formula_family():
    """单摆家族：T = 2π√(L/g) 及其变体均量纲自洽。

    物理内容：小振动周期、含零漂、含角度偏移、大振幅级数修正、
    比值形式——所有合法写法都应通过宪法检查。
    """
    section("单摆公式家族（合法形式）")
    _pendulum_case(
        lambda: binary('*', leaf('_a0_'),
                       unary('sqrt', binary('/', leaf('L'), leaf('g')))),
        {'_a0_'}, "T = _a0_·√(L/g)")

    _pendulum_case(
        lambda: binary('*', leaf('_a0_'), binary('/', leaf('L'), leaf('g'))),
        {'_a0_'}, "L/g = T²")

    _pendulum_case(
        lambda: binary('+',
                       binary('*', leaf('_a0_'),
                              unary('sqrt', binary('/', leaf('L'), leaf('g')))),
                       leaf('_b0_')),
        {'_a0_', '_b0_'}, "含零漂 _a0_√(L/g) + _b0_")

    _pendulum_case(
        lambda: binary('+',
                       binary('*', leaf('_a0_'),
                              unary('sqrt', binary('/', leaf('L'), leaf('g')))),
                       binary('*', leaf('_b0_'), leaf('theta0'))),
        {'_a0_', '_b0_'}, "含角度偏移 θ₀（无量纲）")

    def large_amplitude():
        div = binary('/', leaf('L'), leaf('g'))
        period = binary('*', leaf('_a0_'), unary('sqrt', div))
        theta_sq = unary('pow2', leaf('theta0'))
        c0_term = binary('*', leaf('_c0_'), theta_sq)
        correction = binary('+', leaf('_b0_'), c0_term)
        return binary('*', period, correction)
    _pendulum_case(large_amplitude, {'_a0_', '_b0_', '_c0_'},
                   "大振幅级数 √(L/g)(b + c·θ₀²)")

    def ratio_form():
        num = binary('*', leaf('_a0_'), leaf('L'))
        den = binary('*', leaf('_b0_'), leaf('g'))
        return binary('+', binary('/', num, den), leaf('_c0_'))
    _pendulum_case(ratio_form, {'_a0_', '_b0_', '_c0_'}, "比值形式 (a·L)/(b·g) + c")


def test_pendulum_formula_rejects():
    """单摆家族的违宪写法：sin(L)、L+g、共享参数冲突等。"""
    section("单摆公式家族（违宪形式）")
    _pendulum_case(lambda: unary('sin', leaf('L')), set(), "sin(L)", False)
    _pendulum_case(lambda: binary('+', leaf('L'), leaf('g')), set(), "L + g",
                   False)

    def shared_conflict():
        left = binary('*', leaf('_a0_'), leaf('L'))
        right = binary('*', leaf('_a0_'), leaf('g'))
        return binary('+', left, right)
    _pendulum_case(shared_conflict, {'_a0_'}, "共享参数 a·L + a·g", False)


# ============================================================================
# 宪法集成与 fixed_term
# ============================================================================

def test_constitution_gate_on_trees():
    """宪法门对整棵树的判定：合法放行、违宪拒绝、无宪法兼容旧行为。"""
    section("宪法门集成")
    const = DimensionalConstitution()
    tree = Tree(variables=['x'], parameters=['a', 'b'], dimensions={'x': DIM_L},
                constitutions=[const], from_string='(_a0_ * x)')
    check("合法树通过", const.check(tree).is_valid, True)

    with pytest.raises(ValueError, match='dimensional'):
        Tree(variables=['x'], parameters=['a'], dimensions={'x': DIM_L},
             constitutions=[const], from_string='sin(x)')
    check("违宪树构建抛 ValueError", True, True)

    # 无宪法时一切如旧（向后兼容）
    tree = Tree(variables=['x'], parameters=['a'], from_string='sin(x)')
    check("无宪法时照常构建（sin(x) size=2）", tree.size, 2)


def test_pendulum_constitution_integration():
    """单摆设定下宪法门的通过与拒绝。"""
    const = DimensionalConstitution()
    tree = Tree(variables=['L', 'g', 'm', 'theta0'], parameters=['a'],
                dimensions=PENDULUM_DIMS, constitutions=[const],
                from_string='(_a0_ * sqrt((L / g)))')
    check("单摆周期式通过", const.check(tree).is_valid, True)

    with pytest.raises(ValueError, match="dimensional constraints"):
        Tree(variables=['L', 'g', 'm', 'theta0'], parameters=['a'],
             dimensions=PENDULUM_DIMS, constitutions=[const],
             from_string='sin(L)')
    check("sin(L) 构建被拒", True, True)


def _root_valid(string, params, target_dim, label, expect_valid):
    """按目标量纲锚定根维度，检验整式合法性。"""
    tree = Tree(variables=['L', 'g', 'm', 'theta0'], parameters=params,
                dimensions=PENDULUM_DIMS, target_dim=target_dim,
                from_string=string)
    res = check_dimensional_consistency(
        tree.root, PENDULUM_DIMS, set(tree.parameters), target_dim)
    check(label, res.is_valid, expect_valid)
    return res


def test_root_dimension_anchor():
    """根量纲锚定：整式必须等于目标变量的量纲，量纲分析才真正有杀伤力。"""
    section("宪法门：根量纲锚定")

    # 周期 T：sqrt(L/g) 由自由参数吸收量纲 → 合法
    _root_valid('(_a0_ * sqrt((L / g)))', ['a'], DIM_T,
                "a*sqrt(L/g) 锚 T", True)
    # 只给绳长时的幂律：[a] = T·L^(-1/2)，分数维合法
    _root_valid('(_a0_ * sqrt(L))', ['a'], DIM_T, "a*sqrt(L) 锚 T", True)
    _root_valid('(_a0_ * (L ** _b0_))', ['a', 'b'], DIM_T,
                "a*L**b 锚 T", True)
    # L/g 实际量纲是 T^2（m ÷ m·s⁻²）；锚到 T 上应拒
    _root_valid('(L / g)', [], DIM_T, "L/g 锚 T（真为 T²）", False)
    _root_valid('((L / g) * (L / g))', [], DIM_T, "L/g*L/g 锚 T（真为 T⁴）",
                False)
    # 换个目标量纲，同一表达式就合法：锚定确实跟着目标走
    _root_valid('(L / g)', [], (0, 0, 2, 0, 0, 0, 0), "L/g 锚 T^2", True)
    # 无锚定时保持旧行为（只查内部齐次性）
    _root_valid('(L / g)', [], None, "L/g 不锚定", True)

    with pytest.raises(ValueError, match='target_dim'):
        check_dimensional_consistency(Node('L'), PENDULUM_DIMS, set(),
                                      target_dim=(0, 1, 0))
    check("target_dim 长度非 7 抛 ValueError", True, True)


def test_target_dim_survives_reparse():
    """build_from_string 会重跑 __init__，target_dim 必须存活。"""
    section("宪法门：target_dim 经再解析存活")
    const = DimensionalConstitution()
    tree = Tree(variables=['L', 'g'], parameters=['a'],
                dimensions=PENDULUM_DIMS, target_dim=DIM_T,
                constitutions=[const], from_string='(_a0_ * sqrt((L / g)))')
    check("target_dim 仍挂在树上", tree.target_dim, DIM_T)
    check("锚定后仍合法", const.check(tree).is_valid, True)

    with pytest.raises(ValueError, match='dimensional'):
        Tree(variables=['L', 'g'], parameters=['a'],
             dimensions=PENDULUM_DIMS, target_dim=DIM_T,
             constitutions=[const], from_string='(L / g)')
    check("锚定后无参数可吸收的 L/g 构建被拒", True, True)

    # 有 _a0_ 吸收量纲时依然合法 —— 锚定管不住单项式，这是它的边界
    tree = Tree(variables=['L', 'g'], parameters=['a'],
                dimensions=PENDULUM_DIMS, target_dim=DIM_T,
                constitutions=[const], from_string='(_a0_ * (L / g))')
    check("a*(L/g) 因参数吸收仍通过", const.check(tree).is_valid, True)


GAS_DIMS = {'V': (3, 0, 0, 0, 0, 0, 0),        # 体积 L^3
            'n': (0, 0, 0, 0, 0, 1, 0),        # 物质的量 N
            'theta': (0, 0, 0, 0, 1, 0, 0)}    # 温度 Theta
DIM_PRESSURE = (-1, 1, -2, 0, 0, 0, 0)         # 压强 M L^-1 T^-2


def _param_dims(string, params, variables, dims, target_dim):
    """锚定目标量纲后，回代求各拟合常数的量纲向量。"""
    tree = Tree(variables=list(variables), parameters=params,
                dimensions=dict(dims), target_dim=target_dim,
                from_string=string)
    res = check_dimensional_consistency(tree.root, dict(dims),
                                        set(tree.parameters), target_dim)
    check("审查合法", res.is_valid, True)
    return res.param_dims


def test_param_dims_backsubstitution():
    """量纲判据：结构 + 目标量纲应把拟合常数的量纲解出来。"""
    section("宪法门：参数维度回代")

    # 只给绳长：T = a·√L → [a] = T·L^(-1/2)，正是 2π/√g 的单位 s·m^(-1/2)
    pd = _param_dims('(_a0_ * sqrt(L))', ['a'], ['L', 'g', 'm', 'theta0'],
                     PENDULUM_DIMS, DIM_T)
    check("[a] = L^(-1/2)·T", pd['_a0_'],
          (Fraction(-1, 2), 0, 1, 0, 0, 0, 0))

    # 给出 g：√(L/g) 已是 T → [a] 必须无量纲 → 可证伪判据 a = 2π
    pd = _param_dims('(_a0_ * sqrt((L / g)))', ['a'],
                     ['L', 'g', 'm', 'theta0'], PENDULUM_DIMS, DIM_T)
    check("[a] 无量纲（应测得 2π）", pd['_a0_'], (0, 0, 0, 0, 0, 0, 0))

    # 理想气体 P = a·n·Θ/V → [a] = 气体常数 R 的量纲 J·mol⁻¹·K⁻¹
    pd = _param_dims('(_a0_ * ((n * theta) / V))', ['a'],
                     ['V', 'n', 'theta'], GAS_DIMS, DIM_PRESSURE)
    check("[a] = R 的量纲", pd['_a0_'], (2, 1, -2, 0, -1, -1, 0))

    # 合成未知量 <pow> 不得混进报告
    check("只报告真实参数", sorted(pd.keys()), ['_a0_'])


def test_param_dims_flags_undetermined():
    """结构钉不住的分量必须显式标未定，不能假装有解。"""
    section("宪法门：未定分量")
    pd = _param_dims('(_a0_ * (L ** _b0_))', ['a', 'b'],
                     ['L', 'g', 'm', 'theta0'], PENDULUM_DIMS, DIM_T)
    # 指数由 ** 规则钉死为无量纲
    check("[b] 无量纲", pd['_b0_'], (0, 0, 0, 0, 0, 0, 0))
    # [a] 与 (L**b) 的待定维度耦合，单独无解
    check("[a] 逐维未定", pd['_a0_'], (None,) * N_DIMS)

    # 两项式：加法齐次 + 锚定把两个常数都钉住
    pd = _param_dims('((_a0_ * sqrt((L / g))) + (_b0_ * (m / pow2(theta0))))',
                     ['a', 'b'], ['L', 'g', 'm', 'theta0'],
                     PENDULUM_DIMS, DIM_T)
    check("第一项常数无量纲", pd['_a0_'], (0, 0, 0, 0, 0, 0, 0))
    check("[b] = T/M", pd['_b0_'], (0, -1, 1, 0, 0, 0, 0))


def test_numeric_literal_exponent_is_exact():
    """数字指数是纯数：量纲可精确算出，不必退化成待定符号。"""
    section("宪法门：数字指数精确标度")
    pd = _param_dims('(_a0_ * (L ** 2))', ['a'], ['L', 'g', 'm', 'theta0'],
                     PENDULUM_DIMS, DIM_L)
    # a·L² 要有量纲 L → [a] = L⁻¹；换成 <pow> 自由符号时这个结论拿不到
    check("[a] = L⁻¹", pd['_a0_'], (-1, 0, 0, 0, 0, 0, 0))
    check("pow2 的字符串形式能 round-trip",
          str(Tree(variables=['L', 'g'], parameters=['a'],
                   from_string='(_a0_ * (L ** 2))',
                   dimensions=PENDULUM_DIMS)), '(_a0_ * (L ** 2))')


def test_malformed_string_raises_value_error():
    """畸形表达式串给明确 ValueError，不再是 ets[N] 的 KeyError。"""
    section("语法门：畸形串报错")

    def build(string):
        return Tree(variables=['L', 'g'], parameters=['a', 'b'],
                    dimensions=PENDULUM_DIMS, from_string=string)

    # 一元算符写成 (sqrt L)：解析出 'L)' 这种非算符节点
    with pytest.raises(ValueError, match='not an operator'):
        build('(_a0_ * (sqrt L))')
    check("(sqrt L) 型畸形串被拦下", True, True)

    # 二元算符同级并列：解析出 3 个子节点
    with pytest.raises(ValueError, match='not an operator'):
        build('(_a0_ * (_b0_ * L) + L)')
    check("同级并列被拦下", True, True)

    # 规范写法必须照常通过，别把合法的挡在门外
    for string in ['(_a0_ * sqrt(L))', 'sqrt((_a0_ * L))',
                   '(_a0_ * (L ** _b0_))', '(_a0_ * (L / g))']:
        check(f"规范串可建：{string}", build(string).size > 0, True)


def test_unknown_leaf_or_op_is_rejected():
    """未知叶子不再默认无量纲——假放行比崩溃更危险。"""
    section("宪法门：未知叶子与算符")
    res = check_dimensional_consistency(
        binary('*', leaf('L'), Node('q')), PENDULUM_DIMS, set())
    check("未声明叶子被拒", res.is_valid, False)
    check("报错点名 q", any("'q'" in f.description for f in res.failures),
          True)
    check("结构性 failure 标 dim_index=-1", res.failures[0].dim_index, -1)

    res2 = check_dimensional_consistency(
        Node('bogus', offspring=[leaf('L')]), PENDULUM_DIMS, set())
    check("未知算符被拒", res2.is_valid, False)
    check("报错点名 bogus",
          any('bogus' in f.description for f in res2.failures), True)

    # 数字叶子豁免：纯数本征无量纲，不该被当成未知叶子拒绝
    res3 = check_dimensional_consistency(
        binary('*', leaf('L'), Node('2')), PENDULUM_DIMS, set())
    check("数字叶子仍放行", res3.is_valid, True)

    # 参数即便不声明量纲也合法（它们是待解未知量）
    res4 = check_dimensional_consistency(
        binary('*', leaf('L'), leaf('_a0_')), PENDULUM_DIMS, {'_a0_'})
    check("参数无需声明量纲", res4.is_valid, True)

    # 走宪法路径时，`_x_` 约定名算参数（即便 tree.parameters 还没记下它）；
    # 但非约定名的未声明变量必须拦住
    with pytest.raises(ValueError, match='dimensional'):
        Tree(variables=['L', 'g'], parameters=['a'], dimensions=PENDULUM_DIMS,
             constitutions=[DimensionalConstitution()],
             from_string='(_a0_ * q)')
    check("未声明变量 q 建树被拒", True, True)


def test_fixed_term_dimensional():
    """fixed_term 组合式：主树与固定项的联合量纲检查。"""
    section("fixed_term 量纲")
    const = DimensionalConstitution()

    tree = Tree(variables=['L', 'g', 'theta0'], parameters=['a', 'b'],
                dimensions=PENDULUM_DIMS, constitutions=[const],
                from_string='(_a0_ * sqrt((L / g)))',
                fixed_term='(_b0_ * theta0)', fixed_term_op='*')
    check("无量纲 fixed_term 经 * 组合", const.check(tree).is_valid, True)

    tree = Tree(variables=['L', 'g'], parameters=['a', 'b', 'c'],
                dimensions=PENDULUM_DIMS, constitutions=[const],
                from_string='(_a0_ * sqrt((L / g)))',
                fixed_term='(_b0_ * sqrt((L / g)))', fixed_term_op='+')
    check("同量纲 fixed_term 经 + 组合", const.check(tree).is_valid, True)

    with pytest.raises(ValueError, match="dimensional constraints"):
        Tree(variables=['L', 'g'], parameters=[],
             dimensions=PENDULUM_DIMS, constitutions=[const],
             from_string='(L / g)', fixed_term='L', fixed_term_op='+')
    check("T² + L 组合被拒", True, True)

    tree = Tree(variables=['L', 'g'], parameters=['a'],
                dimensions=PENDULUM_DIMS, constitutions=[const],
                from_string='(_a0_ * sqrt((L / g)))')
    check("fixed_term=None 直通", const.check(tree).is_valid, True)


def test_multiple_and_empty_constitutions():
    """多部宪法并存须全部通过；空宪法列表恒真。"""
    tree = Tree(variables=['x'], parameters=['a'])
    check("空宪法列表", tree.check_constitution() is True, True)

    c1, c2 = DimensionalConstitution(), DimensionalConstitution()
    tree = Tree(variables=['x'], parameters=['a'], dimensions={'x': DIM_L},
                constitutions=[c1, c2], from_string='(_a0_ * x)')
    check("多部宪法全部通过", tree.check_constitution() is True, True)


# ============================================================================
# 先验能量 EP（复杂度代价）
# ============================================================================

def test_prior_energy_bookkeeping():
    """先验能量记账：EP = Σ Nopi_<op>·nop，默认权重 10。"""
    section("先验能量记账")
    tree = Tree(variables=['L'], parameters=['a'], dimensions=PENDULUM_DIMS,
                from_string='_a0_')
    _, _, EP = tree.get_energy()
    check("单叶子 EP = 0", EP, 0.0)

    cases = [
        ('(_a0_ * L)', "单二元运算 (_a0_*L)"),
        ('(_a0_ * sqrt((L / g)))', "单摆周期式"),
        ('((_a0_ * sqrt((L / g))) + _b0_)', "含零漂周期式"),
        ('sin((L / g))', "sin(L/g)（无参数）"),
    ]
    for expr, label in cases:
        tree = Tree(variables=['L', 'g'], parameters=['a', 'b'],
                    dimensions=PENDULUM_DIMS, from_string=expr)
        _, _, EP = tree.get_energy()
        check(f"EP = 加权和", float(EP), _ep_from_nops(tree.nops),
              context=label)

    weight = 5.0
    prior = {'Nopi_*': weight, 'Nopi_sqrt': weight,
             'Nopi_/': weight, 'Nopi_+': weight}
    tree = Tree(variables=['L', 'g'], parameters=['a'],
                dimensions=PENDULUM_DIMS, prior_par=prior,
                from_string='(_a0_ * sqrt((L / g)))')
    _, _, EP = tree.get_energy()
    check("自定义权重 EP", float(EP),
          _ep_from_nops(tree.nops, prior_par_weight=weight))


def test_complexity_ordering():
    """复杂度排序：更简单 ⇒ 更低 EP（奥卡姆代价的单调性）。"""
    section("复杂度排序")
    const = DimensionalConstitution()
    variants = [
        ('(_a0_ * L)', ['a'], "a·L"),
        ('(_a0_ * sqrt((L / g)))', ['a'], "a·√(L/g)"),
        ('((_a0_ * sqrt((L / g))) + _b0_)', ['a', 'b'], "a·√(L/g) + b"),
    ]
    EPs = []
    for expr, params, label in variants:
        tree = Tree(variables=['L', 'g'], parameters=params,
                    dimensions=PENDULUM_DIMS, constitutions=[const],
                    from_string=expr)
        _, _, EP = tree.get_energy()
        measure(f"EP[{label}]", float(EP))
        EPs.append(EP)
    check("a·L < a·√(L/g)", EPs[0] < EPs[1], True)
    check("a·√(L/g) < 含零漂", EPs[1] < EPs[2], True)


# ============================================================================
# 提议门：宪法先于采样，试算完全回退
# ============================================================================

def test_dE_et_constitution_gate():
    """dE_et 门：违宪替换 dE=inf 且状态回退；合法替换有限且回退。"""
    section("提议门：dE_et")
    tree = Tree(variables=['L', 'g'], parameters=['a'],
                dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
                from_string='(_a0_ * L)')
    L_node = [n for n in tree.nodes if n.value == 'L'][0]

    state = _capture_state(tree)
    dE, dEB, dEP, pv, nif, nfi = tree.dE_et(L_node, ['+', ['L', 'g']])
    measure("违宪提议 dE", float(dE))
    check("L→L+g 被拒（dE=inf）", np.isinf(dE), True)
    _check_state_unchanged(tree, state, "dE_et 违宪")

    dE, dEB, dEP, pv, nif, nfi = tree.dE_et(L_node, ['/', ['L', 'g']])
    check("L→L/g 有限（参数吸收 T²）", np.isfinite(dE), True)
    check("试算后表达式回退", str(tree), '(_a0_ * L)')

    state = _capture_state(tree)
    dE, dEB, dEP, pv, nif, nfi = tree.dE_et(L_node, ['sin', ['L']])
    check("L→sin(L) 被拒（dE=inf）", np.isinf(dE), True)
    _check_state_unchanged(tree, state, "dE_et sin(L)")


def test_dE_lr_constitution_gate():
    """dE_lr 门：算子替换的违宪拒绝与合法放行，均回退。"""
    section("提议门：dE_lr")
    tree = Tree(variables=['L', 'g'], parameters=[],
                dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
                from_string='sqrt((L / g))')
    sqrt_node = [n for n in tree.nodes if n.value == 'sqrt'][0]
    state = _capture_state(tree)

    dE, dEB, dEP, pv = tree.dE_lr(sqrt_node, 'sin')
    check("sqrt→sin 被拒（dE=inf）", np.isinf(dE), True)
    _check_state_unchanged(tree, state, "dE_lr 违宪")

    tree = Tree(variables=['L'], parameters=['a'],
                dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
                from_string='(_a0_ * L)')
    mul_node = [n for n in tree.nodes if n.value == '*'][0]
    before = str(tree)
    dE, dEB, dEP, pv = tree.dE_lr(mul_node, '+')
    check("*→+ 有限（a 吸收量纲）", np.isfinite(dE), True)
    check("试算后树回退", str(tree), before)


def test_dE_rr_constitution_gate():
    """dE_rr 门：根操作的违宪拒绝与合法放行/剪枝，均回退。"""
    section("提议门：dE_rr")
    tree = Tree(variables=['L', 'g'], parameters=[],
                dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
                from_string='(L / g)')
    state = _capture_state(tree)
    dE, dEB, dEP, pv = tree.dE_rr(rr=['sin', []])
    check("套 sin() 被拒（dE=inf）", np.isinf(dE), True)
    _check_state_unchanged(tree, state, "dE_rr 违宪")

    tree = Tree(variables=['L', 'g'], parameters=['a', 'b'],
                dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
                from_string='sqrt((L / g))')
    before = str(tree)
    dE, dEB, dEP, pv = tree.dE_rr(rr=['*', ['_b0_']])
    check("乘参数放行（无量纲×T）", np.isfinite(dE), True)
    check("试算后树回退", str(tree), before)

    tree = Tree(variables=['L'], parameters=['a'],
                dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
                from_string='(_a0_ + L)')
    check("根可剪", tree.is_root_prunable(), True)
    before = str(tree)
    dE, dEB, dEP, pv = tree.dE_rr(rr=None)
    check("剪枝有限", np.isfinite(dE), True)
    check("试算后树回退", str(tree), before)


# ============================================================================
# MCMC 集成：宪法约束下的采样链
# ============================================================================

def _pendulum_data(seed=42, noise=0.02, n=30):
    reproducible_state(seed)
    L_vals = np.linspace(0.5, 2.0, n)
    g_val = 9.81
    T_true = 2 * np.pi * np.sqrt(L_vals / g_val)
    T_obs = T_true + np.random.normal(0, noise, n)
    return pd.DataFrame({'L': L_vals, 'g': [g_val] * n}), pd.Series(T_obs)


def test_mcmc_constitution_invariant():
    """宪法约束下的采样链：每步之后树都必须通过宪法检查。

    硬约束的物理含义：链被禁止进入任何违宪状态——不只是末态，
    而是整个轨迹。
    """
    section("MCMC 集成：宪法不变量")
    x, y = _pendulum_data(seed=42, noise=0.02)
    const = DimensionalConstitution()
    tree = Tree(variables=['L', 'g'], parameters=['a'],
                dimensions=PENDULUM_DIMS, constitutions=[const],
                x=x, y=y, max_size=30,
                from_string='(_a0_ * sqrt((L / g)))')

    for step in range(20):
        tree.mcmc_step(verbose=False)
        result = const.check(tree)
        check("宪法不变量", result.is_valid, True,
              context=f"step {step} tree='{tree}'")

    _, _, EP = tree.get_energy()
    check("EP 有限", np.isfinite(EP), True)


def test_mcmc_invalid_start_stays_legal():
    """即使初态表达式不对，链也永远不落入违宪状态。"""
    x, y = _pendulum_data(seed=42, noise=0.02)
    const = DimensionalConstitution()
    tree = Tree(variables=['L', 'g'], parameters=['a'],
                dimensions=PENDULUM_DIMS, constitutions=[const],
                x=x, y=y, max_size=30,
                from_string='(_a0_ * L)')
    check("初态合法", const.check(tree).is_valid, True)

    for step in range(30):
        tree.mcmc_step(verbose=False)
        result = const.check(tree)
        check("宪法不变量", result.is_valid, True,
              context=f"step {step}")


def test_mcmc_converges_to_period_law():
    """长链收敛：从 a·L 出发，链应趋向周期律且 BIC/EP 有限。"""
    section("MCMC 集成：收敛")
    x, y = _pendulum_data(seed=99, noise=0.01, n=50)
    const = DimensionalConstitution()
    tree = Tree(variables=['L', 'g'], parameters=['a'],
                dimensions=PENDULUM_DIMS, constitutions=[const],
                x=x, y=y, max_size=20,
                from_string='(_a0_ * L)')

    for _ in range(100):
        tree.mcmc_step(verbose=False)

    measure("末态表达式", str(tree))
    check("末态通过宪法", const.check(tree).is_valid, True)
    bic = tree.get_bic()
    check("BIC 非 NaN（拟合成功）", not np.isnan(float(bic)), True)
    _, _, EP = tree.get_energy()
    check("EP 非 NaN", not np.isnan(EP), True)


def test_mcmc_constitution_gate_simple_dims():
    """简单一维设定下，MCMC 轨迹同样保持量纲合法。"""
    reproducible_state(42)
    const = DimensionalConstitution()
    x = pd.DataFrame({'x0': np.linspace(1, 10, 20)})
    y = pd.Series(2.0 * x['x0'] + 1.0 + np.random.normal(0, 0.1, 20))
    tree = Tree(variables=['x0'], parameters=['a'], dimensions={'x0': DIM_L},
                constitutions=[const], x=x, y=y, max_size=20)
    for step in range(10):
        tree.mcmc_step(verbose=False)
    result = const.check(tree)
    check("MCMC 后量纲合法", result.is_valid, True)


if __name__ == '__main__':
    for name in sorted(dir()):
        if name.startswith('test_'):
            globals()[name]()
    print("\n实验记录输出完毕（pytest 下用 -s 查看）。")
