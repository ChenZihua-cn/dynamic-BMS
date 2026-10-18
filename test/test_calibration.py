"""仪器校准：核心数据结构的冒烟检查与规范不变性。

物理类比 —— 正式实验前校准仪器。若这些基本不变量被破坏，
后续所有实验（守恒律、定律恢复、量纲检查）的结果均不可信。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core import Tree, Node, OPS
from test.lab import check, measure, section

PRIOR_FLAT = dict([('Nopi_%s' % op, 0) for op in OPS])
PRIOR_UNIT = dict([('Nopi_%s' % op, 1.0) for op in OPS])


def test_import():
    """冒烟检查：仪器存在（core 模块可导入、非空）。"""
    section("模块导入")
    check("Tree 可导入", Tree is not None, True)
    check("Node 可导入", Node is not None, True)
    check("OPS 操作符表非空", len(OPS) > 0, True)


def test_node_print():
    """节点打印格式：叶子与二元算子。"""
    section("节点打印")
    check("叶子 pr()", Node('x', offspring=[]).pr(), 'x')
    check("二元算子 pr()",
          Node('+', offspring=[Node('x'), Node('y')]).pr(), '(x + y)')


def test_tree_birth():
    """初态构造：空先验下单节点树；字符串构造 (x + _a0_) 为三节点。"""
    section("初态构造")
    t = Tree(prior_par=PRIOR_FLAT)
    check("空树 size", t.size, 1)
    t2 = Tree(prior_par=PRIOR_FLAT, from_string='(x + _a0_)')
    check("(x + _a0_) size", t2.size, 3)


def test_canonical_reparametrization():
    """规范选择：canonical 形式将自由参数重命名为 c1, c2, ..."""
    section("规范不变性 (canonical form)")
    t1 = Tree(prior_par=PRIOR_FLAT, from_string='(x + _a0_)')
    can1 = t1.canonical()
    check("参数重命名为 c1", 'c1' in can1, True)
    check("不残留原参数名", '_a0_' not in can1, True)


def test_canonical_idempotence():
    """幂等性：重复施加规范变换不产生新信息。"""
    t1 = Tree(prior_par=PRIOR_FLAT, from_string='(x + _a0_)')
    check("canonical 幂等", t1.canonical(), t1.canonical())


def test_canonical_commutation():
    """对称性：交换律等价的表达式必须落在同一规范形
    （否则简并检测会把等价公式当作不同公式）。"""
    can1 = Tree(prior_par=PRIOR_FLAT, from_string='(x + _a0_)').canonical()
    can2 = Tree(prior_par=PRIOR_FLAT, from_string='(_a0_ + x)').canonical()
    check("(x + a) 与 (a + x) 同规范形", can1, can2)


def test_canonical_distinctness():
    """可区分性：不同表达式的规范形必须不同
    （否则简并检测误杀合法公式）。"""
    can1 = Tree(prior_par=PRIOR_FLAT, from_string='(x + _a0_)').canonical()
    can3 = Tree(prior_par=PRIOR_FLAT, from_string='(x * _a0_)').canonical()
    check("(x + a) 与 (x * a) 规范形不同", can1 != can3, True)


def test_state_reconstruction():
    """态重构可逆性：MCMC 演化后的树序列化再重建，表达式逐字一致。"""
    section("演化后的态重构")
    t_rt = Tree(prior_par=PRIOR_UNIT, from_string='(x0 + _a0_)')
    for _ in range(20):
        t_rt.mcmc_step(verbose=False)
    expr1 = str(t_rt)
    measure("演化 20 步后的表达式", expr1)
    t_rt2 = Tree(prior_par=PRIOR_UNIT, from_string=expr1)
    check("序列化往返一致", str(t_rt2), expr1)


if __name__ == '__main__':
    for name in sorted(dir()):
        if name.startswith('test_'):
            globals()[name]()
    print("\n实验记录输出完毕（pytest 下用 -s 查看）。")
