# prior/coverage.py
"""构型覆盖率诊断：回答"语料到底覆盖了多少种可表达的树类型"。

*BMS2020 的先验来自 wiki 公式语料。语料覆盖得全不全，原方案里没人问过——
4080 条公式听起来很多，但"所有可表达的表达式树类型"是一个**有限可枚举**
的集合，可以逐个数出来对账。本模块就是那把尺子。

树类型（局部构型）的定义
------------------------
表达式树空间 F(OPS, V, P) 是无限的（深度无界），**无法**枚举树本身。但树
是由有限的局部构型拼起来的，构型集合是有限的：

* ``root:<label>`` —— 根节点的标签；
* ``slot:<parent>|<i>|<child>`` —— 父算子 ``parent`` 的第 ``i`` 个子槽位上
  放了 ``child``。

``label`` 取自 ``OPS`` 的算子名，或三种叶子类别 ``VAR`` / ``PARAM`` / ``NUM``。
注意叶子只区分类别、不区分具体符号，这样构型空间的大小与变量数 nv、参数数
np **无关**——这正是相对原方案的关键改进之一（原方案的先验和 nv/np 绑死）。

规模：本项目 18 个算子（15 一元 + 3 二元）+ 3 种叶子 = 21 个标签；

* 根构型 21 个；
* 子槽构型 ``(Σ_o arity(o)) × 21 = 21 × 21 = 441`` 个；
* 合计 **462** 个构型。

任何一棵树都唯一分解为"一个根构型 + 一组子槽构型的多重集"，所以这 462 个
构型**生成闭包 = 整个树空间**（截断到 max_size 后严格成立）。覆盖率因此可以
被精确定义和精确测量：语料里出现过多少种 / 462。

为什么这比"4080 条公式"更能说明问题
----------------------------------
语料条数多不等于构型覆盖全。公式长度集中在 5–15 个节点，用到的构型高度
重复（`+` 套变量、`*` 套变量），而深嵌套、稀有算子（`fac`、`tanh`）相关的
构型几乎是空的。于是先验在那些构型上等于没有信息——只能靠平滑兜底。
跑一次 :func:`coverage_report` 就能把这个"空洞地图"画出来。
"""

from collections import Counter

from sympy import sympify, Add, Mul, Pow, Symbol, Number, Rational

from .fit import _FUNC_TO_OP

LEAF_VAR = 'VAR'
LEAF_PARAM = 'PARAM'
LEAF_NUM = 'NUM'
LEAF_KINDS = (LEAF_VAR, LEAF_PARAM, LEAF_NUM)

OTHER = 'OTHER'          # OPS 之外的函数，无法归类到任何构型


# ---------------------------------------------------------------------------
# 构型空间
# ---------------------------------------------------------------------------

def config_space(ops=None, leaf_kinds=LEAF_KINDS):
    """枚举全部局部构型，返回构型键列表。

    Returns
    -------
    list of str
        ``['root:<label>', ..., 'slot:<parent>|<i>|<child>', ...]``
    """
    if ops is None:
        from core.constants import OPS
        ops = list(OPS.keys())
        arity = OPS
    else:
        ops = list(ops)
        from core.constants import OPS as _OPS
        arity = _OPS
    labels = list(ops) + list(leaf_kinds)
    keys = ['root:%s' % lab for lab in labels]
    for p in ops:
        for i in range(arity.get(p, 1)):
            for c in labels:
                keys.append('slot:%s|%d|%s' % (p, i, c))
    return keys


def n_configs(ops=None, leaf_kinds=LEAF_KINDS):
    """构型总数（本项目默认 462）。"""
    return len(config_space(ops=ops, leaf_kinds=leaf_kinds))


# ---------------------------------------------------------------------------
# 语料统计
# ---------------------------------------------------------------------------

def _leaf_kind(expr):
    if isinstance(expr, Number):
        return LEAF_NUM
    name = getattr(expr, 'name', '')
    if name.startswith('_') and name.endswith('_'):
        return LEAF_PARAM
    return LEAF_VAR


def _record(counter, parent, slot, child):
    counter['slot:%s|%d|%s' % (parent, slot, child)] += 1


def _walk(expr, counter, unmapped):
    """递归走一棵 sympy 表达式，记录全部子槽构型，返回该节点的标签。

    sympy 的 n 元 ``Add`` / ``Mul`` 按**左结合**拆成二叉树
    （``((a+b)+c)``），与 BMS 的二元算子树一致。
    """
    if isinstance(expr, Number) or isinstance(expr, Symbol):
        return _leaf_kind(expr)

    if isinstance(expr, Add):
        args = expr.args
        if len(args) == 1:
            return _walk(args[0], counter, unmapped)
        acc = _walk(args[0], counter, unmapped)
        for a in args[1:]:
            _record(counter, '+', 0, acc)
            _record(counter, '+', 1, _walk(a, counter, unmapped))
            acc = '+'
        return '+'

    if isinstance(expr, Mul):
        coeff = 1
        factors, divflags = [], []
        for a in expr.args:
            if isinstance(a, Pow) and a.exp == -1:
                factors.append(a.base)
                divflags.append(True)
            elif a.is_Number:
                coeff *= int(a) if a.is_Integer else float(a)
            else:
                factors.append(a)
                divflags.append(False)
        if not factors:
            return LEAF_NUM
        acc = _walk(factors[0], counter, unmapped)
        if coeff < 0:
            _record(counter, '-', 0, acc)
            acc = '-'
        for f, isdiv in zip(factors[1:], divflags[1:]):
            op = '/' if isdiv else '*'
            _record(counter, op, 0, acc)
            _record(counter, op, 1, _walk(f, counter, unmapped))
            acc = op
        return acc

    if isinstance(expr, Pow):
        e = expr.exp
        if e == 2:
            op, children = 'pow2', [expr.base]
        elif e == 3:
            op, children = 'pow3', [expr.base]
        elif e == Rational(1, 2) or e == 0.5:
            op, children = 'sqrt', [expr.base]
        else:
            op, children = '**', [expr.base, e]
        for i, ch in enumerate(children):
            _record(counter, op, i, _walk(ch, counter, unmapped))
        return op

    name = getattr(expr.func, '__name__', None)
    op = _FUNC_TO_OP.get(name)
    if op is None:
        unmapped.add(str(name))
        for a in expr.args:
            _walk(a, counter, unmapped)
        return OTHER
    for i, a in enumerate(expr.args):
        _record(counter, op, i, _walk(a, counter, unmapped))
    return op


def count_configs_in_corpus(formulas, ops=None):
    """统计语料里每个构型出现了多少次。

    Returns
    -------
    (Counter, set, int)
        ``(构型计数, OPS 之外的函数名, 成功解析的公式数)``。
    """
    counter = Counter()
    unmapped = set()
    n = 0
    for f in formulas:
        try:
            ex = sympify(f) if isinstance(f, str) else f
        except Exception:
            unmapped.add('UNPARSEABLE: %s' % str(f)[:60])
            continue
        n += 1
        counter['root:%s' % _walk(ex, counter, unmapped)] += 1
    return counter, unmapped, n


def coverage_report(formulas, ops=None, top_uncovered=20):
    """语料对构型空间的覆盖率诊断。

    Returns
    -------
    dict
        ``{'n_configs': 构型总数, 'n_covered': 出现的构型数,
        'coverage': 覆盖率, 'n_samples': 构型出现总次数,
        'n_formulas': 公式条数, 'by_parent': {父算子: (已覆盖/总数)},
        'uncovered_examples': 未覆盖构型的样例,
        'out_of_ops': OPS 外函数名}``
    """
    if ops is None:
        from core.constants import OPS
        ops = list(OPS.keys())
    space = config_space(ops=ops)
    counts, unmapped, n_formulas = count_configs_in_corpus(formulas, ops=ops)

    covered = [k for k in space if counts.get(k, 0) > 0]
    uncovered = [k for k in space if counts.get(k, 0) == 0]

    by_parent = {}
    for p in ops:
        tot = sum(1 for k in space if k.startswith('slot:%s|' % p))
        hit = sum(1 for k in space
                  if k.startswith('slot:%s|' % p) and counts.get(k, 0) > 0)
        by_parent[p] = (hit, tot)

    return {
        'n_configs': len(space),
        'n_roots': sum(1 for k in space if k.startswith('root:')),
        'n_covered': len(covered),
        'coverage': len(covered) / float(len(space)),
        'n_samples': int(sum(counts.values())),
        'n_formulas': n_formulas,
        'by_parent': by_parent,
        'uncovered_examples': uncovered[:top_uncovered],
        'n_uncovered': len(uncovered),
        'out_of_ops': sorted([u for u in unmapped
                              if not u.startswith('UNPARSEABLE')]),
    }


def format_report(rep):
    """把 :func:`coverage_report` 的结果渲染成可读文本。"""
    lines = [
        '构型空间: %d 个（根构型 %d / 子槽构型 %d）'
        % (rep['n_configs'], rep['n_roots'],
           rep['n_configs'] - rep['n_roots']),
        '语料: %d 条公式，产生 %d 次构型出现'
        % (rep['n_formulas'], rep['n_samples']),
        '覆盖: %d / %d = %.1f%%   （未覆盖 %d 个）'
        % (rep['n_covered'], rep['n_configs'], 100 * rep['coverage'],
           rep['n_uncovered']),
        '',
        '按父算子拆分的覆盖率:',
    ]
    for p, (hit, tot) in sorted(rep['by_parent'].items(),
                                key=lambda kv: kv[1][0] / float(kv[1][1])):
        lines.append('  %-6s %3d / %3d  (%5.1f%%)'
                     % (p, hit, tot, 100 * hit / float(tot)))
    if rep['out_of_ops']:
        lines += ['', '语料中出现但 OPS 表里没有的函数: %s'
                  % ', '.join(rep['out_of_ops'])]
    return '\n'.join(lines)
