# prior/fit.py
"""从公式语料计算先验超参数。

:BMS2020 的 ``a_o`` / ``b_o`` 是从 Wikipedia 公式语料里"数"出来的。本模块
把这件事做成可复现的函数，有两个层次：

``fit_unigram_prior``
    闭式估计。把公式里算子的出现当成一个 unigram 语言模型，
    ``a_o = -log p_o``（``p_o`` 带 add-k 平滑）。
    **这是惩罚力度的下界**：它只算"每个算子有多常见"，不算"有多少棵不同的树
    恰好用到这么多次这个算子"。算子计数组合起来能对应的树的数量随计数爆炸
    增长，真实的 ``a_o`` 必须更大才能抵消这种组合爆炸。[BMS2020] 官方值
    （3.3–10 nat）明显大于典型语料频率给出的值，差额就来自这里。

``fit_moment_matched_prior``
    矩匹配（[BMS2020] 式 14）。用重要性采样估计当前先验下生成表达式的
    ``<n_o>``，与语料的 ``<n_o>`` 比较后迭代修正 ``a_o``。因为是在**树的
    生成分布**上做的期望，组合多重性被自动计入，得到的量级与官方值可比。
    代价：这是随机近似，需要看有效样本量（ESS）判断结果是否可信。

两者都返回标准 ``prior_par`` 字典，可直接 :func:`prior.io.write_prior_par`
落盘，也可直接喂给 ``Tree(prior_par=...)``。
"""

import math
from collections import Counter

import numpy as np
from sympy import sympify, Add, Mul, Pow, Symbol, Number, Integer, Float
from sympy import Rational
from sympy import sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, Abs, factorial

# sympy 函数名 → OPS 算子名
_FUNC_TO_OP = {
    'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
    'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh',
    'exp': 'exp', 'log': 'log',
    'Abs': 'abs', 'sqrt': 'sqrt', 'factorial': 'fac',
}


# ---------------------------------------------------------------------------
# 语料统计
# ---------------------------------------------------------------------------

def _count_node(expr, counter, unmapped):
    """递归统计一棵 sympy 表达式树里各 OPS 算子的出现次数。

    sympy 的 ``Add`` / ``Mul`` 是 n 元的，而 BMS 树里 ``+`` / ``*`` 是二元的，
    所以 n 个求和项对应 ``n - 1`` 个 ``+`` 节点。除法在 sympy 里是
    ``Pow(base, -1)``，单独识别为 ``/``；负的数值系数识别为一元 ``-``。
    """
    if isinstance(expr, (Number, Integer, Float)) or expr.is_Number:
        return
    if isinstance(expr, Symbol):
        return

    if isinstance(expr, Add):
        counter['+'] += len(expr.args) - 1
        for a in expr.args:
            _count_node(a, counter, unmapped)
        return

    if isinstance(expr, Mul):
        # sympy 把 x/y 表示成 Mul(x, Pow(y, -1))，把 -x 表示成 Mul(-1, x)。
        # 先把数值系数提出来（负系数 = 一元 '-'），再按因子个数分配 '*' 与 '/'。
        coeff = 1
        factors = []
        ndiv = 0
        for a in expr.args:
            if isinstance(a, Pow) and a.exp == -1:
                ndiv += 1
                factors.append(a.base)
            elif a.is_Number:
                coeff *= int(a) if a.is_Integer else float(a)
            else:
                factors.append(a)
        internal = max(0, len(factors) - 1)
        counter['/'] += ndiv
        counter['*'] += max(0, internal - ndiv)
        if coeff < 0:
            counter['-'] += 1
        for f in factors:
            _count_node(f, counter, unmapped)
        return

    if isinstance(expr, Pow):
        e = expr.exp
        if e == 2:
            counter['pow2'] += 1
        elif e == 3:
            counter['pow3'] += 1
        elif e == Rational(1, 2) or e == 0.5:
            counter['sqrt'] += 1      # sympy 把 sqrt(x) 存成 Pow(x, 1/2)
        elif e == -1:
            counter['/'] += 1
        else:
            counter['**'] += 1
        _count_node(expr.base, counter, unmapped)
        if e not in (2, 3, -1) and not (e == Rational(1, 2) or e == 0.5):
            _count_node(e, counter, unmapped)
        return

    name = getattr(expr.func, '__name__', None)
    op = _FUNC_TO_OP.get(name)
    if op is not None:
        counter[op] += 1
        for a in expr.args:
            _count_node(a, counter, unmapped)
        return

    # 其它函数（asin / erf / ...）不在 OPS 里，记下来供人工核查
    unmapped.add(str(name))
    for a in expr.args:
        _count_node(a, counter, unmapped)


def count_ops_in_corpus(formulas, ops=None):
    """统计一批公式里各算子的出现次数。

    Parameters
    ----------
    formulas : iterable
        公式，可以是字符串（走 ``sympify``）或 sympy 表达式。
    ops : iterable or None
        只保留这些算子（``None`` 时用 ``core.constants.OPS``）。

    Returns
    -------
    (Counter, set)
        算子计数，以及语料中出现但 OPS 表里没有的函数名集合。
    """
    if ops is None:
        from core.constants import OPS
        ops = list(OPS.keys())
    counter = Counter(dict([(o, 0) for o in ops]))
    unmapped = set()
    parsed = 0
    for f in formulas:
        try:
            ex = sympify(f) if isinstance(f, str) else f
        except Exception:
            unmapped.add('UNPARSEABLE: %s' % str(f)[:60])
            continue
        parsed += 1
        _count_node(ex, counter, unmapped)
    counter['_n_formulas'] = parsed
    return counter, unmapped


def corpus_moments(formulas, ops=None):
    """语料的算子频率（一阶矩的归一化版本）。

    Returns
    -------
    (dict, dict, int)
        ``({op: 平均出现次数}, {op: 频率}, 成功解析的公式数)``。
    """
    counts, unmapped = count_ops_in_corpus(formulas, ops=ops)
    n = counts.pop('_n_formulas', 0)
    total = sum(counts.values())
    mean = dict([(o, (counts[o] / n) if n else 0.) for o in counts])
    freq = dict([(o, (counts[o] / total) if total else 0.) for o in counts])
    return mean, freq, n


# ---------------------------------------------------------------------------
# 闭式估计：unigram（下界）
# ---------------------------------------------------------------------------

def fit_unigram_prior(formulas, ops=None, smoothing=0.5, scale=1.0):
    """闭式 unigram 估计：``a_o = -log p_o``。

    Parameters
    ----------
    formulas : iterable
        公式语料。
    ops : iterable or None
        算子表。
    smoothing : float
        add-k 平滑系数，避免语料里没出现的算子得到 ``-log 0 = inf``。
    scale : float
        整体缩放。

    Returns
    -------
    dict
        ``{'Nopi_<op>': a_o}``，不含二阶项。
    """
    if ops is None:
        from core.constants import OPS
        ops = list(OPS.keys())
    counts, _ = count_ops_in_corpus(formulas, ops=ops)
    counts.pop('_n_formulas', None)
    total = float(sum(counts.values()))
    k = float(len(ops))
    out = {}
    for op in ops:
        p = (counts.get(op, 0) + smoothing) / (total + smoothing * k)
        out['Nopi_%s' % op] = -math.log(p) * scale
    return out


# ---------------------------------------------------------------------------
# 矩匹配（[BMS2020] 式 14 的重要性采样近似）
# ---------------------------------------------------------------------------

def _sample_tree(depth, unary, binary, p_leaf, max_depth, rng):
    """从一个显式的树生成分布里采样一个 (算子计数, log q) 对。

    每个节点：以概率 ``p_leaf`` 变成叶子结束；否则在 OPS 里均匀抽一个算子，
    按其元数递归。因为生成过程完全显式，``log q`` 可以精确累计——重要性
    采样需要它。
    """
    nops = list(unary) + list(binary)
    if not nops:
        return Counter(), 0.0
    counts = Counter()
    logq = 0.0
    stack = [depth]
    while stack:
        d = stack.pop()
        if d >= max_depth:
            continue          # 深度触顶：确定性变叶子，q 贡献为 1
        if rng.random() < p_leaf:
            logq += math.log(p_leaf)
            continue
        logq += math.log(1.0 - p_leaf) + math.log(1.0 / len(nops))
        op = nops[int(rng.random() * len(nops))]
        counts[op] += 1
        if op in unary:
            stack.append(d + 1)
        else:
            stack.append(d + 1)
            stack.append(d + 1)
    return counts, logq


def fit_moment_matched_prior(formulas, ops=None, n_samples=20000,
                             n_iter=60, eta=0.5, p_leaf=0.45, max_depth=6,
                             seed=0, init=None, return_diagnostics=False):
    """矩匹配校准：迭代调整 ``a_o`` 使生成表达式的 ``<n_o>`` 与语料一致。

    每轮迭代：用固定的提议分布 ``q``（:func:`_sample_tree`）采 ``n_samples``
    棵树，按 ``w_i = exp(-sum_o a_o n_o,i) / q_i`` 计算重要性权重，得到模型
    侧的 ``<n_o>``，再按 ``a_o += eta * (<n_o>_model - <n_o>_corpus)`` 修正。

    Parameters
    ----------
    formulas : iterable
        公式语料。
    ops : iterable or None
        算子表。
    n_samples : int
        每轮采样的树数。
    n_iter : int
        迭代轮数。
    eta : float
        步长。
    p_leaf, max_depth : float, int
        提议分布参数：叶子概率与深度上限。控制生成树的典型规模，应调到来
        使生成规模与语料规模大致相当（诊断信息里会给出）。
    init : dict or None
        初值（``{'Nopi_<op>': a_o}``）。``None`` 时用 :func:`fit_unigram_prior`。

    Returns
    -------
    dict or (dict, dict)
        ``prior_par``；``return_diagnostics=True`` 时额外返回诊断字典
        （有效样本量 ESS、生成树的平均规模、逐轮的目标失配范数）。
    """
    if ops is None:
        from core.constants import OPS
        ops = list(OPS.keys())
    rng = np.random.RandomState(seed)

    target, _, n_formulas = corpus_moments(formulas, ops=ops)
    if init is None:
        a = dict([(o, fit_unigram_prior(formulas, ops=ops)['Nopi_%s' % o])
                  for o in ops])
    else:
        a = dict([(o, init.get('Nopi_%s' % o, 0.)) for o in ops])

    # 一元/二元算子分开（按 OPS 的元数）
    from core.constants import OPS as _OPS
    unary = [o for o in ops if _OPS.get(o, 1) == 1]
    binary = [o for o in ops if _OPS.get(o, 1) == 2]

    # 一次性采样，之后每轮只做矩阵-向量运算（N: n_samples x n_ops）
    counts, logq = [], []
    for _ in range(n_samples):
        c, lq = _sample_tree(0, unary, binary, p_leaf, max_depth, rng)
        counts.append([c[o] for o in ops])
        logq.append(lq)
    N = np.asarray(counts, dtype=float)
    logq = np.asarray(logq, dtype=float)
    sizes = N.sum(axis=1)
    tvec = np.asarray([target[o] for o in ops], dtype=float)

    diag = {'ess': [], 'mean_size': [], 'mismatch': []}
    avec = np.asarray([a[o] for o in ops], dtype=float)
    for _ in range(n_iter):
        logw = -(N @ avec) - logq
        logw -= logw.max()
        w = np.exp(logw)
        w /= w.sum()
        model = w @ N
        mismatch = float(np.linalg.norm(model - tvec))
        diag['ess'].append(float(1.0 / np.sum(w ** 2)))
        diag['mean_size'].append(float(w @ sizes))
        diag['mismatch'].append(mismatch)
        avec = avec + eta * (model - tvec)

    out = dict([('Nopi_%s' % o, float(avec[i])) for i, o in enumerate(ops)])

    out = dict([('Nopi_%s' % o, a[o]) for o in ops])
    if return_diagnostics:
        diag['n_formulas'] = n_formulas
        diag['target_mean'] = target
        return out, diag
    return out
