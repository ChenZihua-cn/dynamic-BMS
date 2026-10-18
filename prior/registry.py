# prior/registry.py
"""先验超参数登记表。

本模块把散落在 `archieve-9defb7aa/mcmc.py` 里的先验超参数（原本硬编码在
`test3` / `test4` / `test5` 三个演示函数体内，随快照一起封存、无法被外部
调用）提取为具名常量，加上来源元数据后可被项目任何位置直接取用。

先验的数学形式（[BMS2020] 式 13）::

    -log p(f) = sum_o [ a_o * n_o(f) + b_o * n_o(f)^2 ]

其中 ``n_o(f)`` 是算子 ``o`` 在表达式树 f 中的出现次数。代码里 ``a_o`` 记作
``Nopi_<o>``，``b_o`` 记作 ``Nopi2_<o>``，两者都以 nat 为单位进入能量
``EP``，再经 ``paccept = exp(-dEB/BT - dEP/PT)`` 影响 MCMC 接受率。

``a_o`` / ``b_o`` 的来源：上游用 Wikipedia "List of scientific equations
named after people" 页面解析出的 4080 条公式作为语料，统计各算子出现次数的
一阶矩与二阶矩，再用最大熵（指数随机图）矩匹配迭代校准得到。详见
``log/pendulum_multiplanet_prior_audit_2026-10-16.md`` 的 §1.1。
"""

from copy import deepcopy

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 一阶惩罚的默认回退值。与 ``core/tree_base.py`` 中 ``prior_par == {}`` 时的
#: 回退值保持一致，保证"缺哪个算子就按代码默认处理"这一行为可预期。
DEFAULT_NOPI = 10.0

#: 上游 OPS 表的算子数（core/constants.py::OPS）。登记表里的官方先验恰好覆盖
#: 这 18 个算子，不多不少。
N_OPS_BMS = 18


# ---------------------------------------------------------------------------
# 从 archieve 快照中提取的先验参数
# ---------------------------------------------------------------------------

#: 官方拟合值（含乘法的二次项）。
#:
#: 来源：``archieve-9defb7aa/mcmc.py:1467``，``test3()`` 函数体内的
#: ``prior_par`` 字面量。对应上游
#: ``Prior/final_prior_param_sq.named_equations.nv13.np13.2016-09-01 17:05:57.196882.dat``
#: （nv=13 变量 / np=13 参数）的拟合结果，即 [BMS2020] 论文正文与补充材料中
#: 使用的那组超参数。
#:
#: 特征：18 个一阶项 + 1 个二阶项（只有 ``Nopi2_*``）。``fac`` 取默认值 10.0
#: （语料里几乎不出现阶乘，矩匹配把它推到默认的"极不常见"档）。
BMS_WIKI_NV13_NP13 = {
    'Nopi_/': 5.912205942815285,
    'Nopi_cosh': 8.12720511103694,
    'Nopi_-': 3.350846072163632,
    'Nopi_sin': 5.965917796154835,
    'Nopi_tan': 8.127427922862411,
    'Nopi_tanh': 7.799259068142255,
    'Nopi_**': 6.4734429542245495,
    'Nopi_pow2': 3.3017352779079734,
    'Nopi_pow3': 5.9907496760026175,
    'Nopi_exp': 4.768665265735502,
    'Nopi_log': 4.745957377206544,
    'Nopi_sqrt': 4.760686909134266,
    'Nopi_cos': 5.452564657261127,
    'Nopi_sinh': 7.955723540761046,
    'Nopi_abs': 6.333544134938385,
    'Nopi_+': 5.808163661224514,
    'Nopi_*': 5.002213595420244,
    'Nopi_fac': 10.0,
    'Nopi2_*': 1.0,
}

#: 官方拟合值的纯一阶版本（去掉 ``Nopi2_*``）。
#:
#: 来源：``archieve-9defb7aa/mcmc.py:1503``，``test4()`` 函数体内的
#: ``prior_par`` 字面量。与上一组数值完全相同，只是少了二次项，因此
#: ``-log p(f)`` 对算子计数是严格线性的。适合做消融对照：有无二次项对
#: "乘法堆叠"类过拟合的抑制力度差异。
BMS_WIKI_NV13_NP13_LINEAR = {
    'Nopi_/': 5.912205942815285,
    'Nopi_cosh': 8.12720511103694,
    'Nopi_-': 3.350846072163632,
    'Nopi_sin': 5.965917796154835,
    'Nopi_tan': 8.127427922862411,
    'Nopi_tanh': 7.799259068142255,
    'Nopi_**': 6.4734429542245495,
    'Nopi_pow2': 3.3017352779079734,
    'Nopi_pow3': 5.9907496760026175,
    'Nopi_exp': 4.768665265735502,
    'Nopi_log': 4.745957377206544,
    'Nopi_sqrt': 4.760686909134266,
    'Nopi_cos': 5.452564657261127,
    'Nopi_sinh': 7.955723540761046,
    'Nopi_abs': 6.333544134938385,
    'Nopi_+': 5.808163661224514,
    'Nopi_*': 5.002213595420244,
    'Nopi_fac': 10.0,
}

#: 全零先验（纯 BIC 模式）。
#:
#: 来源：``archieve-9defb7aa/mcmc.py:1524``，``test5()`` 函数体。所有算子
#: 惩罚置 0，``-log p(f) == 0``，后验退化为纯 BIC 模型选择。这是 [BMS2020]
#: 明确警告会结构性过拟合的设定，仅作消融基线与调试用。
FLAT_ZERO_VALUE = 0.0

#: 平值先验（代码内建默认）。
#:
#: 来源：``core/tree_base.py:67``，``prior_par == {}`` 时的回退值。所有算子
#: 一律罚 10 nat——"每种算子每出现一次罚 10"，是未校准的强奥卡姆偏置
#: （比官方拟合值更严厉，且对 sin 与 fac 一视同仁）。
FLAT_DEFAULT_VALUE = 10.0


def flat_prior_par(ops=None, value=FLAT_ZERO_VALUE):
    """构造平值先验：所有算子的一阶惩罚取同一个常数。

    Parameters
    ----------
    ops : iterable or None
        算子名列表。``None`` 时使用 ``core.constants.OPS`` 的全部算子。
    value : float
        每个算子每次出现的惩罚（nat）。``0.0`` 即纯 BIC，``10.0`` 即
        ``core/tree_base.py`` 的内建默认。

    Returns
    -------
    dict
        ``{'Nopi_<op>': value, ...}``，不含任何二阶项。
    """
    return dict([('Nopi_%s' % op, float(value)) for op in _resolve_ops(ops)])


# ---------------------------------------------------------------------------
# 登记表
# ---------------------------------------------------------------------------

class PriorSpec:
    """一组先验超参数及其来源元数据。

    Attributes
    ----------
    name : str
        登记名，供 :func:`get_prior_par` 索引。
    description : str
        一句话说明这组先验是什么、适合干什么。
    source : str
        出处（文件 + 行号，或上游数据文件名）。
    builder : callable or None
        无参调用返回参数字典。用于平值这类可按算子表生成的先验。
    params : dict or None
        固定参数字典（与 ``builder`` 二选一）。
    calibrated : bool
        是否由语料校准得到。``False`` 表示手工/平值先验。
    """

    def __init__(self, name, description, source, params=None, builder=None,
                 calibrated=False):
        if (params is None) == (builder is None):
            raise ValueError('PriorSpec requires exactly one of params/builder')
        self.name = name
        self.description = description
        self.source = source
        self.params = params
        self.builder = builder
        self.calibrated = calibrated

    def build(self):
        """返回参数字典的深拷贝。"""
        if self.builder is not None:
            return dict(self.builder())
        return deepcopy(self.params)

    def __repr__(self):
        return '<PriorSpec %s (calibrated=%s)>' % (self.name, self.calibrated)


PRIOR_REGISTRY = {
    'bms_wiki_nv13_np13': PriorSpec(
        name='bms_wiki_nv13_np13',
        description=(
            '官方 wiki 语料校准值（一阶+乘法二次项）。[BMS2020] 论文所用'
            '先验，应作为生产默认。'
        ),
        source='archieve-9defb7aa/mcmc.py:1467 (test3)；上游 '
               'Prior/final_prior_param_sq.named_equations.nv13.np13'
               '.2016-09-01 17:05:57.196882.dat',
        params=BMS_WIKI_NV13_NP13,
        calibrated=True,
    ),
    'bms_wiki_nv13_np13_linear': PriorSpec(
        name='bms_wiki_nv13_np13_linear',
        description=(
            '官方 wiki 校准值的纯一阶版本（无二次项）。用于消融：检验'
            'Nopi2_* 对乘法堆叠型过拟合的抑制贡献。'
        ),
        source='archieve-9defb7aa/mcmc.py:1503 (test4)',
        params=BMS_WIKI_NV13_NP13_LINEAR,
        calibrated=True,
    ),
    'flat_zero': PriorSpec(
        name='flat_zero',
        description='全零先验：-log p(f) == 0，后验退化为纯 BIC。仅作基线/调试。',
        source='archieve-9defb7aa/mcmc.py:1524 (test5)',
        builder=lambda: flat_prior_par(value=FLAT_ZERO_VALUE),
        calibrated=False,
    ),
    'flat_default': PriorSpec(
        name='flat_default',
        description=(
            '平值先验：每算子每次出现罚 10 nat。core/tree_base.py 的内建'
            '回退值，未校准，仅作对照。'
        ),
        source='core/tree_base.py:67 (prior_par == {} 回退分支)',
        builder=lambda: flat_prior_par(value=FLAT_DEFAULT_VALUE),
        calibrated=False,
    ),
}

#: 默认先验名。
DEFAULT_PRIOR = 'bms_wiki_nv13_np13'


# ---------------------------------------------------------------------------
# 取值
# ---------------------------------------------------------------------------

def _resolve_ops(ops=None):
    """把 ``ops`` 解析为算子名列表；``None`` 时取 ``core.constants.OPS``。"""
    if ops is None:
        from core.constants import OPS
        return list(OPS.keys())
    return list(ops)


def list_priors():
    """列出登记表中所有先验的 ``(name, description)``。"""
    return [(n, s.description) for n, s in PRIOR_REGISTRY.items()]


def get_prior_par(name=DEFAULT_PRIOR, ops=None, scale=1.0,
                  fill_missing=DEFAULT_NOPI, keep_extra=True):
    """按名取先验超参数字典，可直接喂给 ``Tree(prior_par=...)``。

    Parameters
    ----------
    name : str
        登记名，见 :func:`list_priors`。
    ops : iterable or None
        目标算子表。``None``（默认）时原样返回登记的参数字典；给定则按该
        算子表重建：保留已有一阶/二阶项，缺失的一阶项用 ``fill_missing``
        补齐，**二阶项不补齐**（缺即表示"无二次惩罚"，与上游语义一致）。
    scale : float
        全体参数的乘性缩放。等价于对先验整体调温（放大 = 更强的结构惩罚）。
    fill_missing : float or None
        缺失一阶项的填充值。``None`` 表示遇到缺失算子直接抛 ``KeyError``。
    keep_extra : bool
        是否保留不与任何算子对应的键（例如将来扩展的结构先验键
        ``Nopi_root_+``、``Nopi_pair_*|/``）。

    Returns
    -------
    dict
        参数字典的拷贝；调用方可以自由修改而不污染登记表。

    Raises
    ------
    KeyError
        登记名不存在，或 ``fill_missing is None`` 且算子表含未知算子。
    """
    try:
        spec = PRIOR_REGISTRY[name]
    except KeyError:
        raise KeyError(
            'Unknown prior %r. Available: %s'
            % (name, ', '.join(sorted(PRIOR_REGISTRY)))
        )
    base = spec.build()

    if ops is not None:
        resolved = _resolve_ops(ops)
        out = {}
        for op in resolved:
            key = 'Nopi_%s' % op
            if key in base:
                out[key] = base[key]
            elif fill_missing is None:
                raise KeyError(
                    'Prior %r has no entry for operator %r and '
                    'fill_missing is None.' % (name, op)
                )
            else:
                out[key] = float(fill_missing)
            qkey = 'Nopi2_%s' % op
            if qkey in base:
                out[qkey] = base[qkey]
        if keep_extra:
            known = set(['Nopi_%s' % o for o in resolved]) | \
                    set(['Nopi2_%s' % o for o in resolved])
            for k, v in base.items():
                if k not in known:
                    out[k] = v
        base = out

    if scale != 1.0:
        base = dict([(k, v * scale) for k, v in base.items()])
    return base


def scale_prior_par(prior_par, factor):
    """整体缩放先验参数（等价于给先验调温）。返回新字典。"""
    return dict([(k, v * factor) for k, v in prior_par.items()])


def ops_of(prior_par):
    """返回参数字典中出现过的算子名集合（一阶与二阶合并）。"""
    ops = set()
    for k in prior_par:
        for prefix in ('Nopi_', 'Nopi2_'):
            if k.startswith(prefix):
                ops.add(k[len(prefix):])
    return ops


def validate_prior_par(prior_par, ops=None):
    """检查先验参数与算子表的匹配情况。

    Returns
    -------
    dict
        ``{'missing': 缺一阶项的算子, 'extra': 多出的算子,
        'quadratic': 有二阶项的算子, 'unknown': 无法识别的键}``。
    """
    resolved = set(_resolve_ops(ops))
    present = ops_of(prior_par)
    known_prefixed = set()
    for op in resolved:
        known_prefixed.add('Nopi_%s' % op)
        known_prefixed.add('Nopi2_%s' % op)
    return {
        'missing': sorted(resolved - present),
        'extra': sorted(present - resolved),
        'quadratic': sorted([k[len('Nopi2_'):] for k in prior_par
                             if k.startswith('Nopi2_')]),
        'unknown': sorted([k for k in prior_par
                           if not (k.startswith('Nopi_') or
                                   k.startswith('Nopi2_'))]),
    }


def prior_energy(prior_par, nops):
    """按 ``-log p(f) = sum_o [a_o n_o + b_o n_o^2]`` 计算先验能量 EP。

    与 ``core/energy.py::get_energy`` 中的 EP 分支、``core/proposal.py``
    中三个 ``dEP`` 分支完全同一套记账规则（缺失键静默跳过），因此可以脱离
    Tree 单独使用：给定一棵树的算子计数，直接比较两组先验下的 EP，或比较
    两棵树在同一先验下的 EP 差。

    Parameters
    ----------
    prior_par : dict
        先验超参数字典。
    nops : dict
        ``{算子名: 出现次数}``，对应 ``Tree.nops``。

    Returns
    -------
    float
        先验能量 EP（nat），即 ``-log p(f)`` 的模型结构部分。
    """
    ep = 0.0
    for op, n in nops.items():
        try:
            ep += prior_par['Nopi_%s' % op] * n
        except KeyError:
            pass
        try:
            ep += prior_par['Nopi2_%s' % op] * n ** 2
        except KeyError:
            pass
    return ep


def describe(name=DEFAULT_PRIOR):
    """返回一组先验的可读摘要（来源、算子覆盖、数值表）。"""
    spec = PRIOR_REGISTRY[name]
    par = spec.build()
    lines = [
        'prior: %s' % name,
        '  source    : %s' % spec.source,
        '  calibrated: %s' % spec.calibrated,
        '  operators : %d first-order, %d second-order'
        % (len([k for k in par if k.startswith('Nopi_')]),
           len([k for k in par if k.startswith('Nopi2_')])),
        '  values (nat):',
    ]
    for k in sorted(par, key=lambda k: -par[k]):
        lines.append('    %-12s %12.6f' % (k, par[k]))
    return '\n'.join(lines)
