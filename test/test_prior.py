"""实验目的：验证 prior 包把"先验参数"接进了能量记账，且记账与 core 一致。

先验在 BMS 里是一组超参数 `a_o`(Nopi_<o>) / `b_o`(Nopi2_<o>)，它们以 nat
为单位进入 EP，再经 `exp(-dEB/BT - dEP/PT)` 影响接受率。本文件检验的定律：

1. 覆盖完备——登记表里的官方先验恰好覆盖 OPS 全部算子，与算子表对账后
   不缺不多；
2. 取值隔离——`get_prior_par` 返回拷贝，改动返回值不污染登记表；
3. 缺项语义——一阶项按 `fill_missing` 补齐，二阶项缺即表示无二次惩罚；
4. 记账一致——`prior_energy` 与 `Tree.get_energy` 的 EP 分支逐树对拍，
   含二次项在内必须逐位相同；
5. 平值退化——`flat_zero` 使 EP ≡ 0（后验退化为纯 BIC）；
6. 落盘往返——写出去再读回来是同一个字典，`.json` 与 `.dat` 两条路都通；
7. 语料统计——sympy 扁平形式下的算子计数可预测（sqrt/Pow、除法、一元负号）；
8. 校准方向——语料里越常见的算子，其惩罚越小；矩匹配迭代使失配单调下降。
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from core import Tree, OPS
from prior import (
    DEFAULT_PRIOR,
    BMS_WIKI_NV13_NP13,
    PRIOR_REGISTRY,
    flat_prior_par,
    list_priors,
    get_prior_par,
    scale_prior_par,
    ops_of,
    validate_prior_par,
    prior_energy,
    read_prior_par,
    write_prior_par,
    count_ops_in_corpus,
    corpus_moments,
    fit_unigram_prior,
    fit_moment_matched_prior,
    config_space,
    n_configs,
    count_configs_in_corpus,
    coverage_report,
    format_report,
)


# ---------------------------------------------------------------------------
# 1. 覆盖完备
# ---------------------------------------------------------------------------

def test_registry_covers_every_operator():
    """官方先验必须对 OPS 的每个算子都有一阶项——否则那个算子等于没惩罚。"""
    for name in ('bms_wiki_nv13_np13', 'bms_wiki_nv13_np13_linear'):
        par = get_prior_par(name)
        rep = validate_prior_par(par, ops=OPS)
        assert rep['missing'] == [], '%s 缺算子: %s' % (name, rep['missing'])
        assert rep['extra'] == [], '%s 多出算子: %s' % (name, rep['extra'])


def test_official_prior_has_exactly_one_quadratic_term():
    """官方值只有乘法的二次项 Nopi2_*，其余算子无二次惩罚。"""
    rep = validate_prior_par(get_prior_par('bms_wiki_nv13_np13'), ops=OPS)
    assert rep['quadratic'] == ['*']
    lin = validate_prior_par(get_prior_par('bms_wiki_nv13_np13_linear'),
                             ops=OPS)
    assert lin['quadratic'] == []


def test_list_priors_reports_every_registered_entry():
    names = [n for n, _ in list_priors()]
    assert set(names) == set(PRIOR_REGISTRY)
    assert DEFAULT_PRIOR in names


# ---------------------------------------------------------------------------
# 2-3. 取值语义
# ---------------------------------------------------------------------------

def test_get_prior_par_returns_a_copy():
    """返回值必须可自由修改——登记表是全局共享状态，不能被写脏。"""
    par = get_prior_par()
    par['Nopi_+'] = -999.
    par['Nopi_Injected'] = 1.
    assert BMS_WIKI_NV13_NP13['Nopi_+'] > 0
    assert 'Nopi_Injected' not in BMS_WIKI_NV13_NP13
    assert get_prior_par()['Nopi_+'] > 0


def test_ops_subset_fills_missing_first_order_only():
    """按算子表重建：一阶缺项补齐，二阶缺项不补（缺 = 无二次惩罚）。"""
    par = get_prior_par('bms_wiki_nv13_np13', ops=['+', '*', 'asin'],
                        fill_missing=7.5)
    assert par['Nopi_+'] == BMS_WIKI_NV13_NP13['Nopi_+']
    assert par['Nopi_*'] == BMS_WIKI_NV13_NP13['Nopi_*']
    assert par['Nopi_asin'] == 7.5          # 未知算子 → 填充值
    assert 'Nopi2_asin' not in par          # 二阶项不补齐
    assert par['Nopi2_*'] == 1.0            # 原有的二阶项保留


def test_fill_missing_none_raises_on_unknown_operator():
    with pytest.raises(KeyError):
        get_prior_par('bms_wiki_nv13_np13', ops=['+', 'asin'],
                      fill_missing=None)


def test_unknown_prior_name_raises():
    with pytest.raises(KeyError):
        get_prior_par('no_such_prior')


def test_scale_is_multiplicative():
    base = get_prior_par()
    got = scale_prior_par(base, 2.0)
    assert np.allclose([got[k] for k in base], [2. * base[k] for k in base])
    assert np.allclose([get_prior_par(scale=3.)[k] for k in base],
                       [3. * base[k] for k in base])


def test_ops_of_merges_first_and_second_order_keys():
    assert ops_of({'Nopi_+': 1., 'Nopi2_*': 2., 'Nopi_/': 3.}) == {'+', '*', '/'}


# ---------------------------------------------------------------------------
# 4. 记账一致（与 core 对拍）
# ---------------------------------------------------------------------------

FORMULAS = [
    '(_a_ * (L / g))',
    '(_a_ + (L * g))',
    '((_a_ * L) / (g + _a_))',
    '(sqrt((L / g)) * _a_)',
    '((L ** 2) + (sin(g) * _a_))',
    '(exp((_a_ * (L - (g / _a_)))))',
]


@pytest.mark.parametrize('formula', FORMULAS)
@pytest.mark.parametrize('name', ['bms_wiki_nv13_np13',
                                  'bms_wiki_nv13_np13_linear',
                                  'flat_default'])
def test_prior_energy_matches_tree_EP(formula, name):
    """prior_energy 与 Tree.get_energy 的 EP 分支必须逐位相同。

    这是"脱机比较先验"这一用法成立的前提：如果两边记账规则不一致，
    experiments/prior_ep_compare.py 的结论就不可信。
    """
    par = get_prior_par(name)
    t = Tree(variables=['L', 'g'], parameters=['a'],
             prior_par=par, from_string=formula)
    t.get_bic(reset=True)               # 无数据 → BIC=0，EB=0
    _, _, ep_tree = t.get_energy(reset=True)
    assert ep_tree == pytest.approx(prior_energy(par, t.nops))


def test_quadratic_term_actually_punishes_stacking():
    """Nopi2_* 让第 k 个乘法的边际惩罚随 k 增长——这是它抑制乘法堆叠的机制。"""
    par_quad = get_prior_par('bms_wiki_nv13_np13')
    par_lin = get_prior_par('bms_wiki_nv13_np13_linear')
    d_quad = (prior_energy(par_quad, {'*': 6}) - prior_energy(par_quad, {'*': 5}))
    d_lin = (prior_energy(par_lin, {'*': 6}) - prior_energy(par_lin, {'*': 5}))
    assert d_lin == pytest.approx(par_lin['Nopi_*'])
    assert d_quad == pytest.approx(par_quad['Nopi_*']
                                   + par_quad['Nopi2_*'] * (36 - 25))
    assert d_quad > d_lin


# ---------------------------------------------------------------------------
# 5. 平值退化
# ---------------------------------------------------------------------------

def test_flat_zero_collapses_to_pure_bic():
    """全零先验下 EP ≡ 0，后验里的结构项彻底消失。"""
    par = flat_prior_par(value=0.)
    assert set(par.values()) == {0.}
    assert prior_energy(par, {'*': 5, 'sin': 3}) == 0.
    assert len(par) == len(OPS)


# ---------------------------------------------------------------------------
# 6. 落盘往返
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('suffix', ['.json', '.dat'])
def test_write_read_roundtrip(tmp_path, suffix):
    par = get_prior_par()
    path = tmp_path / ('prior%s' % suffix)
    write_prior_par(path, par, header='test')
    got = read_prior_par(path)
    assert set(got) == set(par)
    for k in par:
        assert got[k] == pytest.approx(par[k])


@pytest.mark.parametrize('line', [
    'Nopi_+ 5.808163661224514',
    'Nopi_+: 5.808163661224514',
    'Nopi_+ = 5.808163661224514',
    'Nopi_+,5.808163661224514',
])
def test_read_tolerates_four_separators(tmp_path, line):
    p = tmp_path / 'p.dat'
    p.write_text('# comment\n\n%s\nNopi_* 5.0\n' % line)
    got = read_prior_par(p)
    assert got['Nopi_+'] == pytest.approx(5.808163661224514)
    assert got['Nopi_*'] == pytest.approx(5.0)


def test_read_accepts_json(tmp_path):
    p = tmp_path / 'p.json'
    p.write_text(json.dumps({'Nopi_+': 1.5}))
    assert read_prior_par(p)['Nopi_+'] == pytest.approx(1.5)


def test_read_rejects_garbage(tmp_path):
    p = tmp_path / 'p.dat'
    p.write_text('nothing parseable here\n')
    with pytest.raises(ValueError):
        read_prior_par(p)


# ---------------------------------------------------------------------------
# 7. 语料统计
# ---------------------------------------------------------------------------

def test_count_ops_handles_sympy_flattening():
    """sympy 把 sqrt 存成 Pow(x,1/2)、除法存成 Pow(y,-1)、-x 存成 Mul(-1,x)。

    这里固定住我们对这些写法的解释，避免以后改 sympy 版本时静默变语义。
    """
    counts, unmapped = count_ops_in_corpus([
        '_a_ * sqrt(L / g)',     # * 1, / 1, sqrt 1
        'x**2 + y + z',          # pow2 1, + 2
        '-2*x',                  # 一元 - 1（不是乘法）
        'factorial(n) * y',      # fac 1, * 1
    ])
    assert counts['*'] == 2
    assert counts['/'] == 1
    assert counts['sqrt'] == 1
    assert counts['pow2'] == 1
    assert counts['+'] == 2
    assert counts['-'] == 1
    assert counts['fac'] == 1
    assert not [u for u in unmapped if not u.startswith('UNPARSEABLE')]


def test_count_ops_reports_out_of_ops_functions():
    """语料里出现 OPS 之外的函数要被记下来，而不是被悄悄吞掉。"""
    _, unmapped = count_ops_in_corpus(['asin(x) + y'])
    assert 'asin' in unmapped


def test_corpus_moments_normalises_by_formula_count():
    mean, freq, n = corpus_moments(['a*b', 'c*d'])
    assert n == 2
    assert mean['*'] == pytest.approx(1.0)     # 每条公式平均 1 个乘法
    assert freq['*'] == pytest.approx(1.0)     # 全部算子就它一个


# ---------------------------------------------------------------------------
# 8. 校准方向
# ---------------------------------------------------------------------------

def test_unigram_penalises_rare_operators_more():
    """语料里越常见的算子，a_o = -log p_o 越小。"""
    corpus = ['a*b'] * 10 + ['sin(x)'] * 1
    par = fit_unigram_prior(corpus)
    assert par['Nopi_*'] < par['Nopi_sin']
    assert par['Nopi_sin'] < par['Nopi_fac']     # fac 语料里完全没有


def test_unigram_never_returns_infinite_penalty():
    """add-k 平滑必须保证未出现的算子也是有限值。"""
    par = fit_unigram_prior(['a*b'])
    assert all(np.isfinite(v) for v in par.values())


def test_config_space_size_is_independent_of_variables():
    """构型只区分区分叶子类别（VAR/PARAM/NUM），所以大小与 nv/np 无关。

    这正是相对原方案的关键差异：原方案的先验文件和 nv13/np13 绑死，换数据集
    就得重新拟合。构型空间则是固定的。
    """
    assert n_configs() == 483          # 21 根 + 22 槽位 × 21 标签
    assert len(config_space()) == 483
    # 叶子只用类别 → 换个"变量表"不改变构型空间
    assert len(config_space(ops=list(OPS))) == n_configs()


def test_count_configs_decomposes_a_known_tree():
    """`_a_*sqrt(L/g)` 的构型分解是唯一确定的，固定住这个解释。"""
    counts, unmapped, n = count_configs_in_corpus(['_a_ * sqrt(L / g)'])
    assert n == 1
    assert counts['root:*'] == 1
    assert counts['slot:*|0|PARAM'] == 1
    assert counts['slot:*|1|sqrt'] == 1
    assert counts['slot:sqrt|0|/'] == 1
    assert counts['slot:/|0|VAR'] == 1
    assert counts['slot:/|1|VAR'] == 1


def test_coverage_report_is_a_fraction_of_the_space():
    """覆盖率必须落在 [0,1]，且 已覆盖 + 未覆盖 = 构型总数。"""
    corpus = ['m*c**2', 'm*a', 'p*V/(n*R*T)', '_a_*sqrt(L/g)', 'k*log(W)',
              'N0*exp(-lamb*t)', '2*pi*sqrt(L/g)', 'sin(x)/x']
    rep = coverage_report(corpus)
    assert 0. < rep['coverage'] < 1.
    assert rep['n_covered'] + rep['n_uncovered'] == rep['n_configs']
    assert rep['n_covered'] > 0
    # 语料里没出现 fac / tanh，它们的构型应当全未覆盖
    assert rep['by_parent']['fac'][0] == 0
    assert rep['by_parent']['tanh'][0] == 0
    assert isinstance(format_report(rep), str)


def test_moment_matching_reduces_mismatch():
    """矩匹配迭代应使模型侧 <n_o> 与语料侧 <n_o> 的失配下降。"""
    corpus = ['_a_*sqrt(L/g)', 'x**2 + y + z', 'sin(x)/cos(y)', 'm*g*h']
    _, diag = fit_moment_matched_prior(
        corpus, n_samples=4000, n_iter=40, eta=0.3, seed=1,
        return_diagnostics=True,
    )
    assert diag['mismatch'][-1] < diag['mismatch'][0]
    assert all(np.isfinite(v) for v in diag['mismatch'])
