# prior/__init__.py
"""先验超参数包：登记、读取、落盘、从语料校准。

三步走完一件事——把"先验"从硬编码的散落数字变成项目里的一等公民::

    from prior import get_prior_par, read_prior_par, write_prior_par
    from core import Tree

    # 1) 取：按名拿一组已经登记好的先验（官方 wiki 校准值）
    prior_par = get_prior_par('bms_wiki_nv13_np13')

    # 2) 传：直接喂给 Tree —— 它进入 EP，再进入 MCMC 接受率
    t = Tree(variables=['L', 'g'], parameters=['a'], x=x, y=y,
             prior_par=prior_par)

    # 3a) 存：校准结果落盘，下次直接读
    write_prior_par('prior/data/my_corpus.json', prior_par)

    # 3b) 读：从文件读回，还是同一个字典
    prior_par = read_prior_par('prior/data/my_corpus.json')

关键 API
--------
``get_prior_par(name, ops=..., scale=...)``  按名取先验
``list_priors()``                            列出登记表
``flat_prior_par(ops, value)``               平值先验（0 = 纯 BIC）
``prior_energy(prior_par, nops)``            脱机算 EP，比较两组先验
``validate_prior_par(prior_par, ops)``       检查与算子表的匹配情况
``read_prior_par / write_prior_par``         落盘与读回
``count_ops_in_corpus / fit_unigram_prior``  从公式语料统计/闭式估计
``fit_moment_matched_prior``                 矩匹配校准（[BMS2020] 式 14）
"""

from .registry import (
    DEFAULT_NOPI,
    DEFAULT_PRIOR,
    BMS_WIKI_NV13_NP13,
    BMS_WIKI_NV13_NP13_LINEAR,
    FLAT_ZERO_VALUE,
    FLAT_DEFAULT_VALUE,
    PRIOR_REGISTRY,
    PriorSpec,
    flat_prior_par,
    list_priors,
    get_prior_par,
    scale_prior_par,
    ops_of,
    validate_prior_par,
    prior_energy,
    describe,
)
from .io import read_prior_par, write_prior_par, export_registry
from .fit import (
    count_ops_in_corpus,
    corpus_moments,
    fit_unigram_prior,
    fit_moment_matched_prior,
)
from .coverage import (
    LEAF_VAR, LEAF_PARAM, LEAF_NUM, LEAF_KINDS,
    config_space, n_configs, count_configs_in_corpus,
    coverage_report, format_report,
)

__all__ = [
    # 登记与取值
    'DEFAULT_NOPI', 'DEFAULT_PRIOR', 'PRIOR_REGISTRY', 'PriorSpec',
    'BMS_WIKI_NV13_NP13', 'BMS_WIKI_NV13_NP13_LINEAR',
    'FLAT_ZERO_VALUE', 'FLAT_DEFAULT_VALUE',
    'list_priors', 'get_prior_par', 'flat_prior_par', 'scale_prior_par',
    'ops_of', 'validate_prior_par', 'prior_energy', 'describe',
    # 落盘
    'read_prior_par', 'write_prior_par', 'export_registry',
    # 从语料校准
    'count_ops_in_corpus', 'corpus_moments',
    'fit_unigram_prior', 'fit_moment_matched_prior',
    # 构型覆盖率诊断
    'LEAF_VAR', 'LEAF_PARAM', 'LEAF_NUM', 'LEAF_KINDS',
    'config_space', 'n_configs', 'count_configs_in_corpus',
    'coverage_report', 'format_report',
]
