"""先验力度对照实验：同一对公式在不同先验下的 EP 差有多大。

背景（`log/pendulum_multiplanet_prior_audit_2026-10-16.md` §3.3 / §4.2-4）
------------------------------------------------------------------------
单摆实验里有两条链找到了真定律，但报出的 MAP 是"怪物外衣"——十几层常数
折叠把 `_a_` 包起来、数值上与 `c·sqrt(L/g)` 在机器精度内完全相同：

    abs(sqrt(abs(((((fac((sqrt(tanh(tanh(fac(cosh(sin(tanh(_a_))))))) ** 3))
      ** (cos((tanh(_a_) / (g / g))) ** 3)) * abs(L))
      + (L * sinh((L / (exp(tan(_a_)) * -(L))))))
      / (-(g) + ((g / _a_) * cos((cos(_a_) + _a_))))))))

因为 `canonical()` 只做字符串规范化、不化简常数折叠，怪物与简洁形式属于
不同的 canonical 家族，各自占一份后验质量。两者的数据项 BIC 相同（数值
等价），**决定谁能胜出的只有先验项 EP**。审计报告的判断是：平值先验
（`Nopi=1.0`）下两者 EP 差只有 ~17 nat，"不足以压制怪物"。

本实验把这个判断定量化：比较四组先验下 `ΔEP = EP(怪物) − EP(真公式)`。

用法
----
    .venv/bin/python experiments/prior_ep_compare.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core import Tree                                    # noqa: E402
from prior import (get_prior_par, flat_prior_par,         # noqa: E402
                   prior_energy, ops_of)

# 真公式与它的"怪物外衣"（数值等价，结构复杂度差一个数量级）
TRUE_FORMULA = '(_a_ * (L / g))'
MONSTER = (
    'abs(sqrt(abs(((((fac((sqrt(tanh(tanh(fac(cosh(sin(tanh(_a_))))))) '
    '** 3)) ** (cos((tanh(_a_) / (g / g))) ** 3)) * abs(L)) '
    '+ (L * sinh((L / (exp(tan(_a_)) * -(L)))))) '
    '/ (-(g) + ((g / _a_) * cos((cos(_a_) + _a_))))))))'
)

# 与 experiments/pendulum/known_g/pendulum.py 保持一致的建树配置
VARIABLES = ['L', 'g']
PARAMETERS = ['a']

# 参与对照的先验：名字 → prior_par
CANDIDATES = [
    ('flat_zero  (纯 BIC)', flat_prior_par(value=0.0)),
    ('flat 1.0   (pendulum 手设)', flat_prior_par(value=1.0)),
    ('flat 10.0  (core 内建默认)', flat_prior_par(value=10.0)),
    ('bms_wiki   (官方校准)',
     get_prior_par('bms_wiki_nv13_np13')),
    ('bms_wiki x2(官方校准加倍)',
     get_prior_par('bms_wiki_nv13_np13', scale=2.0)),
]


def nops_of(formula):
    """建树并取出算子计数。"""
    t = Tree(variables=VARIABLES, parameters=PARAMETERS,
             prior_par={}, from_string=formula)
    return dict([(k, v) for k, v in t.nops.items() if v]), t.size


def main():
    nops_true, size_true = nops_of(TRUE_FORMULA)
    nops_mon, size_mon = nops_of(MONSTER)

    print('真公式  %-60s  节点数=%d' % (TRUE_FORMULA, size_true))
    print('  算子计数: %s' % sorted(nops_true.items(), key=lambda kv: -kv[1]))
    print('怪物    %-60s  节点数=%d' % ('<monster>', size_mon))
    print('  算子计数: %s' % sorted(nops_mon.items(), key=lambda kv: -kv[1]))
    print()

    header = '%-28s %10s %10s %12s %10s' % (
        '先验', 'EP(真)', 'EP(怪物)', 'ΔEP(怪−真)', '放大倍数')
    print(header)
    print('-' * len(header))
    # 以 flat 1.0（pendulum 实验实设值）为基准，先算一次拿基准值
    baseline = None
    for name, par in CANDIDATES:
        if 'flat 1.0' in name:
            baseline = (prior_energy(par, nops_mon)
                        - prior_energy(par, nops_true))
    for name, par in CANDIDATES:
        ep_true = prior_energy(par, nops_true)
        ep_mon = prior_energy(par, nops_mon)
        dep = ep_mon - ep_true
        ratio = (dep / baseline) if baseline else float('nan')
        print('%-28s %10.2f %10.2f %12.2f %10.2f' % (
            name, ep_true, ep_mon, dep, ratio))
    print()
    print('说明：两条公式数值等价 ⇒ ΔBIC ≈ 0 ⇒ ΔEP 就是 MCMC 在两者之间')
    print('      做选择时看到的全部信号。ΔEP 越大，简洁形式越占优。')
    print('      "放大倍数" 以 flat 1.0（pendulum 实验实设值）为 1。')


if __name__ == '__main__':
    main()
