# prior/ —— 先验超参数包

把"先验"从散落在各处的字面量，变成项目里可登记、可读取、可落盘、可校准的一等公民。

## 为什么要有这个目录

`archieve-9defb7aa/mcmc.py` 里其实**已经**藏着 [BMS2020] 那套由 Wikipedia
4080 条公式校准出来的先验参数——但它们硬编码在 `test3()` / `test4()` /
`test5()` 三个演示函数的函数体里（第 1467 / 1503 / 1524 行），
`Tree(prior_par=...)` 根本拿不到。于是本项目一直在用平值 `Nopi=1.0`
（`experiments/pendulum/pendulum.py`）或内建默认 `10.0`（`core/tree_base.py`）跑实验，
即"先验通道是通的，但流经它的数字不是校准出来的"。

这个目录就是那条断掉的接线。

## 目录结构

```
prior/
├── __init__.py      公共 API（唯一需要 import 的入口）
├── registry.py      先验登记表：常量 + 来源元数据 + get/list/validate/energy
├── io.py            落盘与读回（.json / .dat），宽容解析五种写法
├── fit.py           从公式语料计算先验：计数统计 / 闭式 unigram / 矩匹配
├── coverage.py      构型覆盖率诊断：语料覆盖了多少种可表达的树类型
├── data/            导出产物（登记表里每组先验一份 .json，供外部工具读取）
└── README.md        本文件
```

## 先验的数学形式

```
-log p(f) = Σ_o [ a_o · n_o(f)  +  b_o · n_o(f)² ]
```

`n_o(f)` = 算子 `o` 在表达式树里的出现次数。`a_o` 记作 `Nopi_<o>`，
`b_o` 记作 `Nopi2_<o>`，单位 nat，进入能量 `EP`，再经

```
paccept = exp(−dEB/BT − dEP/PT)
```

影响 MCMC 接受率（见 `core/energy.py::get_energy`、`core/proposal.py` 的三个
`dE_*`）。所有读取点都有 `try/KeyError` 保护，缺哪个键就当那一项不存在。

## 三条使用路径

### 1. 取一组已登记的先验（最常用）

```python
from prior import get_prior_par
from core import Tree

prior_par = get_prior_par('bms_wiki_nv13_np13')   # 官方 wiki 校准值
t = Tree(variables=['L', 'g'], parameters=['a'], x=x, y=y,
         prior_par=prior_par)
```

`get_prior_par(ops=OPS)` 会按当前算子表重建字典：已有键保留、缺失的一阶项
用 `fill_missing`（默认 10.0）补齐、二阶项**不补齐**（缺即"无二次惩罚"，
与上游语义一致）。`scale=2.0` 可整体加倍——等价于给先验调温。

### 2. 从公式语料算一组先验并保存

```python
from prior import fit_unigram_prior, fit_moment_matched_prior, write_prior_par

corpus = ['_a_*sqrt(L/g)', 'm*g*h', 'p*V/(n*R*T)', ...]

par_a = fit_unigram_prior(corpus)                    # 闭式，惩罚力度下界
par_b, diag = fit_moment_matched_prior(corpus, return_diagnostics=True)  # 矩匹配

write_prior_par('prior/data/my_corpus.json', par_b)
```

之后：

```python
from prior import read_prior_par
prior_par = read_prior_par('prior/data/my_corpus.json')
```

### 3. 脱机比较先验，不建树

```python
from prior import get_prior_par, prior_energy

p = get_prior_par()
prior_energy(p, {'*': 1, '/': 1, 'sqrt': 1})   # → 16.675
```

`prior_energy` 与 `core` 里的 EP 记账规则逐字一致（已由
`test/test_prior.py` 用真实 Tree 对拍验证），因此可以脱离 MCMC 单独
比较"两组先验对同一棵树的评价"或"同一先验下两棵树的 EP 差"。
`experiments/prior_ep_compare.py` 就靠它量化了"怪物外衣 vs 真公式"。

## 登记表

| 名字 | 内容 | 来源 | 已校准 |
|---|---|---|---|
| `bms_wiki_nv13_np13` | 官方 wiki 校准值，含 `Nopi2_*` | `archieve-9defb7aa/mcmc.py:1467` | ✅ |
| `bms_wiki_nv13_np13_linear` | 同上，去掉二次项 | `archieve-9defb7aa/mcmc.py:1503` | ✅ |
| `flat_zero` | 全零，退化为纯 BIC | `archieve-9defb7aa/mcmc.py:1524` | ❌ |
| `flat_default` | 全 10 nat，`core` 内建默认 | `core/tree_base.py:67` | ❌ |

前两组覆盖 `core/constants.py::OPS` 的全部 18 个算子，不多不少。

## 已知近似与坑

- **语料计数按 sympy 的扁平 n 元形式**：`sqrt(x)` 被 sympy 存成
  `Pow(x, 1/2)`（已识别为 `sqrt`），`x/y` 存成 `Mul(x, Pow(y, -1))`（识别为 `/`）。
  但 `p*V/(n*R*T)` 这类会被 sympy 摊平成 `Mul(p, V, Pow(n,-1), Pow(R,-1),
  Pow(T,-1))`，于是计成 3 个 `/` + 1 个 `*` 而非 BMS 二叉树下的 1 个 `/` +
  3 个 `*`。**内部节点总数守恒**，只是 `*` 与 `/` 的分配不同。
- **`fit_unigram_prior` 是惩罚下界**：它只算"每个算子有多常见"，不算"有多少棵
  不同的树恰好用到这么多次该算子"。真实 `a_o` 必须更大以抵消树形的组合爆炸，
  所以它的输出普遍小于官方值。要量与官方值可比，用
  `fit_moment_matched_prior`（在树的生成分布上做期望，多重性自动计入）。
- **矩匹配的 ESS 偏低**：用固定的均匀提议做重要性采样，`a` 漂移后权重会集中。
  诊断字典里的 `ess` 就是有效样本量，跑完务必看一眼；ESS < 50 时结果只能当
  量级参考。
- **sympy 保留字会撞上物理变量名**：`E`（自然常数）、`I`（虚数）、`N`、`S`、
  `pi` 在 sympify 里不是普通符号。语料里用 `E` 表示能量会被解析成
  `exp(1)`，`coverage_report` 会把它报进 `out_of_ops`（显示为 `Exp1`）。
  写语料时用 `En`、`E_` 之类规避。
- **`fac` 在官方值里是 10.0**：并非校准收敛的结果，而是语料里几乎不出现阶乘、
  参数被推到"极不常见"的默认档。读这个数字时不要误当成高置信估计。
