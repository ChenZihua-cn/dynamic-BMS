# archieve-9defb7aa — 上游计算库快照

- 快照来源：https://bitbucket.org/rguimera/machine-scientist
- 分支：`no_degeneracy`
- 上游 commit：`9defb7aa2a1382b4a8518f3060e8dd87a20acb99`
- 快照日期：2026-10-08

## 内容（上游主要计算代码）

| 文件 | 说明 |
|---|---|
| `mcmc.py` | BMS 核心：表达式树（Node/Tree）、能量计算、MCMC 采样，单文件 1545 行 |
| `parallel.py` | 并行回火（parallel tempering）驱动，多温度链并行采样 |
| `README.md` | 上游仓库文档（原样保留） |

## 未收录

`Prior/`（先验参数 .dat）、`Test/`、`Validation/`（数据集与分析脚本）、
`Process-Formulas/`、`Images/` 不属于主要计算代码，未纳入快照；
需要时可从上游对应 commit 获取。

## 与本地代码的关系

本仓库 `core/` 即是对本快照中 `mcmc.py` 的重构拆分（按职责分为
constants / node / tree_base / energy / proposal / mcmc 六个模块）。
此快照作为重构的原始参照基线保留。
