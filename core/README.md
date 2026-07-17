# core/ — 贝叶斯机器科学家 (BMS) 核心模块

本目录是对原始 `mcmc.py`（~1760行单文件）的重构，按职责拆分为 5 个模块，通过 Mixin 多重继承组装为最终的 `Tree` 类。

## 文件结构

```
core/
├── __init__.py     # 组装 Tree 类 (Mixin 多重继承)
├── constants.py    # 全局操作符字典 OPS
├── node.py         # 表达式树节点 Node
├── tree_base.py    # 树数据结构 & 初始化 (TreeBase)
├── energy.py       # 拟合 & 能量计算 (EnergyMixin)
├── proposal.py     # MCMC 建议移动 & 能量差计算 (ProposalMixin)
└── mcmc.py         # MCMC 采样循环 & 预测 (MCMCMixin)
```

## 类组装

[__init__.py](__init__.py) 通过 Mixin 多重继承组装最终类：

```python
class Tree(TreeBase, EnergyMixin, ProposalMixin, MCMCMixin):
    pass
```

MRO 按照广度优先(BFS)进行方法查找: `TreeBase → EnergyMixin → ProposalMixin → MCMCMixin`。每个 Mixin 通过 `self` 访问其他 Mixin 的方法和 `TreeBase` 的属性。

## 各模块在 MCMC 循环中的角色

### 1. `mcmc.py` — 顶层循环调度器

`mcmc()` 执行 Burn-in → 采样（带 thinning）两阶段循环。`mcmc_step()` 是单步核心，按概率选择三类移动：

```
mcmc_step()
  ├── 5%  概率 → Root 移动 (dE_rr)
  ├── 45% 概率 → Long-range 移动 (dE_lr)
  └── 50% 概率 → Elementary Tree 移动 (dE_et)
```

每次选中一种移动后：**计算能量差 → 计算接受概率 → 掷骰子 → 接受时更新树状态 + par_values + E/EB/EP**。

`predict()` 和 `trace_predict()` 利用采样链对给定数据进行预测，后者为每个采样点输出一次预测。

### 2. `proposal.py` — 建议移动 & 能量差计算

MCMC 的核心"引擎"。包含三类移动的**执行函数**和**dE 函数**：

| 执行函数 | dE 函数 | 用途 |
|---|---|---|
| `replace_root()` | `dE_rr(rr=...)` | 在旧根上加新根 |
| `prune_root()` | `dE_rr(rr=None)` | 剪掉根，恢复左子节点为新根 |
| `_add_et()` | — | 在叶子节点附加初等树（操作符 + 叶子） |
| `_del_et()` | — | 把初等树替换回叶子节点 |
| `et_replace()` | `dE_et()` | 一棵初等树替换为另一棵 |
| — | `dE_lr()` | 修改单个节点的操作符/变量值（保持arity） |

**dE 函数的通用模式：**
1. 保存当前状态 → **试执行移动** → 调用 `update_representative()` 检查 canonical 简并 → **撤销移动**
2. 若 canonical 已被其他表达式占用（返回 -1），则该公式被禁止，返回 `inf`
3. `dEP`：根据 `nops` 变化计算先验惩罚项的增量
4. `dEB`：实际执行移动 → `get_bic(fit=True)` 计算新的 BIC → 还原状态，`dEB = (BIC_new - BIC_old) / 2`
5. 返回 `(dE, dEB, dEP, par_valuesNew, ...)`

`dE_et` 额外返回 `nif`、`nfi`（移动前后可行移动类型数），用于修正接受概率中的提议比。

### 3. `energy.py` — 拟合与评估

在每次 dE 计算中被反复调用：

- **`get_sse(fit=True)`**：将表达式转为 SymPy 函数 → `scipy.curve_fit` 对自由参数做非线性最小二乘 → 计算 SSE。有缓存机制（`fit_par`）避免重复拟合相同表达式。
- **`get_bic()`**：BIC = 基于 SSE 的贝叶斯信息准则。
- **`get_energy()`**：`E = BIC/2 + Σ prior_par * nops`，即后验的负对数近似。`EB` 是数据项，`EP` 是先验项（对每个操作符的出现次数加权惩罚）。
- **`update_representative()`**：用 SymPy 计算表达式的 canonical 形式，检测公式简并（不同树结构对应同一数学表达式）。若 canonical 已被占用且当前树非代表，返回 `-1` 禁止该移动。

### 4. `tree_base.py` — 数据结构 & 初始化

- **`__init__()`**：初始化变量、参数、初等树空间（`et_space`）、根替换空间（`rr_space`）、操作符计数（`nops`）、先验参数等元数据。
- **`build_et_space()`**：枚举所有可能的一层子树（操作符 + 变量/参数叶子），按 order（子节点数）索引。
- **`build_rr_space()`**：枚举所有可能的根替换树（新增操作符 + 部分叶子）。
- **`canonical()`**：用 SymPy 标准归一表达式，参数重命名为 `c1, c2, ...`，供简并检测使用。
- **`build_from_string()`**：从字符串表达式重建整棵树。

### 5. `node.py` — 原子节点

- `Node.value`：操作符名 / 变量名 / 参数名
- `Node.offspring`：子节点列表
- `Node.order`：子节点数（= arity）
- `Node.pr()`：递归打印表达式，`pow2`/`pow3` 渲染为 `(** 2)` / `(** 3)`

### 6. `constants.py` — 全局操作符配置

```python
OPS = {
    'sin': 1, 'cos': 1, 'tan': 1,   # 一元函数
    'exp': 1, 'log': 1, 'sqrt': 1,
    '+': 2, '*': 2, '/': 2, '**': 2,  # 二元运算
    ...
}
```

键为操作符名，值为 arity（子节点数）。`OPS` 是模块级变量，可在运行时动态修改以扩展或限制操作符集（见 `test_regression.py` 第 10 项测试）。

## MCMC 单步完整调用链

以 **Elementary Tree (ET) 移动**为例，展示一次接受的全过程：

```
mcmc_step()                          [mcmc.py]
  │
  ├── 选择可行的 (oini, ofin)
  ├── target = choice(ets[oini])      # 随机选一个节点
  ├── new = choice(et_space[ofin])    # 随机选一个新ET
  │
  ├── dE_et(target, new)             [proposal.py]
  │   ├── et_replace(target, new, update_gof=False)  # 试执行
  │   │   ├── _del_et(target)        [proposal.py]   # 删除旧ET
  │   │   └── _add_et(target, et=new) [proposal.py]  # 添加新ET
  │   ├── update_representative()    [energy.py]     # canonical 简并检查
  │   ├── et_replace(added, old)     # 撤销移动
  │   ├── 计算 dEP（先验项变化）
  │   ├── et_replace → get_bic(fit=True) → 还原  # 计算数据项变化
  │   └── return (dE, dEB, dEP, par_valuesNew, nif, nfi)
  │
  ├── paccept = (nif * omegai * sf * exp(-dEB/BT - dEP/PT)) / (nfi * omegaf * si)
  │
  └── if random() < paccept:          # 接受
      ├── et_replace(target, new)     [proposal.py]
      ├── par_values = par_valuesNew
      ├── get_bic()                   [energy.py]
      └── E += dE; EB += dEB; EP += dEP
```

Root 替换/剪枝和 Long-range 移动遵循相同的"试-验-撤-决"模式，区别仅在于移动操作和接受概率公式。

## 与原始 mcmc.py 的关系

重构后的代码与原 `mcmc.py.bak` **逻辑完全一致**：
- 所有 28 个方法体和签名均原样保留
- 唯一差异是导入语句从 `from sympy import *` 改为按需导入
- 通过 `test_regression.py` 验证功能一致性

## 扩展点

根据 DP-BSR 研究计划，后续开发可在各模块中扩展：

- **`constants.py`**：添加新的物理操作符或约束
- **`energy.py`**：`get_energy()` 的 `EP` 计算处，将静态 `prior_par` 替换为动态门控网络输出的加权组合
- **`proposal.py`**：`dE_et`/`dE_lr`/`dE_rr` 中增加物理约束检查（量纲兼容性等）
- **`mcmc.py`**：`mcmc_step()` 中可注入新的移动类型
