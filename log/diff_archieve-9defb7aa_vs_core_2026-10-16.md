# 逐函数对比报告：`archieve-9defb7aa/` vs `core/`

- **对比基线（旧）**：`dynamic-BMS/archieve-9defb7aa/` —— 上游 machine-scientist 快照（commit `9defb7aa`，分支 `no_degeneracy`），含单文件 `mcmc.py`（约 1545 行）与 `parallel.py`。
- **对比目标（新）**：`dynamic-BMS/core/` —— 本地重构拆分版，含 `constants.py`、`node.py`、`tree_base.py`、`energy.py`、`proposal.py`、`mcmc.py`、`__init__.py`。
- **对比方法**：基于 Python AST 提取两侧全部类/函数源码，按函数名配对后做逐行 diff。
- **日期**：2026-10-16

---

## 1. 总体结论

本次重构的**主旋律是"拆分 + 接线"，而非改写算法**：

1. **模块拆分**：旧 `Tree` 巨型类按职责拆为 `TreeBase`（结构/解析）、`EnergyMixin`（拟合与能量）、`ProposalMixin`（MCMC 移动提议与能量差）、`MCMCMixin`（采样循环与预测）四个 mixin，再经 `core/__init__.py` 中的 `class Tree(TreeBase, EnergyMixin, ProposalMixin, MCMCMixin, ExpertMixin)` 多继承组装。`Node` 独立为 `node.py`，全局算子表 `OPS` 独立为 `constants.py`。
2. **函数体大规模保持不变**：37 个配对函数中 **19 个逐字节相同**（仅迁移文件），**11 个有实质或格式修改**，**9 个被删除**，core 内**无全新定义的函数**（新增能力全部来自外部 `experts` / `gate` 包的 mixin，通过钩子接入）。
3. **唯一的实质行为变化**：接入了"专家系统"（constitution 维度约束检查 + parliament 结构惩罚），改动集中在 `__init__`、`build_from_string`、`dE_et`、`dE_lr`、`dE_rr`、`get_energy` 六个函数，其余修改均为排版/注释/微小修复。
4. **被删除的部分**：整个并行回火驱动 `parallel.py`（`Parallel` 类 5 个方法）、`Tree.latex_val`、模块级测试函数 `test3/test4/test5` 均未迁入 `core/`。

### 统计汇总

| 类别 | 数量 | 函数 |
|---|---|---|
| 未变（仅迁移） | 19 | `Node.__init__`, `Node.pr`, `Tree.__repr__`, `Tree.pr`, `__parse_recursive`, `__grow_tree`, `build_et_space`, `build_rr_space`, `canonical`, `latex`, `replace_root`, `is_root_prunable`, `prune_root`, `_add_et`, `_del_et`, `et_replace`, `get_bic`, `mcmc` |
| 修改 | 11 | `Tree.__init__`→`TreeBase.__init__`, `build_from_string`, `set_par_values`, `get_sse`, `get_energy`, `update_representative`, `dE_et`, `dE_lr`, `dE_rr`, `mcmc_step`, `predict`, `trace_predict`（其中 5 个仅格式/注释级改动） |
| 删除 | 9 | `Tree.latex_val`, `test3`, `test4`, `test5`, `Parallel.__init__`, `Parallel.mcmc_step`, `Parallel.tree_swap`, `Parallel.anneal`, `Parallel.trace_predict` |
| 新增（core 内） | 0 | ——（新能力由 `experts.base.ExpertMixin` / `gate.base.GateMixin` 提供，不在 core 目录内） |

### 模块对应关系

| 旧位置 | 新位置 |
|---|---|
| `mcmc.py` 全局 `OPS` | `core/constants.py`（内容不变） |
| `mcmc.py: class Node` | `core/node.py: class Node`（不变） |
| `mcmc.py: class Tree`（结构部分） | `core/tree_base.py: class TreeBase` |
| `mcmc.py: class Tree`（能量部分） | `core/energy.py: class EnergyMixin` |
| `mcmc.py: class Tree`（提议部分） | `core/proposal.py: class ProposalMixin` |
| `mcmc.py: class Tree`（采样部分） | `core/mcmc.py: class MCMCMixin` |
| ——（新增组装层） | `core/__init__.py: class Tree(...)` 多继承组装，MRO：`TreeBase → EnergyMixin → ProposalMixin → MCMCMixin → ExpertMixin` |
| `parallel.py: class Parallel` | **未迁移**（已删除） |

---

## 2. 逐函数对比

### 2.1 `core/node.py` —— class Node

#### `Node.__init__(self, value, parent=None, offspring=[])` —— ✅ 未变（迁移）
- **旧位置**：`archieve-9defb7aa/mcmc.py:48`（class Node）
- **核心逻辑**：表达式树节点构造。保存算子/变量/参数名 `value`、父节点引用、子节点列表，并初始化 `x`/`y`（数据引用占位）。
- **变化**：无，逐字节相同。

#### `Node.pr(self, show_pow=False)` —— ✅ 未变（迁移）
- **旧位置**：`mcmc.py:55`
- **核心逻辑**：将节点（子树）递归打印为中缀表达式字符串；`show_pow` 控制 `pow2/pow3` 是否显示为 `**2/**3` 形式。
- **变化**：无。

---

### 2.2 `core/tree_base.py` —— class TreeBase（表达式树结构与解析）

#### `TreeBase.__init__(self, ops=OPS, variables=['x'], parameters=['a'], prior_par=None, x=None, y=None, from_string=None, fixed_term=None, fixed_term_op='*', extra_variables=[], dimensions=None, constitutions=None, parliaments=None, ...)` —— 🔧 **修改（实质）**
- **旧签名**：`Tree.__init__(self, ops=OPS, variables=['x'], parameters=['a'], prior_par=None, x=None, y=None, from_string=None, fixed_term=None, fixed_term_op='*', extra_variables=[], max_size=..., BT=..., PT=...)`
- **核心逻辑**：树的完整初始化——保存算子表/变量/参数/先验参数、数据集 `(x, y)`、温度参数 `BT/PT`、复杂度上限 `max_size`、固定项 `fixed_term`；创建根节点；若给了 `from_string` 则调 `build_from_string`，否则随机/默认建树；初始化 SSE/BIC/能量与 `fit_par`、`nops` 等簿记。
- **具体变化**：
  1. **新增 3 个参数**：`dimensions=None`（量纲声明）、`constitutions=None`（维度约束"宪法"）、`parliaments=None`（结构惩罚"议会"配置）。
  2. **新增初始化逻辑**：`self.dimensions = dimensions or {}`；`self.parliaments = []`（默认空，注释注明由 `GateMixin._init_gate` 覆盖）；若存在 `_init_experts` 方法（即 `ExpertMixin` 已混入），调用 `self._init_experts(constitutions, parliaments)` 完成专家系统接线。
  3. 注释拼写修正（`poosible` → `possible`）。
- **行为差异**：旧版构造的树不带任何维度/专家系统状态；新版在混入 `ExpertMixin` 时自动初始化 constitution/parliament 层，未混入时（`hasattr` 守卫）退化为旧行为，向后兼容。

#### `TreeBase.__repr__(self)` —— ✅ 未变（迁移）
- **核心逻辑**：返回树的规范字符串表示（即 `Tree.pr()` 的全树版本），作为 `build_from_string` 的输入格式。
- **变化**：无。

#### `TreeBase.pr(self, show_pow=True)` —— ✅ 未变（迁移）
- **核心逻辑**：从根节点递归生成表达式字符串，默认 `show_pow=True`。
- **变化**：无。

#### `TreeBase.set_par_values(self, par_values)` —— 🔧 修改（仅注释/格式）
- **核心逻辑**：把外部给定的参数值字典写入树中各参数节点；单数据集情形下用数据标签对齐；最后把当前参数快照存入 `self.fit_par[str(self)]`。
- **变化**：注释错别字修正（`par_lavues` → `par_values`）与空白行清理；**逻辑完全不变**。

#### `TreeBase.canonical(self, verbose=False)` —— ✅ 未变（迁移）
- **核心逻辑**：将树整理为规范形（对 `+`/`*` 等可交换算子的子节点排序），返回规范字符串，用于公式去重（representative 机制）。
- **变化**：无。

#### `TreeBase.latex(self)` —— ✅ 未变（迁移）
- **核心逻辑**：借助 SymPy 把表达式转为 LaTeX 字符串。
- **变化**：无。

#### ~~`Tree.latex_val(self, num_digits=1)`~~ —— ❌ **已删除**
- **旧位置**：`mcmc.py:275`
- **原核心逻辑**：生成带参数数值（按 `num_digits` 保留有效数字）代入后的 LaTeX 表达式。
- **删除影响**：`core/` 中无替代实现；如需数值版 LaTeX 输出须另行实现或从快照拷贝。

#### `TreeBase.__parse_recursive(self, string, variables=None, parameters=None, vpreturn=False)` —— ✅ 未变（迁移）
- **核心逻辑**：递归下降解析 `__repr__` 格式的表达式字符串，返回嵌套列表结构及其中出现的变量/参数集合。
- **变化**：无（名称 mangling 随类名变为 `_TreeBase__parse_recursive`，调用方同模块，无影响）。

#### `TreeBase.__grow_tree(self, target, value, offspring)` —— ✅ 未变（迁移）
- **核心逻辑**：按解析出的嵌套列表递归地把子树挂到 `target` 节点下。
- **变化**：无。

#### `TreeBase.build_from_string(self, string, verbose=False)` —— 🔧 **修改（实质）**
- **核心逻辑**：从字符串重建整棵树——解析后调用 `self.__init__` 重置自身，再 `__grow_tree` 生长结构，最后重算 SSE/BIC 并清空 `fit_par`。
- **具体变化**：
  1. **重建时保留更多配置**：`self.__init__(...)` 调用新增透传 `max_size`、`fixed_term`、`fixed_term_op`、`extra_variables`、`dimensions`、`constitutions`（有则传）、`parliaments`（有则传）。旧版只透传 `ops/prior_par/x/y/BT/PT/parameters/variables`，**重建会丢失 `max_size`、`fixed_term` 等自定义配置**（旧版隐性 bug，重建后退回默认值）。
  2. **新增宪法校验**：树生长完成后，若存在 `check_constitution`（ExpertMixin 提供）且校验不通过，抛出 `ValueError: Expression '<string>' violates dimensional constraints.`，即禁止加载量纲不合法的表达式。
- **行为差异**：旧版可加载任意语法合法的表达式；新版在启用专家系统时会拒绝维度不合法的表达式；且重建后配置保持完整。

#### `TreeBase.build_et_space(self)` —— ✅ 未变（迁移）
- **核心逻辑**：枚举所有"初等树变换"（elementary tree move）目标位置，构建 ET 移动空间 `self.ets`。
- **变化**：无。

#### `TreeBase.build_rr_space(self)` —— ✅ 未变（迁移）
- **核心逻辑**：枚举根节点可替换的（算子, 子节点组合）空间 `self.rrs`，供 root-replacement 移动使用。
- **变化**：无。

---

### 2.3 `core/energy.py` —— class EnergyMixin（拟合优度与能量）

#### `EnergyMixin.get_sse(self, fit=True, verbose=False)` —— 🔧 修改（仅格式）
- **核心逻辑**：计算表达式的平方误差和（SSE）。`fit=True` 时先用 scipy 曲线拟合优化参数值；逐数据集累计误差，更新 `self.sse` 与 `self.par_values`。
- **变化**：仅尾部空白/注释格式清理；**逻辑完全不变**。

#### `EnergyMixin.get_bic(self, reset=True, fit=False, verbose=False)` —— ✅ 未变（迁移）
- **核心逻辑**：由 SSE 计算 BIC（贝叶斯信息准则）作为描述长度近似，更新 `self.bic`。
- **变化**：无。

#### `EnergyMixin.get_energy(self, bic=False, reset=False, verbose=False)` —— 🔧 **修改（实质）**
- **核心逻辑**：能量 = 数据的 BIC 贡献 `EB`（`bic/2`）+ 先验贡献 `EP`（各算子出现次数的一次/二次先验惩罚），返回三元组 `(EB+EP, EB, EP)`；`reset=True` 时刷新 `self.E/EB/EP`。
- **具体变化**：在计算 `EP` 后新增：
  ```python
  # Parliament structural penalty (Phase 1: structural only)
  if self.parliaments:
      EP += self._parliament_energy_structural()
  ```
  即当存在 parliament 配置时，把 `GateMixin._parliament_energy_structural()` 的结构惩罚并入先验能量项。
- **行为差异**：返回值结构不变（仍为三元组），但启用 parliament 时 `EP` 与总能量变大，MCMC 接受率随之偏向结构简单的公式；`self.parliaments` 为空列表时行为与旧版一致。

#### `EnergyMixin.update_representative(self, verbose=False)` —— 🔧 修改（仅 docstring 格式）
- **核心逻辑**：公式去重簿记。计算规范形字符串，若该公式（或其等价形）已见过则返回 `-1`（禁止），否则登记并返回 `1`；维护 `self.representative` 集合。
- **变化**：docstring 换行重排；**逻辑完全不变**。

---

### 2.4 `core/proposal.py` —— class ProposalMixin（MCMC 移动与能量差）

#### `ProposalMixin.replace_root(self, rr=None, update_gof=True, verbose=False)` —— ✅ 未变（迁移）
- **核心逻辑**：执行 root-replacement 移动——把根节点替换为新算子并按 `rr` 指定方式重挂子树；受 `max_size` 约束，超限时返回 `None`；可选即时更新拟合优度。
- **变化**：无。

#### `ProposalMixin.is_root_prunable(self)` —— ✅ 未变（迁移）
- **核心逻辑**：判断当前根节点是否可剪除（根为一元/二元算子且剪除后不违反约束）。
- **变化**：无。

#### `ProposalMixin.prune_root(self, update_gof=True, verbose=False)` —— ✅ 未变（迁移）
- **核心逻辑**：执行 root-pruning 移动——移除根算子，提升其子节点为新根，更新结构与簿记。
- **变化**：无。

#### `ProposalMixin._add_et(self, node, et_order=None, et=None, update_gof=True, verbose=False)` —— ✅ 未变（迁移）
- **核心逻辑**：在指定节点处添加一个初等子树（ET 移动的"加"方向），返回被添加的结构信息。
- **变化**：无。

#### `ProposalMixin._del_et(self, node, leaf=None, update_gof=True, verbose=False)` —— ✅ 未变（迁移）
- **核心逻辑**：删除指定节点处的初等子树（ET 移动的"减"方向）。
- **变化**：无。

#### `ProposalMixin.et_replace(self, target, new, update_gof=True, verbose=False)` —— ✅ 未变（迁移）
- **核心逻辑**：把 `target` 子树整体替换为 `new` 指定的结构（ET 替换移动的实际执行者）。
- **变化**：无。

#### `ProposalMixin.dE_et(self, target, new, verbose=False)` —— 🔧 **修改（实质）**
- **签名**：不变。
- **核心逻辑**：计算"ET 替换"移动的能量差：先把树改到新状态（`et_replace`），统计移动类型数（接受率修正 `nif/nfi`），做合法性检查，累加先验差 `dEP`（含 `Nopi2_` 二次项），再还原树，返回 `(dEB, dEP, dE, par_values, nif, nfi)`；非法移动返回 `np.inf`。
- **具体变化**：
  1. **新增 parliament 结构惩罚快照**：改树前 `_parl_old = self._capture_parliament_structural()` 记录旧状态惩罚。
  2. **合法性检查改为"合并门控"**：旧版只调 `update_representative()`，返回 `-1` 即禁止；新版先做 `check_constitution()`（维度约束，存在时），不通过即禁止，否则再做 canonical 去重检查——两者任一失败即返回 `(inf, inf, inf, ...)`。
  3. **新增结构惩罚差**：在还原树之前 `dEP += self._parliament_structural_delta(_parl_old)`，把新旧状态的结构惩罚差并入先验能量差。
  4. 若干行尾空白清理。
- **行为差异**：启用专家系统后，ET 移动会额外受维度宪法约束与 parliament 结构惩罚影响；未启用时（`hasattr` 守卫 + 空 parliaments）与旧版等价。

#### `ProposalMixin.dE_lr(self, target, new, verbose=False)` —— 🔧 **修改（实质）**
- **签名**：不变。
- **核心逻辑**：计算"叶子/节点 relabel"移动（把节点值改为 `new`）的能量差，结构与 `dE_et` 类似，返回 `(dEB, dEP, dE, par_values)`。
- **具体变化**：与 `dE_et` 完全同构的三处改动——`_capture_parliament_structural()` 快照、constitution+canonical 合并门控、`_parliament_structural_delta()` 并入 `dEP`；外加行尾空白清理。
- **行为差异**：同上。

#### `ProposalMixin.dE_rr(self, rr=None, verbose=False)` —— 🔧 **修改（实质）**
- **签名**：不变。
- **核心逻辑**：计算根操作（prune 或 replace 两个分支）的能量差，返回 `(dEB, dEP, dE, par_values)`。
- **具体变化**：在 prune 分支与 replace 分支**各自**加入了同构的三件套改动（快照 → 合并门控 → 结构惩罚差并入 `dEP`）；replace 分支原有的 `max_size` 不可替换返回 `inf` 逻辑不变；行尾空白清理。
- **行为差异**：同上，根移动同样受宪法门控与 parliament 惩罚约束。

---

### 2.5 `core/mcmc.py` —— class MCMCMixin（采样循环与预测）

#### `MCMCMixin.mcmc_step(self, verbose=False, p_rr=0.05, p_long=.45)` —— 🔧 修改（仅格式）
- **核心逻辑**：单步 MCMC。按概率选择移动类型（root 替换/剪枝、ET、relabel、长程移动），调用对应 `dE_*` 得能量差，按 Metropolis 准则 `paccept = nif·ω·sf·exp(-dEB/BT - dEP/PT)` 决定接受；接受则落实移动并更新 `EB/EP/E`。
- **变化**：仅行尾空白清理；**逻辑完全不变**。

#### `MCMCMixin.mcmc(self, tracefn='trace.dat', progressfn='progress.dat', ...)` —— ✅ 未变（迁移）
- **核心逻辑**：MCMC 主循环——反复执行 `mcmc_step`，按间隔把样本写入 trace 文件、把进度写入 progress 文件。
- **变化**：无。

#### `MCMCMixin.predict(self, x)` —— 🔧 修改（仅格式）
- **核心逻辑**：用当前树对新数据 `x` 做预测——把树转 SymPy 表达式、代入参数值、lambdify 后求值。
- **变化**：删除一处空白行；**逻辑完全不变**。

#### `MCMCMixin.trace_predict(self, x, burnin=1000, thin=2000, samples=1000, tracefn='trace.dat', progressfn='progress.dat', ...)` —— 🔧 **修改（小修复）**
- **核心逻辑**：跑一条 MCMC 链并对 `x` 做迹预测——边采样边把每个样本的能量/公式/参数写入 trace，并累计预测值得到后验平均预测。
- **具体变化**：写 trace 的一行由
  ```python
  float(self.get_energy(verbose=verbose))
  ```
  改为
  ```python
  float(self.get_energy(verbose=verbose)[0])
  ```
  `get_energy` 返回三元组 `(E, EB, EP)`，旧版直接对元组调 `float()` 在 Python 3 下会抛 `TypeError`——**这是对一个潜在崩溃 bug 的修复**（取总能量 `E` 写入）。
- **行为差异**：新版该路径可正常运行；旧版在执行到写 trace 时会报错（若该代码路径被触发）。

---

### 2.6 旧 `parallel.py` —— class Parallel（❌ 整体删除，未迁入 core/）

| 函数 | 原核心逻辑 | 状态 |
|---|---|---|
| `Parallel.__init__(self, Ts, ops=OPS, variables=['x'], parameters=['a'], prior_par=None, x=None, y=None, ...)` | 创建一组不同温度 `Ts` 的 `Tree` 链，构成并行回火系统 | ❌ 删除 |
| `Parallel.mcmc_step(self, verbose=False, p_rr=0.05, p_long=.45)` | 对每条链各执行一步 MCMC | ❌ 删除 |
| `Parallel.tree_swap(self)` | 按并行回火准则在相邻温度链之间交换整棵树（Metropolis 型交换判据） | ❌ 删除 |
| `Parallel.anneal(self, n=1000, factor=5)` | 模拟退火：逐步升高/降低温度并采样 `n` 步 | ❌ 删除 |
| `Parallel.trace_predict(self, x, ...)` | 在并行回火链上做迹预测（以最低温链为准输出） | ❌ 删除 |

- **影响**：`core/` 不再提供并行回火/退火能力；如需该功能须以快照 `parallel.py` 为基础适配新 `Tree` 组装方式（其内部直接实例化 `Tree`，接口兼容，但需注意新增的 `dimensions/constitutions/parliaments` 参数透传）。

### 2.7 旧 `mcmc.py` 模块级测试函数（❌ 删除）

| 函数 | 原核心逻辑 | 状态 |
|---|---|---|
| `test3(num_points=10, samples=100000)` | 用合成数据 `y ~ 50·sin(x0)/x2 - 4·x1 + 3` 跑完整 MCMC 演示 | ❌ 删除 |
| `test4(num_points=10, samples=1000)` | 小规模合成数据快速演示 | ❌ 删除 |
| `test5(string='(P120 + ...)')` | 从字符串建树后反复 `mcmc_step` + `build_from_string` 往返，验证字符串往返一致性 | ❌ 删除 |

- **说明**：这些函数连同 `if __name__ == '__main__':` 入口均未迁移；本地测试改由 `dynamic-BMS/test/` 目录（pytest，`pytest.ini`）承担。

---

## 3. 新增接线点（core 依赖的外部新代码）

以下函数**不在 core/ 内**，但 core 中 6 个被修改的函数通过 `hasattr` 守卫调用它们，构成新版的主要行为扩展：

| 钩子 | 提供方 | 作用 | 被调用处 |
|---|---|---|---|
| `_init_experts(constitutions, parliaments)` | `experts/base.py: ExpertMixin` | 初始化宪法与议会层 | `TreeBase.__init__` |
| `check_constitution()` | `experts/base.py: ExpertMixin` | 维度约束校验（不合法返回 False） | `build_from_string`, `dE_et`, `dE_lr`, `dE_rr` |
| `_parliament_energy_structural()` | `gate/base.py: GateMixin` | 计算当前树的结构惩罚能 | `get_energy` |
| `_capture_parliament_structural()` | `gate/base.py: GateMixin` | 快照移动前的结构惩罚 | `dE_et`, `dE_lr`, `dE_rr` |
| `_parliament_structural_delta(old)` | `gate/base.py: GateMixin` | 新状态惩罚 − 旧快照 | `dE_et`, `dE_lr`, `dE_rr` |

这些钩子均采用防御式调用（`hasattr` 判断或 `if self.parliaments`），因此 **core/ 在未混入 ExpertMixin 或未配置 parliaments 时，行为与旧版快照严格一致**。

---

## 4. 风险与注意事项

1. **`get_energy` 返回值**：新旧均返回三元组 `(E, EB, EP)`；旧版 `trace_predict` 对其直接 `float()` 是潜在 TypeError，新版已修为 `[0]`。若有其他旧代码沿用 `float(tree.get_energy())` 写法需注意。
2. **`build_from_string` 抛错**：启用 constitution 后，加载维度不合法的表达式字符串会抛 `ValueError`，调用方需准备捕获。
3. **并行回火缺失**：`Parallel` 类未迁移，依赖它的实验脚本需重写或适配。
4. **`latex_val` 缺失**：数值代入版 LaTeX 输出无替代。
5. **名称 mangling**：`__parse_recursive`/`__grow_tree` 的私有名随类名改变（`_Tree__*` → `_TreeBase__*`），若有外部代码以 mangled 名称访问（极不可能）会失效。
