# 量纲嵌入表达式树 — 架构方案（修订版）

## Context

为 BMS 表达式树引入量纲一致性约束（宪法层硬约束）。量纲约束属于宪法层，后续议会层三个软约束专家通过 `gate/` 模块影响 proposal 概率。

## 被否决的方案

原方案：在 Node 上存 `dim` 属性，树修改后递归更新。否决原因：[proposal.py](core/proposal.py) 全部采用试探修改→回退模式，存储可变维度状态必然导致 save/restore 遗漏风险，且需要改动 ~11 处代码。

---

## 核心算法：约束求解，非递归计算

七个遗漏中，Gap 1 和 Gap 2 彻底改变了算法本质。

### 为什么参数不能"默认无量纲"

反例：`_a0_ * x + _b0_`（x 量纲 = L）。若参数默认无量纲：
- `_a0_ * x` → L，`_b0_` → 无量纲
- `+` 要求同量纲 → L ≠ 无量纲 → **非法**

但 `y = ax + b` 是最基本的线性模型。正确的理解是：**参数量纲是未知变量，需要通过约束求解确定**。

### 形式化

一棵表达式树定义了一个关于参数维度变量的线性约束系统。

**符号表示**：对每个维度索引 `j ∈ {L,M,T,I,Θ,N,J}`，每个参数 `p` 有未知整数变量 `p_j`（即参数 p 在维度 j 上的指数）。

对每个树节点，我们计算该节点在维度 j 上的**符号维度表达式**：
```
dim_expr[j] = (coeffs, const)
  其中 coeffs = {p1: c1, p2: c2, ...}  (参数到系数的映射)
      const ∈ ℚ  (常数项)
  语义: dim_j = Σ(coeffs[p] * p_j) + const
```

维度表达式支持加、减、标量乘：
- `expr_a + expr_b`: `({p: ca[p]+cb[p]}, const_a+const_b)`
- `expr_a - expr_b`: `({p: ca[p]-cb[p]}, const_a-const_b)`
- `k * expr`: `({p: k*ca[p]}, k*const_a)`

**叶子节点**：
- 已知量纲的变量（如 `x` 有 dim=L）：`dim_expr[j] = ({}, known_dim[j])`
- 参数（如 `_a0_`）：`dim_expr[j] = ({_a0_: 1}, 0)` —— 参数引入未知变量
- 未在 dimensions dict 中指定的叶子：与参数同样处理（作为自由维度变量）

**内部节点**：传播维度表达式 + 收集约束方程。

| 算符 | 子节点维度表达式 | 输出维度表达式 | 约束方程（每个 j） |
|------|-----------------|---------------|-------------------|
| `sin`, `cos`, `tan`, `exp`, `log`, `sinh`, `cosh`, `tanh`, `fac` | `e_c` | `({}, 0)` | `e_c[j] == 0` |
| `-`, `abs` | `e_c` | `e_c` | 无 |
| `+` | `e_a`, `e_b` | `e_a` | `e_a[j] == e_b[j]` |
| `*` | `e_a`, `e_b` | `e_a + e_b` | 无 |
| `/` | `e_a`, `e_b` | `e_a - e_b` | 无 |
| `**` | `e_a`, `e_b` | `({}, 0)` | `e_a[j] == 0` 且 `e_b[j] == 0` |
| `pow2` | `e_c` | `2 * e_c` | 无 |
| `pow3` | `e_c` | `3 * e_c` | 无 |
| `sqrt` | `e_c` | `(1/2) * e_c` | `e_c[j]` 的系数全为偶数且常数项为偶数（充分条件） |

**共享参数**（Gap 2）：由于每个参数 `_a0_` 在所有出现位置使用相同的维度变量 `_a0_j`，unification 自动被符号表达式系统处理。`_a0_ * x + _a0_ * t` 中，两个 `_a0_` 贡献相同的变量，约束系统会自然地要求 x 和 t 有相同量纲（否则约束 `e_left[j] == e_right[j]` 不可满足）。

### 求解

对于每个维度索引 j，收集树中所有约束方程。每个约束方程形如：
```
Σ(c_p * p_j) + const = 0
```
这是关于未知数 `{p_j | p 是参数}` 的线性方程。7 个维度索引独立求解。

使用高斯消元（或 numpy.linalg）判断方程组是否有解：
- 所有 7 个维度索引的约束系统都有解 → 树合法
- 任一维度索引无解 → 树非法

`sqrt` 的偶性约束：要求 `dim_expr[j]` 中所有系数和常数项为偶数。这是充分非必要条件（保守但正确）。更精确的做法需要求解线性丢番图方程，可作为后续增强。

### 算法复杂度

- 遍历树：O(n)，n ≤ max_size (50)
- 对每个维度索引求解：O(k³)，k = 参数个数（通常 < 10）
- 总复杂度：O(n + 7*k³)，完全可以忽略

---

## 架构维度一：量纲计算放在哪里？

### 方案 1-A：约束求解器作为独立模块（在 experts/constitution/ 内）【推荐】

```
experts/constitution/dimensional.py
    DimExpr         — 符号维度表达式 (coeffs dict + const)
    collect_constraints(node, known_dims) → (root_expr, constraints[])
    check_dimensional_consistency(root, known_dims) → (is_valid, param_dim_solution)
    DimensionalConstitution(ConstitutionBase)
        .check(tree) → bool
```

Nodes 完全不感知维度。`check()` 方法接收完整 tree，遍历树收集约束，求解，返回结果。

### 方案 1-B：约束求解器在 core/ 内

放在 `core/dimensional.py`，作为"基础工具"。理由：可能被除宪法外的其他模块使用。

**不推荐**：量纲分析是典型的"专家知识"，在 BMS 核心算法不需要它。议会层的渐近匹配专家如果需要量纲信息，可以直接依赖 experts/constitution/dimensional.py。

### 此维度的建议

**方案 1-A**。

---

## 架构维度二：宪法门在提案流程中的插入点

### 三个 dE_* 的代码路径分析

**dE_et**（proposal.py:227）：
```
L247: added = self.et_replace(target, new, update_gof=False)  # 试探修改
      # ← 宪法门插入点 1（结构检查，无需拟合）
L253: rep_res = self.update_representative()                    # canonical 门
L261: self.et_replace(added, old, ...)                          # 回退
      # ← 宪法门同样在这里被绕过时需 revert
L286-298: 第二次试探修改（带拟合）→ 计算 dEB → 回退
      # ← 宪法门在此不需要重复检查（结构未变）
```

**dE_lr**（proposal.py:315）：
```
L327: target.value = new       # 直接改 value，不走 et_replace
L329: self.nops[old] -= 1      # 手动维护统计
L330: self.nops[new] += 1
      # ← 宪法门插入点（试探修改后）
L334: rep_res = self.update_representative()
L347: target.value = old       # 回退
L349: self.nops[old] += 1
L350: self.nops[new] -= 1
      # ← 宪法门 revert 时需同步恢复
```

**dE_rr**（proposal.py:406）：
```
L423: self.prune_root()              # 或 L480: self.replace_root(rr)
      # ← 宪法门插入点（试探修改后）
L425: rep_res = self.update_representative()
L428: self.replace_root(oldrr)       # 回退
      # ← 宪法门 revert 时需同步恢复
```

### 宪法门的统一位置

在三个 `dE_*` 方法中，宪法门统一放在**试探修改之后、canonical 门之前**。具体插入点和 revert 逻辑：

| 方法 | 试探修改后 | revert 操作 |
|------|-----------|------------|
| `dE_et` | L247 后 | L261 的 `et_replace(added, old)` |
| `dE_lr` | L330 后 | 恢复 `target.value`, `nops[old]`, `nops[new]` |
| `dE_rr` | L423/L480 后 | L428/L493 的 `replace_root`/`prune_root` |

对于 `dE_lr`，宪法门的 revert 代码需要手动写（不回退整个 et_replace），但逻辑与 canonical 门完全平行——可以参考 canonical 门已有的 revert 逻辑（L335-354）复制相同的模式。

### 此维度的建议

**在三个 dE_* 的试探修改后、canonical 门之前插入**。每个方法的具体插入位置已在上面标注。

---

## 架构维度三：宪法接入 Tree 类

### 方案 3-A：ExpertMixin（多继承）【推荐】

```python
# experts/base.py
class ExpertMixin:
    def _init_experts(self, constitutions=None, parliaments=None):
        self.constitutions = constitutions or []
        self.parliaments = parliaments or []

    def check_constitution(self):
        for c in self.constitutions:
            if not c.check(self):
                return False
        return True

# core/__init__.py
class Tree(TreeBase, EnergyMixin, ProposalMixin, MCMCMixin, ExpertMixin):
    pass
```

**优点**：每个 Tree 实例独立持有宪法/议会列表，无全局状态，测试隔离好，与现有 Mixin 模式一致。

**缺点**：增加一层继承（但 ExpertMixin 不覆盖任何现有方法，MRO 无冲突）。

### 方案 3-B：全局注册表

**不推荐**：全局可变状态，多 Tree 实例/并行 MCMC 有风险。

### 方案 3-C：直接注入 TreeBase

**不推荐**：违反关注点分离，后续宪法/议会增多会使 TreeBase 膨胀。

### 此维度的建议

**方案 3-A（ExpertMixin）**。

---

## 架构维度四：宪法与提案生成的耦合

### 方案 4-A：纯检查（Check-only）【起步推荐】

宪法只在试探修改后做 pass/fail。提案生成完全随机，不感知维度。

**优点**：宪法与提案完全解耦。

**缺点**：可能高拒绝率。

### 方案 4-B：维度感知提案（Dimension-aware proposal）

`_del_et` 选择替换叶子时，先计算父节点对该位置的维度约束，只从兼容的叶子中选。

**优点**：降低拒绝率。

**缺点**：
- proposal.py 需感知维度信息
- 需修正 nif/nfi 计算（见 Gap 7）
- nif/nfi 修正的复杂度：对每个 move type `(oi, of)`，需要判断是否存在至少一个维度兼容的候选。naive 做法需要枚举所有候选并逐一运行约束求解 → 计算量爆炸。优化方案：利用维度表达式预计算 candidate 的维度，建立索引加速查询。这需要额外的工程工作。

### 此维度的建议

**方案 4-A 起步**，根据 Phase 3 效率观测决定是否升级。

---

## 架构维度五：硬约束 vs 软约束

### 方案 5-A：硬拒绝（dE = inf）【起步推荐】

非法表达式绝不被接受。

**连通性分析**（回应用户之前的"允许暂时违宪"问题）：

硬拒绝下合法表达式空间是连通的。证明：`prune_root` 总能将任意合法树退化为单叶子。退化过程：反复 prune_root → 每次产生合法子树 → 最终到单叶子（任何叶子都合法，因为单叶子无约束冲突）。从叶子可以 `_add_et` 重建任意合法表达式。因此任意两个合法状态间存在路径。

但某些合法状态间只有通过"先 prune 到叶子再重建"的迂回路径，效率可能受损。

### 方案 5-B：软惩罚（dE 大但有限）

**缺点**：需要为非法表达式拟合参数（数值可能不稳定），模糊宪法/议会边界，引入超参数。不推荐。

### 方案 5-C：两阶段（burn-in 软 + sampling 硬）

**缺点**：复杂度高。不优先考虑。

### 方案 5-D：维度感知提案（4-B）+ 硬拒绝（5-A）

当 4-A 拒绝率过高时升级到此方案。

### 此维度的建议

**方案 5-A 起步**，通过 Phase 3 效率观测驱动升级决策。

---

## 七个遗漏的处理方案

### Gap 1: 参数量纲 → 约束求解（已整合入核心算法）

见"核心算法"节。算法从递归计算变更为符号维度表达式 + 约束方程收集 + 线性系统求解。

### Gap 2: 共享参数 → 自动由符号系统处理

每个参数在所有出现位置共享同一个维度变量 `p_j`。约束系统自动保证所有出现位置的维度一致。

### Gap 3: `build_from_string` 绕过宪法门

**问题**：[tree_base.py:264-276](core/tree_base.py#L264) `__grow_tree` 调用 `self.et_replace()` 直接修改树，不走 `dE_*`，宪法门完全被绕过。

**修复**：在 `build_from_string` 结束时（L289 `__grow_tree` 完成之后，L292 `fit_par = {}` 之前或之后），调用 `self.check_constitution()`。若违宪，抛 `ValueError` 拒绝该字符串。

```python
def build_from_string(self, string, verbose=False):
    ...
    self.__grow_tree(self.root, tlist[0], tlist[1])
    # 宪法验证
    if not self.check_constitution():
        raise ValueError(f"Expression '{string}' violates dimensional constraints.")
    self.get_sse(verbose=verbose)
    ...
```

同时：`__init__` 中 L113-114 的 `if from_string != None: self.build_from_string(from_string)` 会自然继承此验证。

**需要注意**：若用户在构建 Tree 时未传入宪法，`self.constitutions` 为空，`check_constitution()` 返回 True（空宪法 = 无约束），行为不变。

### Gap 4: `dE_lr` 的代码路径差异

已在"架构维度二"中分析。`dE_lr` 不走 `et_replace`，直接操作 `target.value`。宪法门的插入点和 revert 逻辑需要适配，但模式与 canonical check 已有代码完全平行（复制 L335-354 的 revert 结构）。

### Gap 5: `fixed_term`

**问题**：`fixed_term` 是额外的前置因子表达式（如 `sin(x) * _c0_`），与主树通过 `fixed_term_op`（默认 `*`）组合。最终公式维度 = `dim(tree) op dim(fixed_term)`。

**修复**：
- 为 `fixed_term` 单独构建一个临时 Tree 或直接解析其维度
- 将 `fixed_term` 的维度表达式与主树的维度表达式按 `fixed_term_op` 组合
- 在宪法检查时使用组合后的维度

两种实现方式：
a) 解析 `fixed_term` 字符串为临时 Tree → 调用 `compute_dim_expr()` → 与主树组合
b) 直接用 SymPy 解析 `fixed_term` 并做量纲分析

方式 a) 复用现有解析器（`__parse_recursive`），工程上更简单。

### Gap 6: 非整数维度

`sqrt` 引入 1/2 系数，`**` 可能引入任意指数。

**处理**：
- 维度表达式中的系数和常数使用 `fractions.Fraction`（精确有理数），避免浮点误差
- `sqrt` 的偶性约束：要求输入维度表达式中所有系数和常数为偶数（充分条件）
- 浮点比较的消除：使用精确有理数后，相等性比较是精确的（`Fraction(1,2) == Fraction(1,2)`）
- `**` 算符：保守处理——要求底数和指数均无量纲，输出无量纲。`pow2`/`pow3` 独立处理带量纲的平方/立方

### Gap 7: nif/nfi 维度感知修正

**当前状态**：方案 4-A（纯检查）不需要修正 nif/nfi，因为提案分布未改变。

**若升级到 4-B（维度感知提案）**：
- nif: 对每个 move type `(oi, of)`，判断是否存在至少一个维度兼容的候选
  - 对 `of > 0`: 遍历 `et_space[of]` 中所有 ET，检查其维度是否与目标位置兼容
  - 对 `of == 0`: 遍历 `et_space[0]` 中所有叶子，检查其维度是否与父节点兼容
- 优化：预计算 `et_space` 中每个元素的维度表达式并缓存，避免重复求解
- nif 计算可缓存于 `self._nif_cache`，在树结构改变时失效

这是后续升级的工程工作，不在起步方案中实现。但 `dim_expr` 数据结构应设计为可哈希（使用不可变类型），以支持后续缓存。

---

## 功能分离总览

```
experts/constitution/dimensional.py
    DimExpr              — 符号维度表达式 (frozendict + Fraction)
    collect_constraints  — 遍历树，收集约束方程
    solve_constraints    — 对每个维度索引求解线性系统
    check_dimensional    — 顶层接口：遍历+收集+求解 → bool
    DimensionalConstitution(ConstitutionBase)
        .check(tree) → bool

experts/base.py
    ExpertMixin
        .check_constitution()    → bool
        .get_parliament_logp()   → float  (后续)

core/
    node.py               — 不改
    tree_base.py           — __init__ 加 dimensions + constitutions + parliaments
                             build_from_string 结束时加宪法验证
    proposal.py            — 3 个 dE_* 各加宪法门（不同插入点）
    mcmc.py                — 不改
    __init__.py            — Tree 加 ExpertMixin 继承
```

**职责边界**：
- `DimExpr` / `collect_constraints` / `solve_constraints`：知道量纲规则和线性系统求解，不知道树怎么修改
- `DimensionalConstitution.check()`：知道"调用约束求解器"，不知道 proposal
- `ExpertMixin.check_constitution()`：知道"遍历宪法列表"，不知道具体宪法内容
- `proposal.py`：知道"试探修改后要过宪法门"，不知道宪法门里是什么
- `node.py`：完全不知道量纲存在

---

## 起步方案与升级路径

**起步**：1-A（约束求解）+ 2-A（dE_* 内门）+ 3-A（ExpertMixin）+ 4-A（纯检查）+ 5-A（硬拒绝）

### Phase 1: 核心算法测试

```
DimExpr 运算测试：加减标量乘的正确性
collect_constraints 测试：
  ├─ _a0_ * x + _b0_ (x dim=L) → 约束系统有解
  ├─ x + t (x dim=L, t dim=T) → 约束系统无解
  ├─ sin(x) (x dim=L) → 约束系统无解
  ├─ (x / t) * t (x dim=L, t dim=T) → 有解（dim=L）
  │   解释：_a0_ 无参数，x/t = L/T，*t = L，全程无约束冲突
  ├─ _a0_ * x + _a0_ * t (共享参数, x dim=L, t dim=T)
  │   → 有解：_a0_ 可赋值为无量纲（0 系数满足所有约束）
  │   等等... 不对。_a0_ * x = _a0_L + L, _a0_ * t = _a0_T + T
  │   + 要求相等：(coeffs+const) 左 = (coeffs+const) 右
  │   不一定有解。让我们仔细分析。
  │   _a0_ dim = (a, b, c, ...)
  │   _a0_ * x dim = (a+1, b, c, ...)
  │   _a0_ * t dim = (a, b, c+1, ...)  (t 在 T 维度)
  │   + 要求：a+1 == a → 1 == 0 → 无解！
  │   所以这个树是 INVALID 的。正确行为。检验通过。
  └─ _a0_ * sin(x) (x dim=L)
     → sin 要求输入无量纲: x dim = L, 约束: ({}, 1) == 0 → 1 == 0
     → 无解。正确。
```

### Phase 2: 集成测试

```
DimensionalConstitution.check() 测试：
  ├─ 合法树字符串 → True
  ├─ 非法树字符串 → False
  └─ build_from_string 非法字符串 → raise ValueError

无数据 MCMC：trace 中所有公式 check_constitution() == True
```

### Phase 3: 效率观测（关键决策点）

```
带数据 MCMC → 统计：
  ├─ 宪法门拒绝率
  ├─ canonical 门拒绝率（对比基线）
  └─ BIC trace 收敛情况
```

| 观测 | 行动 |
|------|------|
| 宪法拒绝率 < 50%，trace 正常 | 保持起步方案 |
| 宪法拒绝率 50-80%，trace 正常 | 保持，记录数据 |
| 宪法拒绝率 > 80%，trace stuck | 升级到 5-D（维度感知提案+硬拒绝） |
| 升级 5-D 后性能可接受 | 保持 5-D |
| 升级 5-D 后 nif/nfi 计算瓶颈 | 降级到 5-B（软惩罚）作为过渡方案 |

### Phase 4: 合成数据恢复

```
给定目标公式（如 x/t → 速度），对比：
  ├─ 无约束 MCMC 是否产生 sin(x) 等物理错误？
  └─ 有约束 MCMC 是否杜绝了此类错误？
```

### Phase 5: 前向兼容

验证 `ExpertMixin` 承载多宪法 + 多议会的共存。
