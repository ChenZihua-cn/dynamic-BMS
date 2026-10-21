# 并行回火重构差异记录（当前 `core/parallel.py` vs 论文原版 `parallel.py`）

- 日期：2026-10-20
- 范围：**本仓库重构实现**与 `upstream-bms/parallel.py` 的行为差异
- 目的：记录接口与采样语义的变化，避免把重构版的行为当成 BMS 2020 原版结论
- 结论先说：交换式、升温/降温循环与“只用 $E_B$”的接受式在两边一致；差异主要在构造函数接口、burn-in 期间是否交换、`anneal` 的计时单位、trace 字段与返回值

## 1. 构造函数

| 项目 | 原版 `upstream-bms/parallel.py:12--45` | 当前 `dynamic-BMS/core/parallel.py:27--49` |
|---|---|---|
| 签名 | `Parallel(Ts, ops=OPS, variables=..., parameters=..., max_size=50, prior_par={}, x=None, y=None, fixed_term=None, fixed_term_op='*', extra_variables=[])` | `Parallel(Ts, tree_cls=None, **tree_kwargs)` |
| 温度规范化 | `Ts.sort()`，再 `self.Ts = [str(T) for T in Ts]`；键是字符串，冷链键为 `'1'` | `Ts = sorted(set(float(T) for T in Ts))`；键是浮点，冷链键为 `1.0` |
| 温度数/范围检查 | 无显式检查 | 要求至少 2 个温度，且最小值必须是 `1.0` |
| 冷链 | `Tree(..., BT=1)` | `tree_cls(BT=1.0, **tree_kwargs)` |
| 其余副本 | `Tree(..., root_value=str(self.t1), BT=float(BT))` | `tree_cls(BT=BT, root_value=str(self.t1), **tree_kwargs)` |
| 共享对象 | 显式把每个副本的 `fit_par`、`representative` 指向冷链对象 | 同样按引用共享 |
| 额外构造参数 | 只能传固定列表里的参数 | `**tree_kwargs` 可把 `dimensions`、`constitutions` 等传给每个副本 |

行为影响：

- 原版允许 `Ts` 含重复值（`self.trees` 的字符串键会覆盖），当前用 `set` 去重并拒绝缺少 `BT=1` 的阶梯。当前行为更严格，但会拒绝一些原版能“跑起来”的配置。
- 原版键是字符串，`self.trees['1']`；当前键是浮点，`self.trees[1.0]`。直接复用原版的 trace 解析代码时，若按字符串键访问会出错。
- 两边都从同一个表达式字符串出发，只改 `BT`；都共享 `fit_par` 与 `representative`。这一点没有变化。

## 2. `mcmc_step`

原版 `48--56`：遍历所有树各走一步，然后 `self.t1 = self.trees['1']`。
当前 `56--60`：遍历 `self.trees.values()` 各走一步，不更新 `self.t1`（`self.t1` 在交换时更新）。

语义相同：每个温度各走一步。当前不在每步后重设 `self.t1`，因为 `self.t1` 只在 `tree_swap` 交换冷链时改变；这不影响采样，但若外部代码在 `mcmc_step` 后依赖 `self.t1` 指向冷链，需要知道这个差异。

## 3. `tree_swap`

两边一致：

- 随机选相邻温度对（`randint(0, len(Ts)-2)`）；
- 只读 `EB`，不读 `EP`；
- `dE = EB1*(1/BT2 - 1/BT1) + EB2*(1/BT1 - 1/BT2)`；
- `paccept = 1 if dE <= 0 else exp(-dE)`；
- 接受后交换两棵树与两棵树的 `BT`，并重设 `self.t1`。

当前代码在注释里明确写了“所有副本共享同一个 PT，所以 EP 在交换比中严格抵消；若各副本 PT 不同，这个接受式会破坏细致平衡”。这个解释与原版实现一致，也是书稿第 5 章采用的口径。

## 4. `anneal`

| 项目 | 原版 `87--107` | 当前 `87--106` |
|---|---|---|
| 升温 | 所有树 `BT *= factor` | 同 |
| 加热步数 | `n` 步，每步 `mcmc_step()` + `tree_swap()` | 同 |
| 降温步数 | `2n` 步，每步 `mcmc_step()` + `tree_swap()` | 同 |
| 恢复温度 | `for BT, t in self.trees.items(): t.BT = BT` | 同 |
| 冷链指针 | `self.t1 = self.trees['1']` | `self.t1 = self.trees[1.0]` |
| 返回值 | 无 | 无 |

语义相同。注意 `anneal` 的升温/降温不是可逆的 MCMC 移动；它是交换长期停滞时的启发式救援。这一点原版与重构版一致，书稿第 5 章已明确说明。

## 5. `trace_predict` vs `mcmc`

这是差异最大的一处。

| 项目 | 原版 `trace_predict`（110--176） | 当前 `mcmc`（109--181） |
|---|---|---|
| 输出内容 | 预测值 `ypred`、E、BIC、`max_inactive_swap`、当前树；返回预测 DataFrame | 写 JSON trace（`s, bic, E, E_str, str(tree), par_values`）与 progress（`s, E, bic`）；返回 `n_annealed` |
| burn-in 期间是否交换 | **不交换**：只调用 `mcmc_step()` | **交换**：每步 `mcmc_step()` 后调用 `tree_swap()` |
| 交换停滞计时 | 以“样本号 `s`”计；`last_swap[BT1] = s` | 以“MCMC 步数 `step`”计；`last_swap[BT1] = step` |
| 触发条件 | `max_inactive_swap > anneal`（默认 100 个样本） | `inactive > anneal_after`（默认 0，即关闭；单位是步） |
| 触发后的 `anneal` 步数 | `self.anneal(n=anneal*thin, factor=annealf)` | `self.anneal(n=anneal_after, factor=anneal_factor)` |
| 触发后的重置 | `last_swap = {T: s for T in Ts[:-1]}` | `last_swap = {T: step for T in last_swap}`，并把 `step` 增加 `3*anneal_after` |
| 预测有限性检查 | 有：若 `predict(x)` 含 NaN/Inf，重复该样本 | 无：不做预测，因此没有这个循环 |
| 默认 burn-in/thin/samples | 5000 / 100 / 10000 | 2000 / 10 / 10000（与 `Tree.mcmc` 对齐） |
| 写 trace 的对象 | 只写冷链 `trees['1']` | 只写 `self.t1`（冷链） |

行为影响：

1. **burn-in 期间的交换**：当前版在 burn-in 就交换，原版在 burn-in 不交换。当前注释给出的理由是“否则每个副本只在自己的温度上平衡，冷链会保留它在初始盆地里的位置”。这是行为改进还是偏差，取决于实验设计；比较两边结果时必须知道这一点。
2. **`anneal_after` 的单位**：原版 `anneal=100` 指 100 个样本（每个样本 `thin` 步），当前 `anneal_after` 指 MCMC 步。把原版的 100 直接搬到当前接口，会得到短得多的停滞阈值。
3. **`anneal` 的步数**：原版在触发时跑 `anneal*thin` 步加热 + 2 倍降温；当前跑 `anneal_after` 步加热 + 2 倍降温。若 `thin > 1`，同样的“停滞阈值”对应的回火强度不同。
4. **trace 字段**：原版 `trace_predict` 写的是预测值与树，当前写的是 `Tree.mcmc` 风格的 JSON。两者不能直接用同一个解析器读取。
5. **返回值**：原版返回预测 DataFrame；当前返回 `n_annealed`。调用方若按原版接口取返回值，会拿到整数而不是预测。

## 6. 书稿处理

- 书稿第 5 章只引用 `upstream-bms/parallel.py`，按原版讲 `Parallel.__init__`、`mcmc_step`、`tree_swap`、`anneal`、`trace_predict` 与 10 温度测试段。
- 书稿没有把当前 `core/parallel.py` 的 burn-in 交换、`anneal_after` 步数计、trace 字段等写进正文；这些差异只在本 log 中记录。
- 第 7--9 章的多星球钟摆实验使用 20 温度、burn-in 4000、thin 10、samples 400；这是实验设置，不是 `upstream-bms/parallel.py` 测试段的 10 温度设置，也不是原版 `trace_predict` 的默认 5000/100/10000。
