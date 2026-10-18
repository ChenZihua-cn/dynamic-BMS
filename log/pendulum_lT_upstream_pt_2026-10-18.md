# 实验报告：只用绳长与周期，原仓库 BMS（并行回火 + prior.dat）能否还原幂律

日期：2026-10-18
代码：`experiments/pendulum/upstream_lT_powerlaw.py`（上游 `mcmc.py`+`parallel.py`+`Prior/*.dat`）、
`experiments/pendulum/core_lT_powerlaw.py`（本地 `core/` 同数据对照）、
`experiments/pendulum/analyze_powerlaw.py`（严格判据）、`experiments/pendulum/plot_lT_powerlaw.py`
数据与产物：`experiments/pendulum/upstream_runs/`、`experiments/pendulum/pendulum_lT_powerlaw.png`
上游克隆：`/home/labubu/30856/symbolic_regression/upstream-bms`（bitbucket `rguimera/machine-scientist`，
分支 `no_degeneracy`，commit `9defb7aa`，与 `archieve-9defb7aa/` 同版本）
前情：`log/pendulum_paper_prior_rerun_2026-10-17.md`、`log/pendulum_multiplanet_prior_audit_2026-10-16.md`

---

## 0. 结论速览

1. **能，而且很干净。** 只给绳长 L（米尺 0.01 m）和周期 T（秒表），**不告知 g**，
   用上游原仓库代码 + 论文 `prior.dat` + 20 温度并行回火，**12/12 条链**的 MAP
   就是 `sqrt(L) * _a_`，幂律指数 b^ = **0.5000**（0.01 s 计时）/ **0.5029**
   （0.1 s 计时），常数 â = **2.0063**，与 2π/√9.81 = 2.00607 差 **0.01%**。
   三种先验（`nv1.np2`、论文 `nv13.np13`、上游默认平值 10 nat）都是 12/12。
2. **本地还原不了，主因有两个，都不是"搜索算法不行"：**
   - **（结构，已在档1/档2修掉）** 宪法曾把 `a*sqrt(L)` 和 `a*(L**b)` 同时硬拒，
     只给 L 时真公式根本不在空间里。本次会话实测到改动前后两态：改前
     `check_dimensional_consistency` 报 `sqrt requires even-dimensioned input on L,
     got (1)` 与 `Dimension L: constraint system has no solution`；改后同一棵树
     `valid=True`。
   - **（采样，仍在）** `core/` 没有 `Parallel` 类，只有单温度。单温度 + 把
     13 变量语料校准的 `nv13.np13` 用在单变量问题上 → mixing 从 0.25 崩到
     **0.03–0.07**，命中掉到 **2/12**。
3. **先验档位是单温度下的第一杠杆，而且不是本地代码的锅。** 同一份
   `nv13.np13`，上游单温度也只有 2/12、3/12（mixing 0.03）；换成与问题维度匹配的
   `nv1.np2`，上游单温度 7/12、8/12，本地 core+宪法 **10/12、9/12**。
4. **重构是保真的。** `core 无宪法 + nv1.np2` 与 `上游单温度 + nv1.np2` 逐格相同
   （7/12、8/12，mixing 0.242、0.300）——同一批种子走同一条轨迹。
5. **并行回火能救回错配的先验。** 同一份 `nv13.np13`，单温度 2/12 → 20 温度
   12/12。这是 `core/` 目前缺的那一块。

---

## 1. 判据：先把"log 伪装成幂律"筛掉

只在数据区间 L∈[0.15, 2.5] m 上做 log-log 回归会骗人：`log(((_a_*L)+_b_)**_a_)`
这类式子在那段区间里 R² 也能超过 0.999，于是被算作"还原了幂律"。

`experiments/pendulum/analyze_powerlaw.py` 改成两件事一起查：把公式放到
**L∈[0.02, 50] m（近三个数量级）** 上拟合单项式 `A·L^b`，要求

- 宽域最大相对偏离 `dev < 1%`（真单项式恒为 0，伪装式会发散）；
- 区间内 R² ≥ 0.999；
- `|b̂ − 0.5| ≤` 容差（0.01 s 计时取 0.02，0.1 s 计时取 0.08）。

这条判据形式无关：`a*sqrt(L)`、`sqrt(L)/a`、`sqrt(a*L)`、`L**b` 都算同一个族。

**注意**：上一轮报告里按"区间内 R²"统计的命中率因此普遍偏高，本报告全部用宽域判据。

---

## 2. 主表（12 条链/格，MAP 按后验能量 E = BIC/2 + EP 最小取）

`normal` = 10 个周期计时一次、0.01 s 舍入；`coarse` = 单摆一次计时、0.1 s 舍入。

| 搜索端 | 先验 | normal 命中 | coarse 命中 | normal 族占有率 | coarse 族占有率 | mixing (n/c) |
|---|---|---|---|---|---|---|
| 上游 20 温度并行回火 | `nv1.np2` | **12/12** | **12/12** | 97.8% | 83.1% | 0.50 / 0.49 |
| 上游 20 温度并行回火 | `nv13.np13`（论文） | **12/12** | **12/12** | 99.4% | 95.7% | 0.29 / 0.26 |
| 上游 20 温度并行回火 | 上游默认平值 10 nat | **12/12** | **12/12** | 100% | 94.5% | 0.23 / 0.26 |
| 上游 单温度 | `nv1.np2` | 7/12 | 8/12 | 51.3% | 57.7% | 0.24 / 0.30 |
| 上游 单温度 | `nv13.np13`（论文） | 2/12 | 3/12 | 16.7% | 21.4% | **0.03 / 0.03** |
| 本地 core + 宪法 | `nv1.np2` | **10/12** | **9/12** | 81.4% | 72.2% | 0.25 / 0.27 |
| 本地 core + 宪法 | `nv13.np13`（论文） | 2/12 | 2/12 | 16.7% | 16.6% | 0.06 / 0.06 |
| 本地 core 无宪法 | `nv1.np2` | 7/12 | 8/12 | 51.3% | 57.7% | 0.24 / 0.30 |
| 本地 core 无宪法 | `nv13.np13`（论文） | 2/12 | 2/12 | 16.8% | 16.7% | 0.06 / 0.07 |

三条读法：

- **先验列**：单温度下，`nv13.np13` 与 `nv1.np2` 的差距（2/12 vs 10/12）比任何
  算法改动都大。原因是每加一个算子要付 5–8 nat，而单变量问题的真公式只有 2 个
  算子——先验项 EP 与数据项 BIC/2 同量级，链被钉死在原地。上游 README 明说
  超参数要按 nv/np 匹配选文件，这一条我们之前没遵守。
- **宪法列**：`core + 宪法` 与 `core 无宪法` 在 `nv1.np2` 下是 10/12 vs 7/12，
  宪法**不再拖后腿，还略有帮助**（它排除了 log 伪装式）。这与"改前 sqrt(L) 直接
  不合法"的状态是两件事。
- **温度列**：同一份论文先验，单温度 2/12 → 20 温度 12/12。回火是唯一能把
  错配先验救回来的机制。

## 3. g 被彻底藏起来时（5 颗星球混合，只给 L）

| 条件 | MAP 命中 | 曾在链上命中 | 族内 b^ 中位 |
|---|---|---|---|
| 上游 PT `nv1.np2` | 4/12 | 12/12 | 0.5000 |
| 上游 PT `nv13.np13` | 0/12 | 12/12 | 0.5000 |
| 上游 单温度 `nv1.np2` | 4/12 | 10/12 | 0.5000 |

把 5 个不同 g 的数据混在一起又不给 g，等于给单项式乘了一个随机常数——数据本身
不再是干净幂律，MAP 退化成 `(_a_ + L)`（宽域 dev=1.31）。但**只要链上出现过
单项式，其指数就是 0.5000**，12/12 条链都出现过。半定量层面稳健，MAP 层面这个
设定本身是病态的，不该拿它当验收标准。

---

## 4. 并行退火组件到底做了什么（供移植 `core/` 时照抄）

`upstream-bms/parallel.py`，245 行，四个方法：

| 位置 | 作用 |
|---|---|
| `Parallel.__init__` parallel.py:8 | 建 N 棵树，第 i 棵 `BT=Ts[i]`；除最冷树外全部**共享** `fit_par` 与 `representative` 两个字典（parallel.py:43-44），所以一次拟合全链复用 |
| `mcmc_step` parallel.py:48 | 对每个温度各做一次 `Tree.mcmc_step`，温度之间不通信 |
| `tree_swap` parallel.py:59 | 随机挑一对**相邻**温度做 Metropolis 交换，`ΔE = EB1(1/BT2−1/BT1) + EB2(1/BT1−1/BT2)`，**只用 EB（数据项），不含 EP** |
| `anneal` parallel.py:87 | 全部温度 ×factor 升温跑 n 步，再回到原温度跑 2n 步降温 |
| `trace_predict` parallel.py:166 | 主循环里跟踪"某个温度槽位多久没接受过 swap"，超过阈值就调 `anneal` —— 这才是"退火"的触发点 |

配方（README:178-190）：`Ts = [1] + [1.04**k for k in range(1, 20)]`，配
`nv13.np13` 的 `.dat`。本实验照抄。

两个坑（都踩过）：

- `tree_swap` 用 `randint(0, len(Ts)-2)`，**只给 1 个温度会抛
  `ValueError: empty range in randrange(0, 0)`**，把整条 Pool 打断。单温度基线
  只能直接用 `Tree`（`experiments/pendulum/upstream_lT_powerlaw.py` 里已分支处理）。
- 上游拟合偶发跑出复数（`c1**(I*pi)`、`I*sqrt(Abs(c1))`），run1 日志里有 93 处
  异常打印。加了逐步 try/except 的 run2/run4 里 80 万步 0 次抛出，所以**没有**
  把它列为结论，只是移植时值得防一下。

---

## 5. 作废与保留

- `experiments/pendulum/upstream_runs/_void_run3_crossed_edit/`（run3 那批）**作废**，已从默认目录移出：那轮跑在宪法
  档1/档2 改动的中间，前后态混在一起。以 `v2/`（core+宪法、`nv1.np2`）、
  `v3_paper/`（core+宪法、`nv13.np13`）、`v3_matched/`（core 无宪法、`nv1.np2`）为准。
- `core_const_frac` 这个条件（放开 sqrt 偶性）是**空操作**：跑的时候偶性检查已经
  被删掉了，所以它与 `core_const` 逐格相同。已从脚本里移除。
- 顺手记下的上游事实：`build_from_string` 对嵌套一元算符解析畸形
  （`'(_a_ * (sqrt L))'` → 打印成 `'(_a_ * L)(sqrt))'`；
  `'(sqrt (_a_ * L))'` → 崩在 `self.ets[3]` 的 `KeyError`）。采样路径
  （`et_space` → `_add_et`）不受影响，打印正常。这属于尚未批准的档3。

## 6. 下一步（按性价比排序）

1. **把 `Parallel` 移植进 `core/`**（约 240 行，接口与本仓库 `Tree` 完全兼容，
   只需处理 `fit_par`/`representative` 共享与宪法门的温度一致性）。这是唯一
   能把 2/12 拉到 12/12 的动作。
2. **先验按 nv/np 匹配取文件**，并允许 `--prior` 直接指向上游 `.dat`
   （`experiments/pendulum/core_lT_powerlaw.py::load_prior` 已支持）。单变量问题用 `nv1.np*`。
3. `param_dims` 回代（档2 剩下的第二半）：现在 â = 2.0063 只能靠数值对上
   2π/√g，宪法解出的 `[a] = T·L^(-1/2)` 被 `_gaussian_elimination` 丢掉了，
   拿不到可证伪的量纲判据。

## 附：复现

```bash
cd dynamic-BMS
# 上游：并行回火 vs 单温度 × 三种先验
.venv/bin/python experiments/pendulum/upstream_lT_powerlaw.py --chains 12 \
    --data normal,coarse,planets --anneal-after 0 --skip-existing
# 本地 core 对照（宪法开/关；--prior 可给上游 .dat 绝对路径）
.venv/bin/python experiments/pendulum/core_lT_powerlaw.py --chains 12 --data normal,coarse \
    --rundir experiments/pendulum/upstream_runs/v3_paper
# 严格判据（宽域单项式检验）与图
.venv/bin/python experiments/pendulum/analyze_powerlaw.py
.venv/bin/python experiments/pendulum/plot_lT_powerlaw.py
```
