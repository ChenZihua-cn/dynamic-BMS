# 实验手册 (test/)

这里不是软件单元测试，而是**物理实验**：每个 `test_*` 函数是一项实验——
有物理假设、测量量、期望值与容差。实验运行时输出实验记录（测量值、偏差、
判定），用 pytest 查看结果，用 `python test/xx.py` 或 `pytest -s` 查看记录。

## 运行

```bash
.venv/bin/python -m pytest test/ -q          # 全部实验
.venv/bin/python -m pytest test/ -q -s       # 带实验记录输出
.venv/bin/python -m pytest test/ -m "not slow"   # 跳过长时间实验（噪声标度扫描）
.venv/bin/python test/test_conservation.py   # 单独运行一个实验文件
```

## 记录框架：lab.py

| API | 含义 |
|---|---|
| `section(title)` | 实验分区标题 |
| `check(量, 实测, 期望, tol/rtol=, context=)` | 与理论预期比较，超差抛错 |
| `check_above/check_below(量, 实测, 阈值)` | 单向界（如 R² > 阈值、拒绝率 < 80%） |
| `measure(量, 值)` | 只记录观测量（恢复出的表达式、R²、缓存规模） |
| `reproducible_state(seed)` | 固定 numpy + stdlib 双随机源（core 的 MCMC 掷骰子用 stdlib `random`） |

## 实验总表

| 文件 | 物理原则 | 实验数 |
|---|---|---|
| [test_calibration.py](test_calibration.py) | 仪器校准：冒烟检查与规范不变性 | 8 |
| [test_conservation.py](test_conservation.py) | 守恒律：能量记账、尺寸边界、望远镜求和 | 5 |
| [test_known_laws.py](test_known_laws.py) | 已知定律恢复：含噪数据重新发现 ground truth | 10 |
| [test_limits.py](test_limits.py) | 极限行为：x→0 有限性与 Boyle 定律 | 5 |
| [test_dimensions.py](test_dimensions.py) | 量纲齐次性：宪法约束、根量纲锚定与单摆家族 | 34 |
| [test_prior.py](test_prior.py) | 先验超参数：登记覆盖、EP 记账一致、落盘往返、语料校准 | 46 |
| [test_parliament.py](test_parliament.py) | 议会动力学：软先验能量、权重、门控级联 | 27 |
| [test_statistics.py](test_statistics.py) | 统计物理观测：拒绝率、细致平衡、轨迹 | 7 |

## 各实验文件

### test_calibration.py — 仪器校准
正式实验前校准仪器。若这些不变量被破坏，后续所有实验结果不可信。
- 模块导入冒烟；节点打印；初态构造
- 规范不变性（canonical form）：参数重命名、幂等、交换律等价、可区分性
- 态重构可逆性：MCMC 演化后序列化往返一致

### test_conservation.py — 守恒律与对称性
假设：无论提议是否被接受，守恒量沿整条 MCMC 轨迹逐步成立。
- 自由轨迹（无数据）与受驱轨迹（线性数据）：size 有界、E/BIC 有限、树演化
- 能量自洽：`get_energy(reset=True)` 与记账值一致（容差 1e-6）
- **望远镜求和**（新实验）：`ΔE(轨迹) == Σ dE(accepted)`（容差 1e-9）——
  对 `dE_et/dE_lr/dE_rr` 装探测器验证能量记账的增量守恒

### test_known_laws.py — 已知定律恢复
假设：含噪但信息充足的数据下，后验应集中向生成数据的定律。
每轮实验记录恢复出的表达式、训练 R²、ΔBIC 供人工检视。
- 9 条定律：线性、二次、谐波 sin(2x)、有理 x/(1+x²)、指数衰减 exp(-x²)、
  双变量线性、三变量交叉/有理/混合——测试集 R² 阈值 0.05–0.7
- **噪声标度律**（新实验，slow）：σ ∈ {0.01, 0.1, 0.5} 扫描，
  检验恢复精度随噪声的标度行为

### test_limits.py — 极限行为
物理动机：已知观测量在某极限下必须有限（如恒温热力学极限），
则 must-be-finite 变量出现在分母是渐近不匹配。
- AsymptoticPrior 惩罚规则、嵌套分母识别、空声明不表态
- Boyle 定律 V = const/P 的四个渐近排布
- Boyle 定律双专家 MCMC 恢复：软先验与似然失配时似然仍主导

### test_dimensions.py — 量纲齐次性
宪法层的核心假设：加法/减法齐次、乘除相消、初等函数输入无量纲、
sqrt 只把量纲指数折半（分数维合法，`a·√L` 与 `L**b` 的量纲由自由参数吸收）、
`**` 只约束指数无量纲、共享参数须同时满足所有约束。
- DimExpr 代数；量纲判定数据结构
- 约束收集：叶子、加减、乘除、初等函数、sqrt 折半、幂标度
- 根量纲锚定 `target_dim`：因变量量纲作为等式约束并入同一线性系统
- `param_dims` 回代：解出每个待定常数的量纲（钟摆 `[a]`=无量纲、
  理想气体 `[a]`=R 的量纲）；结构未定死的分量标 `None`，不假装有解
- 单摆公式家族：T = 2π√(L/g)、零漂、角度偏移、大振幅级数、比值形式
- 宪法门集成与 fixed_term；先验能量记账与复杂度排序
- 语法门与假放行：畸形串给点名坏 token 的 ValueError；未知叶子/未知算符
  不再默认无量纲，改为 `dim_index=-1` 的结构性 failure
- 提议门：违宪提议 dE=inf 且试算完全回退（dE_et/dE_lr/dE_rr）
- MCMC 集成：轨迹上每步都通过宪法检查（硬约束禁止进入违宪态）

### test_prior.py — 先验超参数
假设：先验以 nat 进入 EP（`exp(-dEB/BT - dEP/PT)`），且这条通道与 core 记账一致。
- 登记表覆盖 OPS 全部算子（对账不缺不多）；`get_prior_par` 返回拷贝不污染登记表
- 缺项语义：一阶项按 `fill_missing` 补齐，二阶项缺即无二次惩罚
- 记账一致：`prior_energy` 与 `Tree.get_energy` 的 EP 分支逐树对拍（含二次项）
- 平值退化 `flat_zero` 使 EP ≡ 0（后验退化为纯 BIC）；落盘往返 .json / .dat
- 语料统计与校准方向：越常见的算子罚项越小，矩匹配迭代使失配单调下降

### test_parliament.py — 议会动力学
软先验层的基本定律：罚项恒正（变大则 EP 升高）、softmax 权重归一、
议会能量是各专家原始评价的加权和、宪法先于议会（级联）、
专家异常静默不中断链条、试算提议完全回退。
- 议会基线/Occam 尺寸罚项；能量记账 EP 含议会贡献
- 议会记忆（缓存）：按规范形、按专家名记录，加权发生在读取时
- 门控级联：违宪提议 inf 拒绝，议会不被询问
- 多专家加权与 softmax 稀释效应；误差韧性
- 端对端：单/双议会配置下 50 步轨迹能量有限、缓存有界

### test_statistics.py — 统计物理观测
把采样链当作统计物理系统测量。
- 宪法门拒绝率（含 plan_origin.md 相变决策规则：<50% 维持硬拒绝，
  50–80% 记录，>80% 升级为量纲感知提议）
- **细致平衡方向性**（新实验）：dE<0 的接受率（≈0.45）应显著高于
  dE>0（≈0.16）——提议比修正使精确比率匹配不成立，故只检验方向性
- 轨迹记录完整性与系综预测（trace_predict）
- 链的涨落（BIC/Energy 非常数 = 未冻结）与预测-实测关联（> 0.5）

## 可复现性说明

`core` 的 MCMC 掷骰子用 **stdlib `random`**（core/mcmc.py），而数据
生成用 **numpy**。旧套件只固定 numpy 种子，轨迹实际不可复现（失败随
测试集合划分游走）。本套件用 `lab.reproducible_state(seed)` 同时固定
两个随机源——同一初态，同一轨迹。
