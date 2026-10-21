# Experiments & Reproducibility

Validates that the hierarchical Bayesian model correctly infers expert weights from data, and that DP-BSR delivers physically robust expressions under distribution shift.

## Phase Plan

| Phase | Goal | Key Verification |
|-------|------|------------------|
| **Phase 1: Synthetic Calibration** | Vary noise level \(\sigma\) and sample size \(N\) on synthetic datasets | High noise → \(g_{\text{Occam}}\) posterior concentrates at high values (emergence, not hand-coded). Small \(N\) → \(g_{\text{asymp}}\) posterior shifts above prior mean. |
| **Phase 2: Ablation Baselines** | Compare against fixed equal weights and hand-designed weight functions | Hierarchical inference outperforms static baselines on model selection metrics |
| **Phase 3: OOD Robustness** | Extrapolation under extreme parameters (very low \(q\), near-extremal spin) | DP-BSR remains physically bounded while pure numerical fits oscillate or diverge |
| **Phase 4: GW BBH Final State** | Predict \(M_f, \chi_f, \omega_{\text{QNM}}\) from SXS/RIT/BAM catalog data | Interpretable closed-form expressions; OOD generalization to unseen NR parameters |

## 目录结构（实际现状）

```
experiments/
├── pendulum/                   # 单摆系列实验：三条实验线各占一个目录，产物自包含
│   ├── known_g/                    ① 多星球、g 作为已知输入列（回收测试）
│   │   ├── pendulum.py                 多星球 L,g 主实验 + 对照 A/B/C（先验可配置，可选并行回火）
│   │   ├── prior_pendulum_compare.py   六档先验力度的对照表与图
│   │   ├── run_prior_ablation.sh       上面那轮消融的驱动脚本
│   │   ├── pendulum_runs/              主实验迹线（按先验名分子目录）
│   │   └── pendulum_*.png / pendulum_prior_ablation.md   该轮的图与表
│   ├── lT_powerlaw/                ② 单变量幂律：只给 L 和 T，g 不作为输入
│   │   ├── upstream_lT_powerlaw.py     上游 mcmc.py+parallel.py+Prior/*.dat 的幂律实验
│   │   ├── core_lT_powerlaw.py         本地 core/ 在同数据上的对照（宪法开/关）
│   │   ├── analyze_powerlaw.py         严格判据：宽域单项式检验
│   │   ├── plot_lT_powerlaw.py         三栏图
│   │   ├── diag_constitution_freeze.py 单链诊断：宪法的实际拦截效果
│   │   ├── upstream_runs/              幂律实验迹线（v2/、v3_paper/、v3_matched/、
│   │   │                               _void_run3_crossed_edit/ 为作废数据）
│   │   └── pendulum_lT_powerlaw.png    三栏图产物
│   └── multidataset/               ③ 多数据集：5 份独立数据、不给 g
│       ├── multidataset_planets.py     主脚本（三格设计）
│       ├── multidataset_xdim_pilot.py  给候选列 X 补量纲声明后的反事实试跑
│       ├── multidataset_job.slurm      集群正式跑量作业
│       ├── multidataset_report.md      正式跑量实验报告（2026-10-18，含 7 张图）
│       ├── multidataset_report_figs.py 报告插图 fig1-6 与派生统计 CSV
│       ├── multidataset_report_fig7.py 插图 fig7（闸门 vs 记账）
│       ├── multidataset_runs/          本地输出（正式产物在集群 ~/bms_runs/）
│       └── multidataset_report/        报告插图（fig1-7）与派生统计 CSV
├── prior_ep_compare.py         # 先验 EP 对照（服务于 prior/ 的设计文档，非钟摆专属）
└── test_fit_failure_lockin_natural.ipynb
```

> 2026-10-18 路径迁移：原 `experiments/{pendulum.py, pendulum_runs/, pendulum_*.png,
> upstream_runs/, analyze_powerlaw.py, ...}` 全部移入 `experiments/pendulum/`。
> 早于该日的实验日志里写的旧路径按此映射读。
>
> 2026-10-20 路径迁移：`experiments/pendulum/` 里混着的三条线各自下钻一层，按"g 是否
> 告知、数据是否合并"分家。早于该日的日志（含上面那条）里写的
> `experiments/pendulum/<文件>` 再按这里映射读：
>
> | 旧路径里的文件 | 现在在 |
> |---|---|
> | `pendulum.py`、`prior_pendulum_compare.py`、`run_prior_ablation.sh`、`pendulum_runs/`、`pendulum_planets.png`、`pendulum_bms_wiki_nv13_np13*.png`、`pendulum_prior_ablation.{md,png}` | `pendulum/known_g/` |
> | `upstream_lT_powerlaw.py`、`core_lT_powerlaw.py`、`analyze_powerlaw.py`、`plot_lT_powerlaw.py`、`diag_constitution_freeze.py`、`upstream_runs/`、`pendulum_lT_powerlaw.png` | `pendulum/lT_powerlaw/` |
> | `multidataset_planets.py`、`multidataset_xdim_pilot.py`、`multidataset_job.slurm`、`multidataset_report*`、`multidataset_runs/` | `pendulum/multidataset/` |

## 实验索引

| 实验 | 脚本 | 回答什么问题 | 结论在哪 |
|---|---|---|---|
| 多星球单摆（给 L 和 g） | `pendulum/known_g/pendulum.py` | 仪器分辨率下能否还原 `T = 2π√(L/g)` | `log/pendulum_multiplanet_prior_audit_2026-10-16.md` |
| 论文先验重跑 + 力度消融 | `pendulum/known_g/pendulum.py --prior`、`pendulum/known_g/prior_pendulum_compare.py`、`pendulum/known_g/run_prior_ablation.sh` | 用 [BMS2020] 的 wiki 校准先验替代手工平值，换来什么 | `log/pendulum_paper_prior_rerun_2026-10-17.md`、`pendulum/known_g/pendulum_prior_ablation.md` |
| 先验 EP 对照 | `prior_ep_compare.py` | 真公式与"怪物外衣"的结构先验差多少 nat | `log/ops_complete_prior_design_2026-10-16.md` |
| **单变量幂律（只给 L 和 T）** | `pendulum/lT_powerlaw/upstream_lT_powerlaw.py`、`pendulum/lT_powerlaw/core_lT_powerlaw.py`、`pendulum/lT_powerlaw/analyze_powerlaw.py` | 不告知 g，只用绳长与周期能否还原 `T = a·L^b`、b≈0.5；上游并行回火与本地单温度差多少 | `log/pendulum_lT_upstream_pt_2026-10-18.md` |
| 宪法冻结诊断 | `pendulum/lT_powerlaw/diag_constitution_freeze.py` | 宪法开启后链还剩多少移动性、幂律式能否进入链条 | 同上 §2、§5 |
| **多数据集单摆（5 份独立数据，不给 g）** | `pendulum/multidataset/multidataset_planets.py`、`pendulum/multidataset/multidataset_job.slurm` | 能否发现"共同公式"（能）；把真 g 作候选列能否发现隐藏变量（不能，BIC 参数量按表达式计、负对照同为 0） | `pendulum/multidataset/multidataset_report.md` |

## 复现要点

- 多星球实验是**回收测试**：目标式已知，检验的是搜索走不走得到那儿。g 作为输入列
  直接给出、5 颗星球合并成一张表且只允许一条公式，所以"存在跨星球的共同公式"是
  前提而不是结论，这一档实验不回答"能否发现共同公式/发现 g"。该问题需要
  `Tree(x=dict(...))` 的多数据集设置（每份只给 L、T，不给 g）。边界见
  `pendulum/known_g/pendulum.py` 的"适用范围"一节。
- 所有脚本从仓库根目录运行：`.venv/bin/python experiments/pendulum/<线>/<脚本> ...`，
  其中 `<线>` 是 `known_g` / `lT_powerlaw` / `multidataset`。每条线的迹线与图都写在
  自己目录里，互不覆盖。
- 幂律实验依赖上游克隆 `/home/labubu/30856/symbolic_regression/upstream-bms`
  （bitbucket `rguimera/machine-scientist`，分支 `no_degeneracy`，commit `9defb7aa`）；
  换位置时用环境变量 `UPSTREAM_BMS=<路径>` 覆盖。
- 判据务必用 `pendulum/lT_powerlaw/analyze_powerlaw.py`（宽域单项式检验）。只在数据区间做
  log-log 回归会把 `log(a·L+b)**c` 这类伪装式算成命中，同一批迹线命中率
  4/12 → 2/12 的差别就在这里。
