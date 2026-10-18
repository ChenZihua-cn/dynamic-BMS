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
├── pendulum/                   # 单摆系列实验（全部产物自包含在该目录内）
│   ├── pendulum.py                 多星球 L,g 主实验 + 对照 A/B（先验可配置）
│   ├── prior_pendulum_compare.py   六档先验力度的对照表与图
│   ├── run_prior_ablation.sh       上面那轮消融的驱动脚本
│   ├── upstream_lT_powerlaw.py     上游 mcmc.py+parallel.py+Prior/*.dat 的幂律实验
│   ├── core_lT_powerlaw.py         本地 core/ 在同数据上的对照（宪法开/关）
│   ├── analyze_powerlaw.py         严格判据：宽域单项式检验
│   ├── plot_lT_powerlaw.py         三栏图
│   ├── diag_constitution_freeze.py 单链诊断：宪法的实际拦截效果
│   ├── pendulum_runs/              主实验迹线（按先验名分子目录）
│   └── upstream_runs/              幂律实验迹线（v2/、v3_paper/、v3_matched/、
│                                   _void_run3_crossed_edit/ 为作废数据）
├── prior_ep_compare.py         # 先验 EP 对照（服务于 prior/ 的设计文档，非钟摆专属）
└── test_fit_failure_lockin_natural.ipynb
```

> 2026-10-18 路径迁移：原 `experiments/{pendulum.py, pendulum_runs/, pendulum_*.png,
> upstream_runs/, analyze_powerlaw.py, ...}` 全部移入 `experiments/pendulum/`。
> 早于该日的实验日志里写的旧路径按此映射读。

## 实验索引

| 实验 | 脚本 | 回答什么问题 | 结论在哪 |
|---|---|---|---|
| 多星球单摆（给 L 和 g） | `pendulum/pendulum.py` | 仪器分辨率下能否还原 `T = 2π√(L/g)` | `log/pendulum_multiplanet_prior_audit_2026-10-16.md` |
| 论文先验重跑 + 力度消融 | `pendulum/pendulum.py --prior`、`pendulum/prior_pendulum_compare.py`、`pendulum/run_prior_ablation.sh` | 用 [BMS2020] 的 wiki 校准先验替代手工平值，换来什么 | `log/pendulum_paper_prior_rerun_2026-10-17.md` |
| 先验 EP 对照 | `prior_ep_compare.py` | 真公式与"怪物外衣"的结构先验差多少 nat | `log/ops_complete_prior_design_2026-10-16.md` |
| **单变量幂律（只给 L 和 T）** | `pendulum/upstream_lT_powerlaw.py`、`pendulum/core_lT_powerlaw.py`、`pendulum/analyze_powerlaw.py` | 不告知 g，只用绳长与周期能否还原 `T = a·L^b`、b≈0.5；上游并行回火与本地单温度差多少 | `log/pendulum_lT_upstream_pt_2026-10-18.md` |
| 宪法冻结诊断 | `pendulum/diag_constitution_freeze.py` | 宪法开启后链还剩多少移动性、幂律式能否进入链条 | 同上 §2、§5 |

## 复现要点

- 所有脚本从仓库根目录运行：`.venv/bin/python experiments/pendulum/<脚本> ...`。
- 幂律实验依赖上游克隆 `/home/labubu/30856/symbolic_regression/upstream-bms`
  （bitbucket `rguimera/machine-scientist`，分支 `no_degeneracy`，commit `9defb7aa`）；
  换位置时用环境变量 `UPSTREAM_BMS=<路径>` 覆盖。
- 判据务必用 `pendulum/analyze_powerlaw.py`（宽域单项式检验）。只在数据区间做
  log-log 回归会把 `log(a·L+b)**c` 这类伪装式算成命中，同一批迹线命中率
  4/12 → 2/12 的差别就在这里。
