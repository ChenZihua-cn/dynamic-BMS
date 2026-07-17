# Experiments & Reproducibility

Validates that the hierarchical Bayesian model correctly infers expert weights from data, and that DP-BSR delivers physically robust expressions under distribution shift.

## Phase Plan

| Phase | Goal | Key Verification |
|-------|------|------------------|
| **Phase 1: Synthetic Calibration** | Vary noise level \(\sigma\) and sample size \(N\) on synthetic datasets | High noise → \(g_{\text{Occam}}\) posterior concentrates at high values (emergence, not hand-coded). Small \(N\) → \(g_{\text{asymp}}\) posterior shifts above prior mean. |
| **Phase 2: Ablation Baselines** | Compare against fixed equal weights and hand-designed weight functions | Hierarchical inference outperforms static baselines on model selection metrics |
| **Phase 3: OOD Robustness** | Extrapolation under extreme parameters (very low \(q\), near-extremal spin) | DP-BSR remains physically bounded while pure numerical fits oscillate or diverge |
| **Phase 4: GW BBH Final State** | Predict \(M_f, \chi_f, \omega_{\text{QNM}}\) from SXS/RIT/BAM catalog data | Interpretable closed-form expressions; OOD generalization to unseen NR parameters |

## Directory Structure

```
experiments/
├── run.py                      # Single experiment runner
├── sweep.py                    # Hyperparameter sweep over σ, N
├── configs/
│   ├── phase1_synthetic.yaml   # Synthetic data calibration
│   ├── phase2_ablation.yaml    # Ablation: hierarchical vs. flat vs. hand-designed
│   ├── phase3_ood.yaml         # OOD extrapolation stress test
│   └── phase4_gw_merger.yaml   # GW BBH final state application
└── logs/
    └── ...
```
