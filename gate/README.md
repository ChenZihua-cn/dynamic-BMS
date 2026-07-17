# Hierarchical Bayesian Hyperprior

Weights \(g_k\) are **not** computed by a neural gating network. In the revised DP-BSR architecture, they are latent variables in a hierarchical Bayesian model, inferred from data via MCMC. The folder name `gate/` is retained for compatibility; conceptually this module implements the **hyperprior** layer.

## Key Design

### Softmax Parameterization

To resolve the non-identifiability of raw exponential weights, \(g_k\) is parameterized via softmax over unconstrained latent variables:

\[
g_k = \frac{\exp(\gamma_k)}{\sum_{j=1}^{K} \exp(\gamma_j)}, \quad \gamma_k \sim \mathcal{N}(0, 1)
\]

MCMC samples in the unconstrained \(\boldsymbol{\gamma} \in \mathbb{R}^K\) space. The simplex constraint \(\sum g_k = 1\) is automatically satisfied.

### Reference Prior Anchoring

For ablation studies and calibration, fix \(g_1 = 1\) (Occam's razor as baseline) and sample only \(g_2, g_3\) relative strengths. \(g_k > 1\) means expert \(k\) is more important than Occam; \(g_k < 1\) means less important. This guarantees that Bayes factors are computable.

## Directory Structure

```
gate/  (→ conceptually: hyperprior/)
├── __init__.py
├── base.py                     # HyperpriorBase abstract class
├── softmax_weight.py           # Softmax parameterization g_k = exp(γ_k) / Σ exp(γ_j)
├── hyperprior.py               # Hyperprior distributions p(γ) ~ N(0,1)
├── joint_proposal.py           # Joint proposal q(f', γ' | f, γ) for coupled MCMC
├── laplace_marginal.py         # Laplace approximation & marginalization over γ
└── posterior_analysis.py       # Post-hoc analysis of g_k posterior (emergence detection)
```
