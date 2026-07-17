# MCMC Inference Engine

Extended-space MCMC sampling over \((f, \boldsymbol{\gamma})\), where \(f\) is the expression tree and \(\boldsymbol{\gamma}\) are unconstrained weight latent variables mapped to the simplex via softmax: \(g_k = \exp(\gamma_k) / \sum_j \exp(\gamma_j)\).

## Sampling Strategy

### Basic Alternation
- **Expression update**: Fix \(\boldsymbol{\gamma}\), mutate expression tree via standard BMS operations (subtree replacement, node mutation). Acceptance uses current \(\mathbf{g}\) in the composite prior.
- **Weight update**: Fix \(f\), perform random-walk Metropolis or HMC on \(\boldsymbol{\gamma}\). Acceptance depends on \(\prod_k p_k(f)^{g_k(\boldsymbol{\gamma})} \cdot p(\boldsymbol{\gamma})\).

### Coupled MCMC (counters tree-weight mismatch)
- **Joint Proposal**: When proposing a new tree \(f'\), simultaneously propose a companion weight \(\boldsymbol{\gamma}'\) from the approximate conditional posterior \(q(\boldsymbol{\gamma}' | f')\). The joint Metropolis-Hastings ratio accounts for both dimensions, avoiding the "new tree, stale weights" mismatch.
- **Laplace Marginalization**: At each tree MCMC step, expand the log-posterior around the MAP estimate \(\hat{\boldsymbol{\gamma}}\) to second order, analytically integrate out \(\boldsymbol{\gamma}\), and run MCMC in tree space only. Once the tree posterior stabilizes, sample \(\boldsymbol{\gamma}\) back from \(p(\boldsymbol{\gamma} | f, D)\).

### Phase Strategy
- **Exploration phase**: Joint proposal — keeps weights synchronized with rapidly changing trees.
- **Refinement phase**: Laplace marginalization — tree posterior has stabilized, weight dimension can be collapsed.

## Directory Structure

```
inference/
├── __init__.py
├── hierarchical_prior.py       # p_total(f | g) = Π_k p_k(f)^{g_k}
├── coupled_mcmc.py             # Joint proposal + Laplace marginalization MCMC
├── constitution_check.py       # Constitution-layer gate (dimensional check before likelihood)
├── cache_manager.py            # Model cache (replaces representative mechanism)
├── trace.py                    # MCMC trace logging & γ posterior analysis
└── configs/
    └── default.yaml            # MCMC parameters, phase transition thresholds
```
