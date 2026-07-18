# gate/__init__.py
"""Hyperprior layer: hierarchical Bayes inference over parliament weights.

The module name ``gate/`` is retained for compatibility. Conceptually
this is the **hyperprior** layer — it infers parliament weights
:math:`g_k` via MCMC over latent variables :math:`\\gamma_k`, not via
a neural gating network.

Phase 1 (current): structural-only parliament energy with equal weights.
Phase 2: joint MCMC proposals over formula + gamma space.
"""

from .base import GateMixin

__all__ = ['GateMixin']
