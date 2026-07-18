# gate/base.py
"""GateMixin: hierarchical Bayes layer for parliament weight inference.

The name "GateMixin" follows the module name ``gate/``. Conceptually
this is the **hyperprior** layer: it infers parliament weights
:math:`g_k = \\mathrm{softmax}(\\gamma_k)` via MCMC in :math:`\\gamma`
space, not via a neural gating network. The module name is retained
for compatibility with the original architecture documents.

Phase 1 (current): equal weights (all :math:`\\gamma_k = 0`).
Phase 2: add joint proposals over formula + gamma, Laplace marginalization,
and posterior analysis of weight emergence.
"""

import numpy as np


class GateMixin:
    """Mixin providing parliament energy computation to the Tree class.

    Manages parliament experts, gamma latent variables, softmax weights,
    and the structural penalty cache. Designed to sit low in the MRO
    so that ``Tree`` methods like ``canonical()`` are available.

    Note for Phase 2:
        ``ParliamentBase.evaluate_fitted`` has no corresponding integration
        method on GateMixin yet (no ``_parliament_energy_fitted``). This
        must be added when fitted-phase evaluation is implemented.
    """

    _MAX_CACHE_SIZE = 10000

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_gate(self, parliaments=None):
        """Initialize parliament list, gamma variables, and structural cache.

        Called from ExpertMixin._init_experts during Tree construction.
        """
        self.parliaments = list(parliaments) if parliaments else []
        self._init_parliament_gamma()
        # canonical_str -> {expert_name: raw_penalty}
        # Per-expert raw values (unweighted). Weights applied at query
        # time, so gamma changes (Phase 2) don't invalidate cached
        # evaluations.
        self._parliament_structural_cache = {}

    # ------------------------------------------------------------------
    # Gamma latent variables and softmax weights
    # ------------------------------------------------------------------

    def _init_parliament_gamma(self):
        """Initialize gamma latent variables for softmax weight computation.

        gamma = 0.0 gives equal weights. Phase 2 (gate/ module) will
        add MCMC sampling in gamma space via joint proposals.
        """
        self.parliament_gamma = {}
        for p in self.parliaments:
            self.parliament_gamma[p.name] = 0.0

    @property
    def parliament_weights(self):
        """Softmax over gamma: g_k = softmax(gamma_k).

        Equal weights (1/K) when all gamma = 0.
        Returns empty dict when no parliaments are configured.
        """
        if not self.parliaments:
            return {}
        names = [p.name for p in self.parliaments]
        gamma = np.array([self.parliament_gamma.get(n, 0.0) for n in names])
        gamma_shifted = gamma - np.max(gamma)  # log-sum-exp stabilization
        exp_g = np.exp(gamma_shifted)
        weights = exp_g / exp_g.sum()
        return dict(zip(names, weights))

    # ------------------------------------------------------------------
    # Energy computation (energy convention: positive = worse)
    # ------------------------------------------------------------------

    def _parliament_energy_structural(self) -> float:
        """Structural penalty from all parliaments (energy convention).

        Cached by canonical formula at per-expert granularity, so
        gamma changes (Phase 2) don't invalidate cached evaluations.
        """
        if not self.parliaments:
            return 0.0
        canonical = self.canonical()
        if canonical not in self._parliament_structural_cache:
            # FIFO eviction if at capacity
            if len(self._parliament_structural_cache) >= self._MAX_CACHE_SIZE:
                self._parliament_structural_cache.pop(
                    next(iter(self._parliament_structural_cache)))
            raw = {}
            for p in self.parliaments:
                try:
                    raw[p.name] = p.evaluate_structural(self)
                except Exception:
                    raw[p.name] = 0.0
            self._parliament_structural_cache[canonical] = raw
        weights = self.parliament_weights
        raw = self._parliament_structural_cache[canonical]
        return sum(weights.get(name, 0.0) * val
                   for name, val in raw.items())

    # ------------------------------------------------------------------
    # High-level delta methods (the API that proposal.py calls)
    # ------------------------------------------------------------------

    def _capture_parliament_structural(self) -> float:
        """Snapshot structural penalty. Call BEFORE any tree edit (OLD state)."""
        try:
            return self._parliament_energy_structural()
        except Exception:
            return 0.0

    def _parliament_structural_delta(self, old_penalty: float) -> float:
        """Compute structural delta to add to dEP.

        Call while tree is in NEW state (after edit, before revert).
        Returns parl_new - old_penalty (positive = penalty increased).
        """
        try:
            parl_new = self._parliament_energy_structural()
            return parl_new - old_penalty
        except Exception:
            return 0.0
