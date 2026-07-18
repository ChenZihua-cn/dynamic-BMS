# experts/parliament/base.py
"""Base class for parliament-layer soft priors.

Parliament experts return penalty contributions (energy convention:
positive = worse model, zero = no opinion), consistent with prior_par
in core/tree_base.py.

Two-phase evaluation (Phase 1: structural only; Phase 2: adds fitted):
  evaluate_structural — called pre-fit, depends only on tree structure
  evaluate_fitted    — called post-fit, needs self.par_values
"""


class ParliamentBase:
    """Soft prior penalty contribution (energy convention: positive = worse).

    Two-phase design:
      evaluate_structural — Phase 1 (this plan): tree structure only
      evaluate_fitted    — Phase 2 (deferred): needs self.par_values

    Fitted evaluation uses first dataset: list(tree.par_values.keys())[0].
    """

    name = 'unnamed_parliament'

    def evaluate_structural(self, tree) -> float:
        """Pre-fit penalty. Positive = penalize. Must not trigger fitting."""
        return 0.0

    def evaluate_fitted(self, tree) -> float:
        """Post-fit penalty. Deferred to Phase 2."""
        return 0.0
