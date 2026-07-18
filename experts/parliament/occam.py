# experts/parliament/occam.py
"""Occam's Razor: penalize tree size."""

from .base import ParliamentBase


class OccamPrior(ParliamentBase):
    """Penalize tree size. Returns +tree.size (energy convention).

    Contrast with prior_par (tree_base.py:66-69):
      - prior_par: per-operator, fixed weights, e.g. Nopi_sin=10.0
      - OccamPrior: per-node uniform, weight g_occam learnable (Phase 2)

    Phase 1 role: minimal smoke test for the structural pipeline.
    """

    name = 'occam'

    def evaluate_structural(self, tree):
        """Return +tree.size as penalty (larger tree = larger penalty)."""
        return float(tree.size)

    def evaluate_fitted(self, tree):
        return 0.0
