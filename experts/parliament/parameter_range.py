# experts/parliament/parameter_range.py
"""ParameterRangePrior: physical parameter domain check (待做).

Planned:
    Detect predictions falling outside the physically valid region
    (e.g., :math:`q \\notin (0,1]`) and apply a penalty. This is a
    Phase 2 feature — the current stub returns 0.0 for both structural
    and fitted evaluation, providing an import anchor so future
    implementation doesn't break existing import paths.

    Implementation will need per-parameter domain boundaries and a
    mechanism to evaluate the tree's predictions at sampled points
    to detect out-of-range outputs.
"""

from .base import ParliamentBase


class ParameterRangePrior(ParliamentBase):
    """待做 (Phase 2): penalize physical parameter range violations."""

    name = 'parameter_range'

    def evaluate_structural(self, tree):
        return 0.0

    def evaluate_fitted(self, tree):
        return 0.0
