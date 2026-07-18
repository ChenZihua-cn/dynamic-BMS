# experts/parliament/asymptotic.py
"""AsymptoticPrior: Phase 1 placeholder for asymptotic matching.

Planned (per ``experts/README.md``):
    Sympy-based symbolic asymptotic computation — evaluate
    :math:`\\lim_{x\\to 0} f(x)` and :math:`\\lim_{x\\to\\infty} f(x)`,
    compare deviations from known theoretical values, and produce a
    continuous penalty proportional to the mismatch.

Current (Phase 1 placeholder):
    AST traversal checking if ``finite_at_zero`` variables appear in
    ``/`` denominators. Each violation adds +0.5 penalty. This is a
    **binary feature check**, not symbolic limit evaluation.

Why downgraded:
    To ship a functional structural pipeline first (Phase 1). The AST
    traversal correctly exercises the parliament -> proposal -> MCMC
    integration path. Phase 2 will upgrade to Sympy symbolic computation
    for continuous penalty scoring.

Limitations:
    Only expresses "variable X must not appear in denominator" — cannot
    express general asymptotic boundary conditions (e.g., "f(x) -> 0
    as x -> 0" with a continuous deviation measure).
"""

from .base import ParliamentBase


class AsymptoticPrior(ParliamentBase):
    """Phase 1 placeholder: penalize division by variables that should be
    finite as :math:`x \\to 0`.

    Current implementation (AST traversal):
        Traverses the tree, counts ``/`` nodes whose denominator subtree
        contains any variable in ``finite_at_zero``. Each violation adds
        +0.5 penalty.

    Phase 2 planned upgrade:
        Symbolic limit computation via Sympy. Replace binary denominator
        check with continuous deviation from known theoretical asymptotics.

    Parameters
    ----------
    finite_at_zero : list of str, optional
        Variable names that should yield finite results as x -> 0.
        If a variable in this list appears in the denominator of any ``/``
        node, the tree is penalized.
    """

    name = 'asymptotic'

    def __init__(self, finite_at_zero=None):
        self.finite_at_zero = set(finite_at_zero) if finite_at_zero else set()

    def evaluate_structural(self, tree):
        if not self.finite_at_zero:
            return 0.0

        penalty = 0.0
        for node in tree.nodes:
            if node.value == '/' and len(node.offspring) == 2:
                if self._subtree_has_variable(node.offspring[1],
                                              self.finite_at_zero):
                    penalty += 0.5
        return penalty

    def evaluate_fitted(self, tree):
        return 0.0

    @staticmethod
    def _subtree_has_variable(node, finite_set):
        """Check if any leaf in the subtree has a value in *finite_set*."""
        if not node.offspring:
            return node.value in finite_set
        return any(
            AsymptoticPrior._subtree_has_variable(child, finite_set)
            for child in node.offspring
        )
