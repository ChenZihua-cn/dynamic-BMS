# experts/constitution/base.py


class ConstitutionBase:
    """Base class for constitution-layer hard constraints.

    Constitutions enforce 0/1 rejection: a tree that fails the check
    receives an infinite energy penalty (dE = inf) and is never accepted
    by the MCMC sampler.

    Subclasses implement .check(tree) -> DimensionCheckResult.
    """

    def check(self, tree):
        """Return a check result. is_valid=True means the tree passes."""
        raise NotImplementedError
