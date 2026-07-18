# experts/base.py
"""ExpertMixin: bridge mixin that manages expert registration for a Tree.

Constitutions are hard constraints (0/1 rejection) checked during
proposal evaluation. Parliament soft priors and their hierarchical
Bayes weight inference are handled by the ``gate/`` module — see
``gate/base.py`` (GateMixin) for gamma/softmax/energy computation.

Responsibility split:
    experts/base.py  — expert registration, constitution checks
    gate/base.py     — parliament weights (gamma, softmax, energy)
"""

from gate.base import GateMixin


class ExpertMixin(GateMixin):
    """Bridge mixin managing constitution checks and expert registration.

    Inherits from ``GateMixin`` for parliament energy computation
    (gamma latent variables, softmax weights, structural penalty cache).

    Linear MRO: Tree -> ... -> ExpertMixin -> GateMixin
    """

    def _init_experts(self, constitutions=None, parliaments=None):
        """Register constitutions and delegate parliament init to GateMixin."""
        self.constitutions = list(constitutions) if constitutions else []
        self._init_gate(parliaments)

    def check_constitution(self):
        """Run all constitution checks. Returns False if any fails."""
        for c in self.constitutions:
            result = c.check(self)
            if not result.is_valid:
                return False
        return True
