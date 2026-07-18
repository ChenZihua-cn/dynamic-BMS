# experts/base.py


class ExpertMixin:
    """Bridge mixin that manages constitutions and parliaments for a Tree.

    Constitutions are hard constraints (0/1 rejection) checked during
    proposal evaluation. Parliaments are soft priors that return
    log-probability adjustments, weighted by the gate/ module via
    hierarchical Bayes.
    """

    def _init_experts(self, constitutions=None, parliaments=None):
        self.constitutions = list(constitutions) if constitutions else []
        self.parliaments = list(parliaments) if parliaments else []

    def check_constitution(self):
        """Run all constitution checks. Returns False if any fails."""
        for c in self.constitutions:
            result = c.check(self)
            if not result.is_valid:
                return False
        return True

    def get_parliament_logp(self):
        """Aggregate log-probability from all parliament experts.

        Each parliament returns a log-probability contribution. The
        gate/ module infers relative weights g_k via hierarchical Bayes.
        """
        return sum(p.evaluate(self) for p in self.parliaments)
