# experts/parliament/__init__.py
"""Parliament layer: soft physical priors (energy convention).

Each parliament expert returns a penalty contribution that enters dEP.
Weights g_k = softmax(gamma_k) are inferred via hierarchical Bayes
by the gate/ module (Phase 2).
"""

from .base import ParliamentBase
from .asymptotic import AsymptoticPrior
from .occam import OccamPrior
from .parameter_range import ParameterRangePrior

__all__ = ['ParliamentBase', 'AsymptoticPrior', 'OccamPrior',
           'ParameterRangePrior']
