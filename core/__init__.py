# core/__init__.py

from .constants import OPS
from .node import Node
from .tree_base import TreeBase
from .energy import EnergyMixin
from .proposal import ProposalMixin
from .mcmc import MCMCMixin
from experts.base import ExpertMixin


class Tree(TreeBase, EnergyMixin, ProposalMixin, MCMCMixin, ExpertMixin):
    """The fully assembled Tree class for Bayesian Machine Scientist.

    This class is composed via multiple inheritance (MRO: TreeBase ->
    EnergyMixin -> ProposalMixin -> MCMCMixin -> ExpertMixin). All
    methods and attributes are preserved from the original monolithic
    mcmc.py implementation.  ExpertMixin provides constitution and
    parliament layer integration.
    """
    pass


__all__ = ['Node', 'Tree', 'OPS']
