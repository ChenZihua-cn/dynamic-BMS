# core/__init__.py

from .constants import OPS
from .node import Node
from .tree_base import TreeBase
from .energy import EnergyMixin
from .proposal import ProposalMixin
from .mcmc import MCMCMixin


class Tree(TreeBase, EnergyMixin, ProposalMixin, MCMCMixin):
    """The fully assembled Tree class for Bayesian Machine Scientist.

    This class is composed via multiple inheritance (MRO: TreeBase ->
    EnergyMixin -> ProposalMixin -> MCMCMixin). All methods and
    attributes are preserved from the original monolithic mcmc.py
    implementation.
    """
    pass


__all__ = ['Node', 'Tree', 'OPS']
