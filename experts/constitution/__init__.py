# experts/constitution/__init__.py

from .base import ConstitutionBase
from .dimensional import (
    DimExpr,
    DIM_NAMES,
    OP_DIM_RULES,
    DimensionCheckResult,
    DimensionFailure,
    collect_constraints,
    solve_constraints,
    check_dimensional_consistency,
    DimensionalConstitution,
)

__all__ = [
    'ConstitutionBase',
    'DimExpr',
    'DIM_NAMES',
    'OP_DIM_RULES',
    'DimensionCheckResult',
    'DimensionFailure',
    'collect_constraints',
    'solve_constraints',
    'check_dimensional_consistency',
    'DimensionalConstitution',
]
