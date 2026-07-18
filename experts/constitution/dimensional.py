# experts/constitution/dimensional.py
"""Dimensional analysis via constraint solving over symbolic dimension expressions.

Each parameter (e.g. _a0_) introduces an unknown 7-vector of dimension
exponents.  Operators impose linear constraints on these unknowns.  The
tree is dimensionally consistent iff the resulting linear system has a
solution for all 7 dimension indices independently.
"""

from fractions import Fraction
from dataclasses import dataclass, field

from .base import ConstitutionBase


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 7 SI base dimensions: [L, M, T, I, Theta, N, J]
DIM_NAMES = ('L', 'M', 'T', 'I', 'Theta', 'N', 'J')
N_DIMS = len(DIM_NAMES)

_ZERO_COEFFS = ()  # canonical empty coeffs tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DimExpr:
    """Symbolic dimension expression for a single dimension index.

    Semantics:  dim_j = sum(coeffs[p] * p_j) + const

    where p_j is the (unknown) dimension exponent of parameter p for
    dimension index j.
    """

    _coeffs: tuple = field(default=_ZERO_COEFFS)
    # _coeffs: sorted tuple of (param_name: str, coefficient: Fraction)

    const: Fraction = field(default=Fraction(0))

    # -- property (readable alias) -------------------------------------------

    @property
    def coeffs(self):
        return self._coeffs

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _merge_coeffs(c1, c2, sign2):
        """Return sorted coeff tuple for c1 + sign2 * c2 (sign2 = +1 or -1)."""
        d = dict(c1)
        for name, val in c2:
            new_val = d.get(name, Fraction(0)) + sign2 * val
            if new_val == Fraction(0):
                d.pop(name, None)
            else:
                d[name] = new_val
        return tuple(sorted(d.items()))

    # -- arithmetic ----------------------------------------------------------

    def __add__(self, other):
        if not isinstance(other, DimExpr):
            return NotImplemented
        return DimExpr(
            _coeffs=self._merge_coeffs(self._coeffs, other._coeffs, +1),
            const=self.const + other.const,
        )

    def __sub__(self, other):
        if not isinstance(other, DimExpr):
            return NotImplemented
        return DimExpr(
            _coeffs=self._merge_coeffs(self._coeffs, other._coeffs, -1),
            const=self.const - other.const,
        )

    def __mul__(self, k):
        if not isinstance(k, (int, Fraction)):
            return NotImplemented
        k = Fraction(k)
        if k == Fraction(0):
            return DimExpr()
        return DimExpr(
            _coeffs=tuple((name, val * k) for name, val in self._coeffs),
            const=self.const * k,
        )

    def __rmul__(self, k):
        return self.__mul__(k)

    def __neg__(self):
        return DimExpr(
            _coeffs=tuple((name, -val) for name, val in self._coeffs),
            const=-self.const,
        )

    # -- queries -------------------------------------------------------------

    def is_zero(self):
        return len(self._coeffs) == 0 and self.const == Fraction(0)

    def all_even(self):
        """Check whether all coefficients and the constant are even integers.

        This is a *conservative sufficient condition* for sqrt validity:
        it never accepts an invalid formula, but may reject some actually
        valid ones (e.g. when parameter dimension assignments cancel odd
        contributions).  For the common case ``sqrt(dimensionless_expr)``
        the input is ZERO and trivially passes.
        """
        for _, val in self._coeffs:
            if val.denominator != 1 or val.numerator % 2 != 0:
                return False
        return (
            self.const.denominator == 1
            and self.const.numerator % 2 == 0
        )

    def __repr__(self):
        parts = []
        for name, val in self._coeffs:
            parts.append(f"{val}*{name}")
        if self.const != 0 or not parts:
            parts.append(str(self.const))
        return " + ".join(parts)

    # -------------------------------------------------------------------
    # Pickle support for dataclass with __eq__ override
    # -------------------------------------------------------------------
    # dataclass(frozen=True) auto-generates __eq__ / __hash__, but when we
    # manually override __eq__ Python does NOT auto-generate __hash__.
    # Explicit __hash__ fixes that.
    def __eq__(self, other):
        if not isinstance(other, DimExpr):
            return NotImplemented
        return self._coeffs == other._coeffs and self.const == other.const

    def __hash__(self):
        return hash((self._coeffs, self.const))


@dataclass
class DimensionFailure:
    """Describes a single dimensional inconsistency."""
    dim_index: int          # 0..6
    dim_name: str           # 'L', 'M', ...
    description: str        # human-readable
    conflicting_exprs: tuple = None  # (DimExpr, DimExpr) or None


@dataclass
class DimensionCheckResult:
    """Result of a dimensional consistency check."""
    is_valid: bool
    failures: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Operator dimension rules
# ---------------------------------------------------------------------------
#
# Every rule is a function:
#     rule(child_exprs: list[DimExpr]) -> (out_expr, constraints)
#
# out_expr: DimExpr  – the node's dimension expression
# constraints: list[tuple[DimExpr, DimExpr]] – (lhs, rhs) pairs meaning
#               "lhs must equal rhs"
# ---------------------------------------------------------------------------

ZERO = DimExpr()


def _rule_dimensionless_unary(child_exprs):
    """sin, cos, exp, log, ... — input must be dimensionless."""
    return ZERO, [(child_exprs[0], ZERO)]


def _rule_preserve(child_exprs):
    """- (unary), abs — preserve dimension."""
    return child_exprs[0], []


def _rule_add(child_exprs):
    """+ — children must have identical dimensions."""
    return child_exprs[0], [(child_exprs[0], child_exprs[1])]


def _rule_mul(child_exprs):
    """* — multiply dimensions (add exponents)."""
    return child_exprs[0] + child_exprs[1], []


def _rule_div(child_exprs):
    """/ — divide dimensions (subtract exponents)."""
    return child_exprs[0] - child_exprs[1], []


def _rule_pow_both_dimensionless(child_exprs):
    """** — both base and exponent must be dimensionless."""
    return ZERO, [(child_exprs[0], ZERO), (child_exprs[1], ZERO)]


def _rule_pow2(child_exprs):
    """pow2 — square the dimension."""
    return 2 * child_exprs[0], []


def _rule_pow3(child_exprs):
    """pow3 — cube the dimension."""
    return 3 * child_exprs[0], []


def _rule_sqrt(child_exprs):
    """sqrt — halve the dimension. Evenness checked separately."""
    return Fraction(1, 2) * child_exprs[0], []


OP_DIM_RULES = {
    # -- unary, dimensionless in, dimensionless out --
    'sin':  _rule_dimensionless_unary,
    'cos':  _rule_dimensionless_unary,
    'tan':  _rule_dimensionless_unary,
    'exp':  _rule_dimensionless_unary,
    'log':  _rule_dimensionless_unary,
    'sinh': _rule_dimensionless_unary,
    'cosh': _rule_dimensionless_unary,
    'tanh': _rule_dimensionless_unary,
    'fac':  _rule_dimensionless_unary,

    # -- unary, preserve dimension --
    '-':   _rule_preserve,
    'abs': _rule_preserve,

    # -- unary, scale dimension --
    'sqrt': _rule_sqrt,
    'pow2': _rule_pow2,
    'pow3': _rule_pow3,

    # -- binary --
    '+':  _rule_add,
    '*':  _rule_mul,
    '/':  _rule_div,
    '**': _rule_pow_both_dimensionless,
}


# ---------------------------------------------------------------------------
# Constraint collection (tree traversal)
# ---------------------------------------------------------------------------

def collect_constraints(node, known_dims, param_set):
    """Walk the expression tree and collect linear dimension constraints.

    Parameters
    ----------
    node : Node
        Root of the expression tree.
    known_dims : dict
        Mapping variable_name -> 7-tuple-of-int (the known dimension).
    param_set : set of str
        Names that should be treated as unknown-dimension parameters.

    Returns
    -------
    root_exprs : list[DimExpr]  (length 7)
        The symbolic dimension expression at the root for each index.
    constraints : list[list[tuple[DimExpr, DimExpr]]]  (length 7)
        Constraint pairs (lhs, rhs) collected for each dimension index.
    sqrt_even_info : list[tuple[int, DimExpr]]
        (dim_index, input_expr) for each sqrt node, to check evenness.
    """
    constraints = [[] for _ in range(N_DIMS)]
    sqrt_even_info = []

    def walk(n):
        if not n.offspring:                      # -- leaf -----------------
            val = n.value
            if val in known_dims:
                dim_tuple = known_dims[val]
                return [DimExpr(const=Fraction(d)) for d in dim_tuple]
            elif val in param_set:
                coeffs = ((val, Fraction(1)),)
                return [DimExpr(_coeffs=coeffs) for _ in range(N_DIMS)]
            else:
                # unspecified leaf (e.g. numeric literal, extra variable)
                # — default to dimensionless
                return [DimExpr() for _ in range(N_DIMS)]

        # -- internal node ---------------------------------------------------
        child_exprs_list = [walk(ch) for ch in n.offspring]
        op = n.value
        rule = OP_DIM_RULES.get(op)
        if rule is None:
            # Unknown op – propagate dimensionless, no constraints
            return [DimExpr() for _ in range(N_DIMS)]

        result = []
        for j in range(N_DIMS):
            child_js = [ch[j] for ch in child_exprs_list]
            out_expr, pairs = rule(child_js)
            result.append(out_expr)
            for lhs, rhs in pairs:
                constraints[j].append((lhs, rhs))

        # Collect sqrt evenness check
        if op == 'sqrt':
            for j in range(N_DIMS):
                sqrt_even_info.append((j, child_exprs_list[0][j]))

        return result

    root_exprs = walk(node)
    return root_exprs, constraints, sqrt_even_info


# ---------------------------------------------------------------------------
# Gaussian elimination (pure Python + Fraction)
# ---------------------------------------------------------------------------

def _gaussian_elimination(augmented):
    """Check consistency of linear system A*x = b.

    ``augmented`` is a list of list of Fraction, shape (m, k+1), where the
    last column is the RHS vector *negated*: A*x + b = 0  <=>  A*x = -b.

    Returns True if the system has at least one solution.
    """
    if not augmented:
        return True

    rows = len(augmented)
    cols = len(augmented[0])
    k = cols - 1                     # number of unknown parameters

    if k == 0:
        # No parameters: each constraint is 0*x + const = 0  →  const == 0
        for row in augmented:
            if row[0] != Fraction(0):
                return False
        return True

    # Work on a copy
    M = [row[:] for row in augmented]

    pivot_row = 0
    for col in range(k):
        # Find a row with non-zero entry in this column
        best = None
        for r in range(pivot_row, rows):
            if M[r][col] != Fraction(0):
                best = r
                break
        if best is None:
            continue

        # Swap to pivot row
        M[pivot_row], M[best] = M[best], M[pivot_row]

        # Normalize pivot row
        pivot = M[pivot_row][col]
        for c in range(col, cols):
            M[pivot_row][c] /= pivot

        # Eliminate from all other rows
        for r in range(rows):
            if r != pivot_row and M[r][col] != Fraction(0):
                factor = M[r][col]
                for c in range(col, cols):
                    M[r][c] -= factor * M[pivot_row][c]

        pivot_row += 1

    # Check for contradiction: all-zero coeffs but non-zero RHS
    for r in range(rows):
        all_zero = all(M[r][c] == Fraction(0) for c in range(k))
        rhs_nonzero = M[r][k] != Fraction(0)
        if all_zero and rhs_nonzero:
            return False

    return True


# ---------------------------------------------------------------------------
# Constraint solving
# ---------------------------------------------------------------------------

def solve_constraints(constraints_per_index, sqrt_even_info):
    """Solve per-dimension linear systems and check sqrt evenness.

    Parameters
    ----------
    constraints_per_index : list[list[tuple[DimExpr, DimExpr]]]
        Length 7.  Each element is a list of (lhs, rhs) equality constraints
        for that dimension index.
    sqrt_even_info : list[tuple[int, DimExpr]]
        (dim_index, input_expr) for sqrt nodes.

    Returns
    -------
    DimensionCheckResult
    """
    failures = []

    for j in range(N_DIMS):
        pairs = constraints_per_index[j]
        if not pairs:
            continue

        # Collect all parameter names
        all_params = set()
        for lhs, rhs in pairs:
            for name, _ in lhs.coeffs:
                all_params.add(name)
            for name, _ in rhs.coeffs:
                all_params.add(name)

        params = sorted(all_params)
        k = len(params)

        if k == 0:
            # No unknown parameters: each constraint is const == 0
            for lhs, rhs in pairs:
                diff = lhs - rhs
                if diff.const != Fraction(0):
                    failures.append(DimensionFailure(
                        dim_index=j,
                        dim_name=DIM_NAMES[j],
                        description=(
                            f"Dimension {DIM_NAMES[j]}: conflicting fixed "
                            f"constraints ({lhs} vs {rhs})"
                        ),
                        conflicting_exprs=(lhs, rhs),
                    ))
                    break
            continue

        param_to_idx = {p: i for i, p in enumerate(params)}

        # Build augmented matrix: A*x = -const
        augmented = []
        for lhs, rhs in pairs:
            diff = lhs - rhs              # diff == ZERO  →  A*x + const = 0
            row = [Fraction(0)] * (k + 1)  # k params + RHS
            for name, val in diff.coeffs:
                row[param_to_idx[name]] = val
            row[k] = -diff.const
            augmented.append(row)

        if not _gaussian_elimination(augmented):
            # Pick a representative pair for diagnostics
            rep = pairs[0]
            failures.append(DimensionFailure(
                dim_index=j,
                dim_name=DIM_NAMES[j],
                description=(
                    f"Dimension {DIM_NAMES[j]}: constraint system has no "
                    f"solution (params: {', '.join(params)})"
                ),
                conflicting_exprs=rep,
            ))

    # sqrt evenness
    for j, expr in sqrt_even_info:
        if not expr.all_even():
            failures.append(DimensionFailure(
                dim_index=j,
                dim_name=DIM_NAMES[j],
                description=(
                    f"sqrt requires even-dimensioned input on "
                    f"{DIM_NAMES[j]}, got ({expr})"
                ),
                conflicting_exprs=(expr, ZERO),
            ))

    return DimensionCheckResult(
        is_valid=len(failures) == 0,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def check_dimensional_consistency(root, known_dims, param_set=None):
    """Check if an expression tree is dimensionally consistent.

    Parameters
    ----------
    root : Node
        Root of the expression tree.
    known_dims : dict
        Mapping variable_name -> 7-tuple-of-int.
    param_set : set of str, optional
        Leaf names that should be treated as unknown-dimension parameters.
        If None, all leaves not in *known_dims* are treated as parameters.

    Returns
    -------
    DimensionCheckResult
    """
    if param_set is None:
        param_set = set()

    _, constraints, sqrt_info = collect_constraints(root, known_dims,
                                                     param_set)
    return solve_constraints(constraints, sqrt_info)


# ---------------------------------------------------------------------------
# Constitution expert
# ---------------------------------------------------------------------------

class DimensionalConstitution(ConstitutionBase):
    """Hard constraint: reject dimensionally inconsistent expression trees.

    Reads dimension declarations from ``tree.dimensions`` (a dict mapping
    variable names to 7-tuples of integer exponents).  Fitted parameters
    automatically receive unknown dimension variables solved via constraint
    satisfaction.
    """

    def check(self, tree):
        known_dims = getattr(tree, 'dimensions', {})
        param_set = set(getattr(tree, 'parameters', []))
        if hasattr(tree, 'fixed_parameters'):
            param_set |= set(tree.fixed_parameters)
        return check_dimensional_consistency(tree.root, known_dims, param_set)
