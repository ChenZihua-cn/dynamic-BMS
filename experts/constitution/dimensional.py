# experts/constitution/dimensional.py
"""Dimensional analysis via constraint solving over symbolic dimension expressions.

Each parameter (e.g. _a0_) introduces an unknown 7-vector of dimension
exponents.  Operators impose linear constraints on these unknowns.  The
tree is dimensionally consistent iff the resulting linear system has a
solution for all 7 dimension indices independently.
"""

from fractions import Fraction
from dataclasses import dataclass, field

from core.node import Node
from .base import ConstitutionBase


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 7 SI base dimensions: [L, M, T, I, Theta, N, J]
DIM_NAMES = ('L', 'M', 'T', 'I', 'Theta', 'N', 'J')
N_DIMS = len(DIM_NAMES)

_ZERO_COEFFS = ()  # canonical empty coeffs tuple


def _numeric_value(token):
    """Return token as a Fraction if it is a bare number, else None.

    BMS renders pow2/pow3 as `(X ** 2)`/`(X ** 3)`, so numeric leaves occur in
    legitimate expression strings; a number is dimensionless by construction.
    """
    try:
        return Fraction(str(token))
    except (ValueError, TypeError):
        return None


def _is_param_name(token):
    """True for BMS's fitted-constant convention `_x_`.

    ``build_from_string`` re-derives ``tree.parameters`` from the symbols that
    actually occur in the string, so a parameter introduced later by a move is
    not necessarily listed there; the naming convention is what makes it a
    parameter.
    """
    return (isinstance(token, str) and len(token) > 2
            and token.startswith('_') and token.endswith('_'))


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
    dim_index: int          # 0..6, or -1 for a structural problem
                            # (unknown leaf/operator, not tied to one base dim)
    dim_name: str           # 'L', 'M', ... or '*' for structural
    description: str        # human-readable
    conflicting_exprs: tuple = None  # (DimExpr, DimExpr), (name,) or None


@dataclass
class DimensionCheckResult:
    """Result of a dimensional consistency check."""
    is_valid: bool
    failures: list = field(default_factory=list)
    # name -> 7-tuple of Fraction (or None where the structure leaves the
    # exponent undetermined): the dimension the fitted constant must carry
    param_dims: dict = field(default_factory=dict)


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

# `X ** Y` has dimension [X]^(value of Y).  In BMS the exponent is always a
# variable or a fitted parameter, so its numeric value — and hence the
# resulting dimension — is unknown at proposal time.  Carried as one free
# unknown per dimension index to keep the constraint system linear.
POW_OUT = DimExpr(_coeffs=(('<pow>', Fraction(1)),))


def _rule_dimensionless_unary(child_exprs):
    """sin, cos, exp, log, ... — input must be dimensionless."""
    return ZERO, [(child_exprs[0], ZERO)]


def _rule_preserve(child_exprs):
    """- (unary), abs — preserve dimension."""
    return child_exprs[0], []


def _rule_minus(child_exprs):
    """- — unary negation preserves the dimension.

    Hand-written and corpus formulas also use it as binary subtraction, which
    the sampler never builds; that must match dimensions like addition, not
    silently ignore the subtrahend.
    """
    if len(child_exprs) == 2:
        return _rule_add(child_exprs)
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


def _rule_pow(child_exprs):
    """** — the exponent must be dimensionless; the base may carry dimension."""
    return POW_OUT, [(child_exprs[1], ZERO)]


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
    '-':   _rule_minus,
    'abs': _rule_preserve,

    # -- unary, scale dimension --
    'sqrt': _rule_sqrt,
    'pow2': _rule_pow2,
    'pow3': _rule_pow3,

    # -- binary --
    '+':  _rule_add,
    '*':  _rule_mul,
    '/':  _rule_div,
    '**': _rule_pow,
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
    unknowns : list[DimensionFailure]
        Structural problems that no constraint can express: a leaf whose
        dimension is neither declared nor solvable, or an operator missing
        from the rules table.  They are not silently dimensionless.
    """
    constraints = [[] for _ in range(N_DIMS)]
    unknowns = []

    def walk(n):
        if not n.offspring:                      # -- leaf -----------------
            val = n.value
            if val in known_dims:
                dim_tuple = known_dims[val]
                return [DimExpr(const=Fraction(d)) for d in dim_tuple]
            elif val in param_set:
                coeffs = ((val, Fraction(1)),)
                return [DimExpr(_coeffs=coeffs) for _ in range(N_DIMS)]
            elif _numeric_value(val) is not None:
                # Numeric literal, e.g. the 2 in `(L ** 2)` — the way Node.pr
                # renders pow2.  A bare number carries no dimension.
                return [DimExpr() for _ in range(N_DIMS)]
            else:
                unknowns.append(DimensionFailure(
                    dim_index=-1,
                    dim_name='*',
                    description=(
                        f"leaf '{val}' has no declared dimension and is not a "
                        f"fitted parameter: add dimensions['{val}'] = (...) or "
                        f"declare it as a parameter"
                    ),
                    conflicting_exprs=(val,),
                ))
                return [DimExpr() for _ in range(N_DIMS)]

        # -- internal node ---------------------------------------------------
        child_exprs_list = [walk(ch) for ch in n.offspring]
        op = n.value
        rule = OP_DIM_RULES.get(op)
        if rule is None:
            unknowns.append(DimensionFailure(
                dim_index=-1,
                dim_name='*',
                description=(
                    f"'{op}' is not a known operator with a dimension rule "
                    f"(tree was built from {len(n.offspring)} offspring)"
                ),
                conflicting_exprs=(op,),
            ))
            return [DimExpr() for _ in range(N_DIMS)]

        # A literal exponent is a known number, so [X ** k] = [X]^k exactly;
        # only a fitted (unknown-valued) exponent needs the POW_OUT symbol.
        literal_exp = None
        if op == '**' and len(n.offspring) == 2 and not n.offspring[1].offspring:
            literal_exp = _numeric_value(n.offspring[1].value)

        result = []
        for j in range(N_DIMS):
            child_js = [ch[j] for ch in child_exprs_list]
            out_expr, pairs = rule(child_js)
            if literal_exp is not None:
                out_expr = literal_exp * child_js[0]
            result.append(out_expr)
            for lhs, rhs in pairs:
                constraints[j].append((lhs, rhs))

        return result

    root_exprs = walk(node)
    return root_exprs, constraints, unknowns


# ---------------------------------------------------------------------------
# Gauss-Jordan reduction (pure Python + Fraction)
# ---------------------------------------------------------------------------

def _solve_system(augmented):
    """Reduce the linear system A*x = b and back-substitute.

    ``augmented`` is a list of list of Fraction, shape (m, k+1), where the
    last column is the RHS vector *negated*: A*x + b = 0  <=>  A*x = -b.

    Returns
    -------
    (consistent, values)
        ``consistent`` is False if the system has no solution.  ``values`` is a
        length-k list holding each unknown's dimension exponent, or None for an
        unknown the structure does not pin down (it is a pivot whose row still
        involves a free variable).
    """
    if not augmented:
        return True, []

    rows = len(augmented)
    cols = len(augmented[0])
    k = cols - 1                     # number of unknown parameters

    if k == 0:
        # No parameters: each constraint is 0*x + const = 0  →  const == 0
        for row in augmented:
            if row[0] != Fraction(0):
                return False, None
        return True, []

    # Work on a copy
    M = [row[:] for row in augmented]

    pivot_row = 0
    pivots = {}                      # column -> row it was pivoted into
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

        pivots[col] = pivot_row
        pivot_row += 1

    # Check for contradiction: all-zero coeffs but non-zero RHS
    for r in range(rows):
        all_zero = all(M[r][c] == Fraction(0) for c in range(k))
        rhs_nonzero = M[r][k] != Fraction(0)
        if all_zero and rhs_nonzero:
            return False, None

    free_cols = [c for c in range(k) if c not in pivots]
    values = [None] * k
    for col, row in pivots.items():
        # x_col is uniquely fixed only if its row does not still depend on a
        # free unknown (otherwise x_col = rhs - sum(coef * x_free))
        if all(M[row][fc] == Fraction(0) for fc in free_cols):
            values[col] = M[row][k]
    return True, values


# ---------------------------------------------------------------------------
# Constraint solving
# ---------------------------------------------------------------------------

def solve_constraints(constraints_per_index, report_params=None,
                      extra_failures=()):
    """Solve the per-dimension linear systems.

    Parameters
    ----------
    constraints_per_index : list[list[tuple[DimExpr, DimExpr]]]
        Length 7.  Each element is a list of (lhs, rhs) equality constraints
        for that dimension index.
    report_params : iterable of str, optional
        Parameter names whose solved dimension vectors should be reported in
        ``result.param_dims``.  A name is reported as determined only when the
        linear system pins it uniquely for that dimension index.
    extra_failures : iterable of DimensionFailure, optional
        Structural failures found while collecting (unknown leaves/operators),
        which no linear constraint can express.

    Returns
    -------
    DimensionCheckResult
    """
    failures = list(extra_failures)
    report = set(report_params or ())
    param_dims = {p: [None] * N_DIMS for p in sorted(report)}

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

        consistent, values = _solve_system(augmented)
        if not consistent:
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
        elif values is not None:
            for name, exp in zip(params, values):
                if exp is not None and name in param_dims:
                    param_dims[name][j] = exp

    return DimensionCheckResult(
        is_valid=len(failures) == 0,
        failures=failures,
        param_dims={p: tuple(v) for p, v in param_dims.items()},
    )


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def check_dimensional_consistency(root, known_dims, param_set=None,
                                  target_dim=None):
    """Check if an expression tree is dimensionally consistent.

    Parameters
    ----------
    root : Node
        Root of the expression tree.
    known_dims : dict
        Mapping variable_name -> 7-tuple-of-int.
    param_set : set of str, optional
        Leaf names treated as unknown-dimension parameters.  If None, no leaf
        is treated as a parameter (unknown leaves fall back to dimensionless).
    target_dim : 7-tuple of int, optional
        Dimension the whole expression must carry (that of the dependent
        variable).  If None, only internal homogeneity is enforced.

    Returns
    -------
    DimensionCheckResult
    """
    if param_set is None:
        param_set = set()

    root_exprs, constraints, unknowns = collect_constraints(
        root, known_dims, param_set)

    if target_dim is not None:
        if len(target_dim) != N_DIMS:
            raise ValueError(
                f"target_dim must have {N_DIMS} entries, got {len(target_dim)}"
            )
        for j, d in enumerate(target_dim):
            constraints[j].append(
                (root_exprs[j], DimExpr(const=Fraction(d)))
            )

    return solve_constraints(constraints, report_params=param_set,
                             extra_failures=unknowns)


# ---------------------------------------------------------------------------
# Constitution expert
# ---------------------------------------------------------------------------

def _parse_list_to_node(parsed):
    """Convert a ``__parse_recursive`` parse list to a ``Node`` tree.

    ``TreeBase.__parse_recursive`` returns a nested list in the form
    ``[op, [child_list, ...]]`` where each child is itself a parse list.
    This helper converts that representation into a ``Node`` tree so it
    can be fed directly to ``collect_constraints``.
    """
    op, children = parsed
    return Node(value=op, offspring=[_parse_list_to_node(c) for c in children])


def _parse_bms_to_node(string, tree):
    """Parse a BMS expression string into a Node tree.

    **Refactored (2026-07-18):** Previously this function duplicated the
    full BMS parsing state machine from ``TreeBase.__parse_recursive``
    (~40 lines of identical character-by-character logic). The only
    difference was the output format — ``__parse_recursive`` returns a
    parse list ``[op, [child, ...]]`` while the old code returned a
    ``Node`` tree directly.

    Now delegates to ``TreeBase.__parse_recursive`` for parsing and
    converts the parse list via ``_parse_list_to_node``. This eliminates
    the need to keep two parsers in sync when the BMS syntax evolves
    (e.g. adding new operators).
    """
    parsed_list = tree._TreeBase__parse_recursive(string)
    return _parse_list_to_node(parsed_list)


class DimensionalConstitution(ConstitutionBase):
    """Hard constraint: reject dimensionally inconsistent expression trees.

    Reads dimension declarations from ``tree.dimensions`` (a dict mapping
    variable names to 7-tuples of integer exponents) and the required root
    dimension from ``tree.target_dim``.  Fitted parameters automatically
    receive unknown dimension variables solved via constraint satisfaction.
    """

    def check(self, tree):
        known_dims = getattr(tree, 'dimensions', {})
        target_dim = getattr(tree, 'target_dim', None)
        param_set = set(getattr(tree, 'parameters', []))
        if hasattr(tree, 'fixed_parameters'):
            param_set |= set(tree.fixed_parameters)
        # A move can introduce a parameter that tree.parameters never learned
        # about; recognise it by the naming convention, not by the book list.
        param_set |= {n.value for n in getattr(tree, 'nodes', [])
                      if not n.offspring and _is_param_name(n.value)}

        if getattr(tree, 'fixed_term', None) is None:
            return check_dimensional_consistency(tree.root, known_dims,
                                                 param_set, target_dim)

        # Parse fixed_term and combine with main tree via fixed_term_op.
        # Overall formula: (main_tree) fixed_term_op (fixed_term)
        fixed_root = _parse_bms_to_node(tree.fixed_term, tree)
        combined_root = Node(value=tree.fixed_term_op,
                             offspring=[tree.root, fixed_root])
        return check_dimensional_consistency(combined_root, known_dims,
                                             param_set, target_dim)
