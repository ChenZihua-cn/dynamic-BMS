"""Tests for dimensional analysis (constitution layer)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fractions import Fraction
from experts.constitution.dimensional import (
    DimExpr, DIM_NAMES, N_DIMS, ZERO,
    collect_constraints, solve_constraints,
    check_dimensional_consistency,
    DimensionalConstitution,
    DimensionCheckResult, DimensionFailure,
)
from core import Tree, Node
import numpy as np
import pandas as pd

PASS, FAIL = 0, 0


def t(name):
    """Decorator to wrap test with pass/fail tracking."""
    def dec(fn):
        def wrapper():
            global PASS, FAIL
            try:
                fn()
                PASS += 1
                print(f"[PASS] {name}")
            except Exception as e:
                FAIL += 1
                print(f"[FAIL] {name}: {e}")
        return wrapper
    return dec


# ============================================================================
# DimExpr unit tests
# ============================================================================

@t("DimExpr basic creation and properties")
def test_dim_expr_basic():
    e = DimExpr()
    assert e.is_zero()
    assert e.const == Fraction(0)
    assert len(e.coeffs) == 0

    e2 = DimExpr(_coeffs=(('_a0_', Fraction(1)),), const=Fraction(2))
    assert not e2.is_zero()
    assert e2.const == Fraction(2)

    # Immutability
    try:
        e2.const = Fraction(3)
        assert False, "Should have raised"
    except Exception:
        pass


@t("DimExpr arithmetic: add, sub, mul")
def test_dim_expr_arithmetic():
    a = DimExpr(_coeffs=(('p', Fraction(1)),), const=Fraction(1))
    b = DimExpr(_coeffs=(('p', Fraction(2)),), const=Fraction(3))

    s = a + b
    assert dict(s.coeffs) == {'p': Fraction(3)}
    assert s.const == Fraction(4)

    d = a - b
    assert dict(d.coeffs) == {'p': Fraction(-1)}
    assert d.const == Fraction(-2)

    m = 3 * a
    assert dict(m.coeffs) == {'p': Fraction(3)}
    assert m.const == Fraction(3)

    z = 0 * a
    assert z.is_zero()


@t("DimExpr equality and hashing")
def test_dim_expr_eq_hash():
    a = DimExpr(_coeffs=(('p', Fraction(1)),), const=Fraction(2))
    b = DimExpr(_coeffs=(('p', Fraction(1)),), const=Fraction(2))
    c = DimExpr(_coeffs=(('p', Fraction(1)),), const=Fraction(3))

    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    s = {a, b}
    assert len(s) == 1


@t("DimExpr all_even check")
def test_dim_expr_all_even():
    # ZERO passes (0 is even)
    assert ZERO.all_even()

    # Even coeffs, even const
    e = DimExpr(_coeffs=(('p', Fraction(2)),), const=Fraction(4))
    assert e.all_even()

    # Odd coeff
    e2 = DimExpr(_coeffs=(('p', Fraction(1)),), const=Fraction(2))
    assert not e2.all_even()

    # Odd const
    e3 = DimExpr(_coeffs=(('p', Fraction(2)),), const=Fraction(1))
    assert not e3.all_even()

    # Fractional coeff (not integer)
    e4 = DimExpr(_coeffs=(('p', Fraction(1, 2)),), const=Fraction(0))
    assert not e4.all_even()


# ============================================================================
# Constraint collection tests
# ============================================================================

# Helper to build a simple tree
def leaf_node(value):
    return Node(value, parent=None, offspring=[])


def unary_node(op, child):
    n = Node(op, parent=None, offspring=[child])
    child.parent = n
    return n


def binary_node(op, left, right):
    n = Node(op, parent=None, offspring=[left, right])
    left.parent = n
    right.parent = n
    return n


# Dimension definitions
DIM_L = (1, 0, 0, 0, 0, 0, 0)    # length
DIM_T = (0, 0, 1, 0, 0, 0, 0)    # time
DIM_V = (1, 0, -1, 0, 0, 0, 0)   # velocity (L/T)
DIM_D = (0, 0, 0, 0, 0, 0, 0)    # dimensionless


@t("collect_constraints: leaf with known dim")
def test_leaf_known_dim():
    dims = {'x': DIM_L}
    root = leaf_node('x')
    exprs, constraints, sqrt_info = collect_constraints(root, dims, set())
    # Root dimension should be L: (1,0,0,0,0,0,0)
    for j in range(N_DIMS):
        expected = Fraction(DIM_L[j])
        assert exprs[j].const == expected, f"index {j}: expected {expected}, got {exprs[j].const}"
        assert len(exprs[j].coeffs) == 0
    assert all(len(c) == 0 for c in constraints)
    assert len(sqrt_info) == 0


@t("collect_constraints: leaf parameter (unknown dim)")
def test_leaf_parameter():
    dims = {'x': DIM_L}
    root = leaf_node('_a0_')
    exprs, constraints, sqrt_info = collect_constraints(root, dims, {'_a0_'})
    for j in range(N_DIMS):
        assert exprs[j].const == Fraction(0)
        assert dict(exprs[j].coeffs) == {'_a0_': Fraction(1)}


@t("collect_constraints: sin(dimensionless) – valid")
def test_sin_dimensionless_valid():
    dims = {'x': DIM_L}
    # sin(x / x) = dimensionless input, valid
    div_node = binary_node('/', leaf_node('x'), leaf_node('x'))
    sin_node = unary_node('sin', div_node)

    exprs, constraints, sqrt_info = collect_constraints(sin_node, dims, set())

    # x/x has dim dimensionless (0,0,0,0,0,0,0)
    # sin(dimensionless) = dimensionless
    # No params, so constraint is x/x == ZERO → const == 0 → always true
    for j in range(N_DIMS):
        assert exprs[j].is_zero()


@t("collect_constraints: sin(x) with x dim=L – invalid")
def test_sin_dim_L_invalid():
    dims = {'x': DIM_L}
    sin_node = unary_node('sin', leaf_node('x'))

    _, constraints, _ = collect_constraints(sin_node, dims, set())
    result = solve_constraints(constraints, [])
    assert not result.is_valid
    # Failure should be on dimension L (index 0) since x has dim L
    assert any(f.dim_index == 0 for f in result.failures)


@t("collect_constraints: x + t (dim L + dim T) – invalid")
def test_add_L_plus_T_invalid():
    dims = {'x': DIM_L, 't': DIM_T}
    root = binary_node('+', leaf_node('x'), leaf_node('t'))

    _, constraints, _ = collect_constraints(root, dims, set())
    result = solve_constraints(constraints, [])
    assert not result.is_valid


@t("collect_constraints: a*x + b with param solving – valid linear model")
def test_linear_model_valid():
    """_a0_ * x + _b0_  where x dim=L.
    Should be valid: _a0_ gets dim = output_dim - L, _b0_ gets output_dim."""
    dims = {'x': DIM_L}
    mul = binary_node('*', leaf_node('_a0_'), leaf_node('x'))
    root = binary_node('+', mul, leaf_node('_b0_'))

    _, constraints, _ = collect_constraints(root, dims, {'_a0_', '_b0_'})
    result = solve_constraints(constraints, [])
    assert result.is_valid, f"Should be valid but got failures: {result.failures}"


@t("collect_constraints: a*x + a*t with shared param – invalid")
def test_shared_param_invalid():
    """_a0_ * x + _a0_ * t where x dim=L, t dim=T.
    Cannot have same _a0_ satisfy both L and T constraints."""
    dims = {'x': DIM_L, 't': DIM_T}
    left = binary_node('*', leaf_node('_a0_'), leaf_node('x'))
    right = binary_node('*', leaf_node('_a0_'), leaf_node('t'))
    root = binary_node('+', left, right)

    _, constraints, _ = collect_constraints(root, dims, {'_a0_'})
    result = solve_constraints(constraints, [])
    assert not result.is_valid


@t("collect_constraints: (x/t)*t – valid (dim L after cancel)")
def test_cancel_dim():
    """(x / t) * t where x dim=L, t dim=T. Should give L."""
    dims = {'x': DIM_L, 't': DIM_T}
    div_node = binary_node('/', leaf_node('x'), leaf_node('t'))
    root = binary_node('*', div_node, leaf_node('t'))

    exprs, constraints, sqrt_info = collect_constraints(root, dims, set())
    result = solve_constraints(constraints, sqrt_info)
    assert result.is_valid
    # Root dim should be L
    assert exprs[0].const == Fraction(1)    # L^1
    assert exprs[2].const == Fraction(0)    # T^0


@t("collect_constraints: sqrt(x/x) – valid (input ZERO)")
def test_sqrt_dimensionless_valid():
    dims = {'x': DIM_L}
    div = binary_node('/', leaf_node('x'), leaf_node('x'))
    sqrt_node = unary_node('sqrt', div)

    _, constraints, sqrt_info = collect_constraints(sqrt_node, dims, set())
    result = solve_constraints(constraints, sqrt_info)
    assert result.is_valid


@t("collect_constraints: sqrt(a*x) with x dim=L – conservative reject")
def test_sqrt_a_x_conservative_reject():
    """sqrt(_a0_ * x) where x dim=L. Input dim = ({a0:1}, 1).
    Coefficient 1 is odd → conservative evenness check rejects."""
    dims = {'x': DIM_L}
    mul = binary_node('*', leaf_node('_a0_'), leaf_node('x'))
    sqrt_node = unary_node('sqrt', mul)

    _, constraints, sqrt_info = collect_constraints(sqrt_node, dims, {'_a0_'})
    result = solve_constraints(constraints, sqrt_info)
    assert not result.is_valid
    assert any('sqrt' in f.description.lower() for f in result.failures)


@t("collect_constraints: exp(sin(x/x)) – valid nested dimensionless")
def test_nested_dimensionless():
    """sin(x/x) and then exp: x/x is dimensionless, sin(dimless) is
    dimless, exp(dimless) is dimless. Should be valid."""
    dims = {'x': DIM_L}
    div = binary_node('/', leaf_node('x'), leaf_node('x'))
    sin_node = unary_node('sin', div)
    exp_node = unary_node('exp', sin_node)

    _, constraints, sqrt_info = collect_constraints(exp_node, dims, set())
    result = solve_constraints(constraints, sqrt_info)
    assert result.is_valid


@t("collect_constraints: pow2(x) gives 2*dim(x)")
def test_pow2():
    dims = {'x': DIM_L}
    root = unary_node('pow2', leaf_node('x'))

    exprs, constraints, _ = collect_constraints(root, dims, set())
    # pow2(x): dim L² → L exponent = 2
    assert exprs[0].const == Fraction(2)  # L
    assert exprs[1].const == Fraction(0)  # M


@t("collect_constraints: sqrt(pow2(x)) gives dim(x) – valid")
def test_sqrt_pow2():
    """sqrt(x^2) where x dim=L. Input dim = L² (all even), output = L."""
    dims = {'x': DIM_L}
    pow2_node = unary_node('pow2', leaf_node('x'))
    sqrt_node = unary_node('sqrt', pow2_node)

    exprs, constraints, sqrt_info = collect_constraints(sqrt_node, dims, set())
    result = solve_constraints(constraints, sqrt_info)
    assert result.is_valid
    assert exprs[0].const == Fraction(1)  # L


@t("collect_constraints: x ** y with dims – invalid")
def test_pow_both_must_be_dimensionless():
    """x ** t where x dim=L, t dim=T. Both must be dimensionless."""
    dims = {'x': DIM_L, 't': DIM_T}
    root = binary_node('**', leaf_node('x'), leaf_node('t'))

    _, constraints, _ = collect_constraints(root, dims, set())
    result = solve_constraints(constraints, [])
    assert not result.is_valid


# ============================================================================
# DimensionalConstitution integration tests
# ============================================================================

@t("DimensionalConstitution with valid tree")
def test_constitution_valid_tree():
    dims = {'x': DIM_L}
    const = DimensionalConstitution()
    tree = Tree(
        variables=['x'], parameters=['a', 'b'],
        dimensions=dims,
        constitutions=[const],
        from_string='(_a0_ * x)'
    )
    # _a0_ * x with x dim=L is valid
    assert const.check(tree).is_valid


@t("DimensionalConstitution with invalid tree raises in build_from_string")
def test_constitution_invalid_from_string():
    dims = {'x': DIM_L}
    const = DimensionalConstitution()
    try:
        tree = Tree(
            variables=['x'], parameters=['a'],
            dimensions=dims,
            constitutions=[const],
            from_string='sin(x)'
        )
        # If no exception, check manually
        result = const.check(tree)
        assert not result.is_valid, "sin(x) with x dim=L should be rejected"
    except ValueError as e:
        assert 'dimensional' in str(e).lower()


@t("DimensionalConstitution: backward compat – no constitution, no error")
def test_no_constitution_backward_compat():
    """Without constitution, everything should work as before."""
    tree = Tree(
        variables=['x'], parameters=['a'],
        from_string='sin(x)'  # dimensionally invalid, but no constitution
    )
    # sin(x) has 2 nodes: the sin operator and the x leaf
    assert tree.size == 2


# ============================================================================
# MCMC with constitution gate
# ============================================================================

@t("MCMC with constitution: all accepted formulas are dimensionally valid")
def test_mcmc_constitution_gate():
    dims = {'x0': DIM_L}
    const = DimensionalConstitution()

    x = pd.DataFrame({'x0': np.linspace(1, 10, 20)})
    # y = 2*x + 1 (linear relationship, physically valid)
    y = pd.Series(2.0 * x['x0'] + 1.0 + np.random.normal(0, 0.1, 20))

    tree = Tree(
        variables=['x0'], parameters=['a'],
        dimensions=dims,
        constitutions=[const],
        x=x, y=y,
        max_size=20,
    )

    # Run a few steps
    for step in range(10):
        tree.mcmc_step(verbose=False)

    # Check that current tree passes constitution
    result = const.check(tree)
    assert result.is_valid, f"Tree after MCMC should be valid: {result.failures}"


@t("Empty constitution list – check_constitution returns True")
def test_empty_constitution():
    tree = Tree(variables=['x'], parameters=['a'])
    assert tree.check_constitution() is True


@t("Multiple constitutions – all must pass")
def test_multiple_constitutions():
    """Two DimensionalConstitutions should both pass on a valid tree."""
    dims = {'x': DIM_L}
    c1 = DimensionalConstitution()
    c2 = DimensionalConstitution()
    tree = Tree(
        variables=['x'], parameters=['a'],
        dimensions=dims,
        constitutions=[c1, c2],
        from_string='(_a0_ * x)'
    )
    assert tree.check_constitution() is True


# ============================================================================
# run
# ============================================================================

if __name__ == '__main__':
    # Run all tests
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()

    print(f"\n{'='*60}")
    print(f"Passed: {PASS}, Failed: {FAIL}")
    print(f"{'='*60}")

    if FAIL > 0:
        sys.exit(1)
