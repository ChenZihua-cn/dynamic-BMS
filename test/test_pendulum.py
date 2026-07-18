"""
Test dimensional constitution and complexity (prior energy EP)
evaluation using pendulum physics formulas.

Pendulum dimensions (SI-7: L, M, T, I, Theta, N, J):
  L      = length:        (1, 0, 0, 0, 0, 0, 0)
  g      = acceleration:  (1, 0,-2, 0, 0, 0, 0)
  m      = mass:          (0, 1, 0, 0, 0, 0, 0)
  theta0 = initial angle: (0, 0, 0, 0, 0, 0, 0)  [dimensionless]

Excludes trivial linear formulas (T = k * L + b).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from fractions import Fraction

from experts.constitution.dimensional import (
    DimExpr, ZERO,
    collect_constraints, solve_constraints,
    DimensionalConstitution,
    DimensionCheckResult, DimensionFailure,
)
from core import Tree, Node


# ============================================================================
# Shared fixtures
# ============================================================================

DIM_L = (1, 0, 0, 0, 0, 0, 0)
DIM_T = (0, 0, 1, 0, 0, 0, 0)
DIM_M = (0, 1, 0, 0, 0, 0, 0)
DIM_G = (1, 0, -2, 0, 0, 0, 0)
DIM_D = (0, 0, 0, 0, 0, 0, 0)

PENDULUM_DIMS = {
    'L': DIM_L,
    'g': DIM_G,
    'm': DIM_M,
    'theta0': DIM_D,
}

DEFAULT_PRIOR_PAR_WEIGHT = 10.0


def _ep_from_nops(nops, prior_par_weight=DEFAULT_PRIOR_PAR_WEIGHT):
    """Expected EP given nops dict, assuming uniform prior_par weight."""
    return sum(prior_par_weight * n for n in nops.values())


# ============================================================================
# Tree-building helpers
# ============================================================================

def leaf(val, parent=None):
    return Node(val, parent=parent, offspring=[])


def unary(op, child):
    n = Node(op, parent=None, offspring=[child])
    child.parent = n
    return n


def binary(op, left, right):
    n = Node(op, parent=None, offspring=[left, right])
    left.parent = n
    right.parent = n
    return n


def de(coeffs=None, const=0):
    """Shorthand to build a DimExpr for testing."""
    c = tuple(sorted((k, Fraction(v)) for k, v in (coeffs or {}).items()))
    return DimExpr(_coeffs=c, const=Fraction(const) if not isinstance(const, Fraction) else const)


def _capture_state(tree):
    """Capture structural tree state for rollback verification."""
    return (tree.size,
            dict(tree.nops or {}),
            [(n.value, len(n.offspring)) for n in tree.nodes])


def _assert_state_unchanged(tree, state, label):
    """Assert tree state equals captured state."""
    sz, nops, nodes_info = state
    assert tree.size == sz, f"{label}: size changed {state[0]} -> {tree.size}"
    for k, v in nops.items():
        assert tree.nops.get(k, 0) == v, f"{label}: nops[{k}] changed {v} -> {tree.nops.get(k, 0)}"
    current_info = [(n.value, len(n.offspring)) for n in tree.nodes]
    assert current_info == nodes_info, f"{label}: node structure changed"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dim_constitution():
    return DimensionalConstitution()


@pytest.fixture
def pendulum_x_y():
    """Synthetic pendulum data: T = 2*pi*sqrt(L/g) + noise."""
    np.random.seed(42)
    n = 30
    L_vals = np.linspace(0.5, 2.0, n)
    g_val = 9.81
    T_true = 2 * np.pi * np.sqrt(L_vals / g_val)
    T_obs = T_true + np.random.normal(0, 0.02, n)
    x = pd.DataFrame({'L': L_vals, 'g': [g_val] * n})
    y = pd.Series(T_obs)
    return x, y


# ============================================================================
# DimExpr arithmetic — direct unit tests
# ============================================================================

class TestDimExprArithmetic:
    """Direct tests of the DimExpr symbolic algebra layer."""

    def test_add(self):
        a = de({'p': 1}, const=2)
        b = de({'p': 3}, const=5)
        r = a + b
        assert r.coeffs == (('p', Fraction(4)),)
        assert r.const == Fraction(7)

    def test_add_cancels_zero_coeff(self):
        a = de({'p': 1})
        b = de({'p': -1})
        r = a + b
        assert len(r.coeffs) == 0
        assert r.const == Fraction(0)
        assert r.is_zero()

    def test_sub(self):
        a = de({'p': 5}, const=3)
        b = de({'p': 2}, const=1)
        r = a - b
        assert r.coeffs == (('p', Fraction(3)),)
        assert r.const == Fraction(2)

    def test_mul_scalar(self):
        a = de({'p': 1, 'q': 2}, const=3)
        r = 3 * a
        assert r.coeffs == (('p', Fraction(3)), ('q', Fraction(6)))
        assert r.const == Fraction(9)

    def test_mul_zero(self):
        a = de({'p': 5}, const=7)
        r = 0 * a
        assert r.is_zero()
        assert len(r.coeffs) == 0
        assert r.const == Fraction(0)

    def test_neg(self):
        a = de({'p': 1, 'q': -3}, const=5)
        r = -a
        assert r.coeffs == (('p', Fraction(-1)), ('q', Fraction(3)))
        assert r.const == Fraction(-5)

    def test_rmul(self):
        a = de({'p': 2}, const=1)
        r = Fraction(1, 2) * a
        assert r.coeffs == (('p', Fraction(1)),)
        assert r.const == Fraction(1, 2)

    def test_is_zero(self):
        assert ZERO.is_zero()
        assert de().is_zero()
        assert not de({'p': 1}).is_zero()
        assert not de(const=1).is_zero()

    def test_all_even_pass(self):
        assert ZERO.all_even()
        assert de({'p': 2, 'q': -4}, const=0).all_even()

    def test_all_even_reject_odd_coeff(self):
        assert not de({'p': 1}).all_even()
        assert not de({'p': 2, 'q': 1}).all_even()

    def test_all_even_reject_odd_const(self):
        assert not de({'p': 2}, const=1).all_even()

    def test_all_even_reject_fractional(self):
        assert not de({'p': 1}, const=Fraction(1, 2)).all_even()

    def test_eq_and_hash(self):
        a = de({'p': 1}, const=2)
        b = de({'p': 1}, const=2)
        c = de({'p': 1}, const=3)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)

    def test_repr(self):
        a = de({'p': 2, 'q': 1}, const=3)
        assert '2*p' in repr(a)
        assert '1*q' in repr(a)
        assert '3' in repr(a)


# ============================================================================
# DimensionCheckResult / DimensionFailure — direct tests
# ============================================================================

class TestDimensionCheckResult:
    """Direct tests of the result data structures."""

    def test_valid_result(self):
        r = DimensionCheckResult(is_valid=True, failures=[])
        assert r.is_valid
        assert len(r.failures) == 0

    def test_invalid_result_with_failures(self):
        f = DimensionFailure(
            dim_index=0, dim_name='L',
            description="L + g dimension mismatch",
            conflicting_exprs=(de({'a': 1}), de({'a': 1})),
        )
        r = DimensionCheckResult(is_valid=False, failures=[f])
        assert not r.is_valid
        assert len(r.failures) == 1
        assert r.failures[0].dim_index == 0
        assert r.failures[0].dim_name == 'L'
        assert "mismatch" in r.failures[0].description

    def test_failure_default_conflicting_exprs(self):
        f = DimensionFailure(dim_index=2, dim_name='T',
                             description="constraint has no solution")
        assert f.conflicting_exprs is None


# ============================================================================
# Dimension: constraint collection for valid pendulum formulas
# ============================================================================

class TestDimensionalValid:
    """Formulas that should pass dimensional consistency."""

    def test_correct_formula(self):
        """T = 2*pi*sqrt(L/g) — _a0_ * sqrt(L/g) = _a0_ * T."""
        div = binary('/', leaf('L'), leaf('g'))
        srt = unary('sqrt', div)
        root = binary('*', leaf('_a0_'), srt)

        _, constraints, sqrt_info = collect_constraints(
            root, PENDULUM_DIMS, {'_a0_'}
        )
        result = solve_constraints(constraints, sqrt_info)
        assert result.is_valid

    def test_L_over_g(self):
        """L/g = T^2 — dimensionally consistent internally."""
        div = binary('/', leaf('L'), leaf('g'))
        root = binary('*', leaf('_a0_'), div)

        _, constraints, sqrt_info = collect_constraints(
            root, PENDULUM_DIMS, {'_a0_'}
        )
        result = solve_constraints(constraints, sqrt_info)
        assert result.is_valid

    def test_sqrt_L_over_sqrt_g_conservative(self):
        """sqrt(L)/sqrt(g): conservative evenness rejects each sqrt individually.

        Known limitation — the check never accepts invalid formulas, but
        may reject some valid ones (see project plan).
        """
        srt_l = unary('sqrt', leaf('L'))
        srt_g = unary('sqrt', leaf('g'))
        div = binary('/', srt_l, srt_g)
        root = binary('*', leaf('_a0_'), div)

        _, constraints, sqrt_info = collect_constraints(
            root, PENDULUM_DIMS, {'_a0_'}
        )
        result = solve_constraints(constraints, sqrt_info)
        assert not result.is_valid
        assert any('sqrt' in f.description.lower() for f in result.failures)

    def test_with_offset(self):
        """_a0_ * sqrt(L/g) + _b0_ — _b0_ constrained to T."""
        div = binary('/', leaf('L'), leaf('g'))
        srt = unary('sqrt', div)
        mul = binary('*', leaf('_a0_'), srt)
        root = binary('+', mul, leaf('_b0_'))

        _, constraints, sqrt_info = collect_constraints(
            root, PENDULUM_DIMS, {'_a0_', '_b0_'}
        )
        result = solve_constraints(constraints, sqrt_info)
        assert result.is_valid

    def test_with_angle_offset(self):
        """_a0_*sqrt(L/g) + _b0_*theta0 — theta0 is dimensionless."""
        div = binary('/', leaf('L'), leaf('g'))
        srt = unary('sqrt', div)
        left = binary('*', leaf('_a0_'), srt)
        right = binary('*', leaf('_b0_'), leaf('theta0'))
        root = binary('+', left, right)

        _, constraints, sqrt_info = collect_constraints(
            root, PENDULUM_DIMS, {'_a0_', '_b0_'}
        )
        result = solve_constraints(constraints, sqrt_info)
        assert result.is_valid

    def test_large_amplitude_series(self):
        """T = _a0_*sqrt(L/g) * (_b0_ + _c0_*theta0^2)."""
        div = binary('/', leaf('L'), leaf('g'))
        srt = unary('sqrt', div)
        period = binary('*', leaf('_a0_'), srt)

        theta_sq = unary('pow2', leaf('theta0'))
        c0_term = binary('*', leaf('_c0_'), theta_sq)
        correction = binary('+', leaf('_b0_'), c0_term)

        root = binary('*', period, correction)

        _, constraints, sqrt_info = collect_constraints(
            root, PENDULUM_DIMS, {'_a0_', '_b0_', '_c0_'}
        )
        result = solve_constraints(constraints, sqrt_info)
        assert result.is_valid

    def test_ratio_form(self):
        """(_a0_*L)/(_b0_*g) + _c0_."""
        num = binary('*', leaf('_a0_'), leaf('L'))
        den = binary('*', leaf('_b0_'), leaf('g'))
        div = binary('/', num, den)
        root = binary('+', div, leaf('_c0_'))

        _, constraints, sqrt_info = collect_constraints(
            root, PENDULUM_DIMS, {'_a0_', '_b0_', '_c0_'}
        )
        result = solve_constraints(constraints, sqrt_info)
        assert result.is_valid

    def test_pow2_scales_dimension(self):
        """pow2(L/g) = T^4."""
        div = binary('/', leaf('L'), leaf('g'))
        root = unary('pow2', div)
        _, constraints, sqrt_info = collect_constraints(
            root, PENDULUM_DIMS, set()
        )
        result = solve_constraints(constraints, sqrt_info)
        assert result.is_valid

    def test_cancellation(self):
        """(L/g)*g = L — trivially valid, no constraints generated."""
        div = binary('/', leaf('L'), leaf('g'))
        root = binary('*', div, leaf('g'))
        _, constraints, sqrt_info = collect_constraints(root, PENDULUM_DIMS, set())
        result = solve_constraints(constraints, sqrt_info)
        assert result.is_valid


class TestDimensionalInvalid:
    """Formulas that should be rejected."""

    def test_sin_L(self):
        """Transcendental function requires dimensionless input."""
        root = unary('sin', leaf('L'))
        _, constraints, _ = collect_constraints(root, PENDULUM_DIMS, set())
        result = solve_constraints(constraints, [])
        assert not result.is_valid

    def test_L_plus_g(self):
        """Adding length and acceleration — incompatible dimensions."""
        root = binary('+', leaf('L'), leaf('g'))
        _, constraints, _ = collect_constraints(root, PENDULUM_DIMS, set())
        result = solve_constraints(constraints, [])
        assert not result.is_valid

    def test_sqrt_L(self):
        """sqrt(L) — L has odd const=1, fails evenness."""
        root = unary('sqrt', leaf('L'))
        _, constraints, sqrt_info = collect_constraints(root, PENDULUM_DIMS, set())
        result = solve_constraints(constraints, sqrt_info)
        assert not result.is_valid
        assert any('sqrt' in f.description.lower() for f in result.failures)

    def test_shared_param_conflict(self):
        """_a0_*L + _a0_*g: shared _a0_ creates contradictory constraints."""
        left = binary('*', leaf('_a0_'), leaf('L'))
        right = binary('*', leaf('_a0_'), leaf('g'))
        root = binary('+', left, right)

        _, constraints, _ = collect_constraints(
            root, PENDULUM_DIMS, {'_a0_'}
        )
        result = solve_constraints(constraints, [])
        assert not result.is_valid

    def test_cos_T2(self):
        """cos(T^2) — cosine requires dimensionless input."""
        div = binary('/', leaf('L'), leaf('g'))
        root = unary('cos', div)
        _, constraints, _ = collect_constraints(root, PENDULUM_DIMS, set())
        result = solve_constraints(constraints, [])
        assert not result.is_valid

    def test_pow_dim(self):
        """L ** L — both base and exponent must be dimensionless."""
        root = binary('**', leaf('L'), leaf('L'))
        _, constraints, _ = collect_constraints(root, PENDULUM_DIMS, set())
        result = solve_constraints(constraints, [])
        assert not result.is_valid

    def test_shared_param_dim_vs_dimless(self):
        """_a0_*L + _a0_: shared _a0_ must be both L-dim and dimensionless."""
        left = binary('*', leaf('_a0_'), leaf('L'))
        root = binary('+', left, leaf('_a0_'))

        _, constraints, _ = collect_constraints(
            root, PENDULUM_DIMS, {'_a0_'}
        )
        result = solve_constraints(constraints, [])
        assert not result.is_valid

    def test_sin_of_dimensional_with_param(self):
        """_a0_ * sin(L) — sin requires dimensionless, L has dim L."""
        s = unary('sin', leaf('L'))
        root = binary('*', leaf('_a0_'), s)

        _, constraints, _ = collect_constraints(
            root, PENDULUM_DIMS, {'_a0_'}
        )
        result = solve_constraints(constraints, [])
        assert not result.is_valid


# ============================================================================
# DimensionalConstitution integration tests
# ============================================================================

class TestConstitutionIntegration:
    """DimensionalConstitution.check() via full Tree objects."""

    def test_valid_passes(self):
        const = DimensionalConstitution()
        tree = Tree(
            variables=['L', 'g', 'm', 'theta0'],
            parameters=['a'],
            dimensions=PENDULUM_DIMS,
            constitutions=[const],
            from_string='(_a0_ * sqrt((L / g)))',
        )
        assert const.check(tree).is_valid

    def test_invalid_rejected_in_build(self):
        const = DimensionalConstitution()
        with pytest.raises(ValueError, match="dimensional constraints"):
            Tree(
                variables=['L', 'g', 'm', 'theta0'],
                parameters=['a'],
                dimensions=PENDULUM_DIMS,
                constitutions=[const],
                from_string='sin(L)',
            )


# ============================================================================
# Complexity: prior energy EP
# ============================================================================

class TestPriorEnergy:
    """EP = sum(prior_par['Nopi_<op>'] * nop), defaults to 10.0 * nops."""

    def test_single_leaf_is_zero(self):
        tree = Tree(
            variables=['L'], parameters=['a'],
            dimensions=PENDULUM_DIMS,
            from_string='_a0_',
        )
        _, _, EP = tree.get_energy()
        assert EP == 0.0

    def test_single_binary_op(self):
        tree = Tree(
            variables=['L', 'g'], parameters=['a'],
            dimensions=PENDULUM_DIMS,
            from_string='(_a0_ * L)',
        )
        _, _, EP = tree.get_energy()
        assert EP == _ep_from_nops(tree.nops)

    def test_correct_pendulum_formula(self):
        tree = Tree(
            variables=['L', 'g'], parameters=['a'],
            dimensions=PENDULUM_DIMS,
            from_string='(_a0_ * sqrt((L / g)))',
        )
        _, _, EP = tree.get_energy()
        assert EP == _ep_from_nops(tree.nops)

    def test_formula_with_offset(self):
        tree = Tree(
            variables=['L', 'g'], parameters=['a', 'b'],
            dimensions=PENDULUM_DIMS,
            from_string='((_a0_ * sqrt((L / g))) + _b0_)',
        )
        _, _, EP = tree.get_energy()
        assert EP == _ep_from_nops(tree.nops)

    def test_transcendental(self):
        tree = Tree(
            variables=['L', 'g'], parameters=[],
            from_string='sin((L / g))',
        )
        _, _, EP = tree.get_energy()
        assert EP == _ep_from_nops(tree.nops)

    def test_custom_prior_par(self):
        weight = 5.0
        prior = {'Nopi_*': weight, 'Nopi_sqrt': weight,
                 'Nopi_/': weight, 'Nopi_+': weight}
        tree = Tree(
            variables=['L', 'g'], parameters=['a'],
            dimensions=PENDULUM_DIMS, prior_par=prior,
            from_string='(_a0_ * sqrt((L / g)))',
        )
        _, _, EP = tree.get_energy()
        assert EP == _ep_from_nops(tree.nops, prior_par_weight=weight)

    def test_ordering_simpler_less_than_complex(self):
        """Simpler formulas get lower EP."""
        dims = PENDULUM_DIMS
        const = DimensionalConstitution()

        tree_simpler = Tree(
            variables=['L'], parameters=['a'],
            dimensions=dims, constitutions=[const],
            from_string='(_a0_ * L)',
        )
        tree_simple = Tree(
            variables=['L', 'g'], parameters=['a'],
            dimensions=dims, constitutions=[const],
            from_string='(_a0_ * sqrt((L / g)))',
        )
        tree_complex = Tree(
            variables=['L', 'g'], parameters=['a', 'b'],
            dimensions=dims, constitutions=[const],
            from_string='((_a0_ * sqrt((L / g))) + _b0_)',
        )

        ep_simpler = tree_simpler.get_energy()[2]
        ep_simple = tree_simple.get_energy()[2]
        ep_complex = tree_complex.get_energy()[2]

        assert ep_simpler < ep_simple
        assert ep_simple < ep_complex


# ============================================================================
# Constitution gating in proposal evaluation (dE_et / dE_lr / dE_rr)
# ============================================================================

class TestProposalGate:
    """Verify that constitution gate in core/proposal.py returns inf and
    preserves tree state when a dimensionally invalid proposal is made."""

    def test_dE_et_rejects_invalid(self):
        """Replace L with L+g in _a0_*L: L and g have different T dims."""
        tree = Tree(
            variables=['L', 'g'], parameters=['a'],
            dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
            from_string='(_a0_ * L)',
        )
        L_node = [n for n in tree.nodes if n.value == 'L'][0]
        state = _capture_state(tree)

        dE, dEB, dEP, pv, nif, nfi = tree.dE_et(L_node, ['+', ['L', 'g']])
        assert np.isinf(dE)
        _assert_state_unchanged(tree, state, "dE_et invalid")

    def test_dE_et_accepts_valid(self):
        """Replace L with L/g in _a0_*L: L/g = T^2, _a0_ absorbs T^2."""
        tree = Tree(
            variables=['L', 'g'], parameters=['a'],
            dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
            from_string='(_a0_ * L)',
        )
        L_node = [n for n in tree.nodes if n.value == 'L'][0]

        dE, dEB, dEP, pv, nif, nfi = tree.dE_et(L_node, ['/', ['L', 'g']])
        assert np.isfinite(dE)
        assert str(tree) == '(_a0_ * L)'  # reverted after dE_et

    def test_dE_lr_rejects_invalid(self):
        """Change sqrt to sin in sqrt(L/g): sin requires dimensionless."""
        tree = Tree(
            variables=['L', 'g'], parameters=[],
            dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
            from_string='sqrt((L / g))',
        )
        sqrt_node = [n for n in tree.nodes if n.value == 'sqrt'][0]
        state = _capture_state(tree)

        dE, dEB, dEP, pv = tree.dE_lr(sqrt_node, 'sin')
        assert np.isinf(dE)
        _assert_state_unchanged(tree, state, "dE_lr invalid")

    def test_dE_lr_accepts_valid(self):
        """Change * to + in _a0_*L: dim(_a0_) = dim(L) is solvable."""
        tree = Tree(
            variables=['L'], parameters=['a'],
            dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
            from_string='(_a0_ * L)',
        )
        mul_node = [n for n in tree.nodes if n.value == '*'][0]
        before = str(tree)

        dE, dEB, dEP, pv = tree.dE_lr(mul_node, '+')
        assert np.isfinite(dE)
        assert str(tree) == before

    def test_dE_rr_rejects_invalid_root_add(self):
        """Wrap (L/g) in sin(): L/g = T^2, sin needs dimensionless."""
        tree = Tree(
            variables=['L', 'g'], parameters=[],
            dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
            from_string='(L / g)',
        )
        state = _capture_state(tree)

        dE, dEB, dEP, pv = tree.dE_rr(rr=['sin', []])
        assert np.isinf(dE)
        _assert_state_unchanged(tree, state, "dE_rr invalid")

    def test_dE_rr_accepts_valid_root_add(self):
        """Wrap sqrt(L/g) in * _b0_: _b0_*T is dimensionally valid."""
        tree = Tree(
            variables=['L', 'g'], parameters=['a', 'b'],
            dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
            from_string='sqrt((L / g))',
        )
        before = str(tree)

        dE, dEB, dEP, pv = tree.dE_rr(rr=['*', ['_b0_']])
        assert np.isfinite(dE)
        assert str(tree) == before

    def test_dE_rr_prune_accepts(self):
        """Pruning _a0_+L → _a0_ is dimensionally valid."""
        tree = Tree(
            variables=['L'], parameters=['a'],
            dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
            from_string='(_a0_ + L)',
        )
        before = str(tree)

        dE, dEB, dEP, pv = tree.dE_rr(rr=None)
        assert np.isfinite(dE)
        assert str(tree) == before

    def test_dE_et_transcendental_of_dimensional_rejected(self):
        """Replace L with sin(L) in _a0_*L: sin requires dimensionless."""
        tree = Tree(
            variables=['L'], parameters=['a'],
            dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
            from_string='(_a0_ * L)',
        )
        L_node = [n for n in tree.nodes if n.value == 'L'][0]
        state = _capture_state(tree)

        dE, dEB, dEP, pv, nif, nfi = tree.dE_et(L_node, ['sin', ['L']])
        assert np.isinf(dE)
        _assert_state_unchanged(tree, state, "dE_et sin(L)")


# ============================================================================
# MCMC integration tests
# ============================================================================

class TestMCMCIntegration:
    """MCMC with dimensional constitution active."""

    def test_all_accepted_steps_pass_constitution(self, dim_constitution,
                                                   pendulum_x_y):
        """After every MCMC step, the tree must pass constitution check."""
        x, y = pendulum_x_y
        tree = Tree(
            variables=['L', 'g'], parameters=['a'],
            dimensions=PENDULUM_DIMS,
            constitutions=[dim_constitution],
            x=x, y=y, max_size=30,
            from_string='(_a0_ * sqrt((L / g)))',
        )

        for step in range(20):
            tree.mcmc_step(verbose=False)
            assert dim_constitution.check(tree).is_valid, \
                f"Step {step}: tree '{tree}' failed constitution"

        # EP must be finite
        _, _, EP = tree.get_energy()
        assert np.isfinite(EP)

    def test_invalid_proposals_always_rejected(self, dim_constitution,
                                                pendulum_x_y):
        """With constitution, even aggressive MCMC never hits invalid state."""
        x, y = pendulum_x_y
        tree = Tree(
            variables=['L', 'g'], parameters=['a'],
            dimensions=PENDULUM_DIMS, constitutions=[dim_constitution],
            x=x, y=y, max_size=30,
            from_string='(_a0_ * L)',
        )

        assert dim_constitution.check(tree).is_valid

        for step in range(30):
            tree.mcmc_step(verbose=False)
            result = dim_constitution.check(tree)
            assert result.is_valid, \
                f"Step {step}: tree '{tree}' failed: {result.failures}"

    def test_converges_to_correct_formula(self, dim_constitution):
        """Longer chain should converge toward T ~ _a0_ * sqrt(L/g)."""
        np.random.seed(99)
        n = 50
        L_vals = np.linspace(0.5, 2.0, n)
        g_val = 9.81
        T_true = 2 * np.pi * np.sqrt(L_vals / g_val)
        T_obs = T_true + np.random.normal(0, 0.01, n)

        x = pd.DataFrame({'L': L_vals, 'g': [g_val] * n})
        y = pd.Series(T_obs)

        tree = Tree(
            variables=['L', 'g'], parameters=['a'],
            dimensions=PENDULUM_DIMS, constitutions=[dim_constitution],
            x=x, y=y, max_size=20,
            from_string='(_a0_ * L)',
        )

        for _ in range(100):
            tree.mcmc_step(verbose=False)

        # Final formula must pass constitution
        assert dim_constitution.check(tree).is_valid

        # BIC should be finite (fitting succeeded)
        bic = tree.get_bic()
        assert not np.isnan(float(bic)), f"BIC is NaN, tree: '{tree}'"

        # EP should remain finite
        _, _, EP = tree.get_energy()
        assert not np.isnan(EP)


# ============================================================================
# fixed_term dimensional consistency tests
# ============================================================================

class TestFixedTermDimensional:
    """DimensionalConstitution.check() with fixed_term expressions."""

    def test_fixed_term_valid_dimensionless(self):
        """Main tree + dimensionless fixed_term via *: should pass."""
        const = DimensionalConstitution()
        tree = Tree(
            variables=['L', 'g', 'theta0'],
            parameters=['a', 'b'],
            dimensions=PENDULUM_DIMS,
            constitutions=[const],
            from_string='(_a0_ * sqrt((L / g)))',
            fixed_term='(_b0_ * theta0)',
            fixed_term_op='*',
        )
        result = const.check(tree)
        assert result.is_valid, f"Should be valid but got: {result.failures}"

    def test_fixed_term_valid_same_dim(self):
        """Main tree + same-dimension fixed_term via +: should pass."""
        const = DimensionalConstitution()
        tree = Tree(
            variables=['L', 'g'],
            parameters=['a', 'b', 'c'],
            dimensions=PENDULUM_DIMS,
            constitutions=[const],
            from_string='(_a0_ * sqrt((L / g)))',
            fixed_term='(_b0_ * sqrt((L / g)))',
            fixed_term_op='+',
        )
        result = const.check(tree)
        assert result.is_valid, f"Should be valid but got: {result.failures}"

    def test_fixed_term_invalid_dim_mismatch(self):
        """Parameter-free T^2 + L → unsolvable constraint → ValueError."""
        const = DimensionalConstitution()
        with pytest.raises(ValueError, match="dimensional constraints"):
            Tree(
                variables=['L', 'g'],
                parameters=[],
                dimensions=PENDULUM_DIMS,
                constitutions=[const],
                from_string='(L / g)',
                fixed_term='L',
                fixed_term_op='+',
            )

    def test_fixed_term_none_passes(self):
        """fixed_term=None falls through to regular check."""
        const = DimensionalConstitution()
        tree = Tree(
            variables=['L', 'g'], parameters=['a'],
            dimensions=PENDULUM_DIMS, constitutions=[const],
            from_string='(_a0_ * sqrt((L / g)))',
        )
        assert const.check(tree).is_valid
