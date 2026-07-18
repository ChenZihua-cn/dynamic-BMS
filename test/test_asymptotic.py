# test/test_asymptotic.py
"""Tests for AsymptoticPrior parliament expert.

Covers: unit correctness, ideal-gas physics case, multi-expert
weighted combination, and MCMC end-to-end regression.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from experts.parliament.base import ParliamentBase
from experts.parliament.occam import OccamPrior
from experts.parliament.asymptotic import AsymptoticPrior
from core import Tree


# ------------------------------------------------------------------
# Test 1: AsymptoticPrior unit tests
# ------------------------------------------------------------------

class TestAsymptoticPriorUnit:

    def test_no_division_nodes(self):
        """Expression without '/' returns 0 penalty."""
        t = Tree(variables=['x'], parameters=['a'],
                 from_string='(_a0_ * x)')
        expert = AsymptoticPrior(finite_at_zero=['x'])
        assert expert.evaluate_structural(t) == 0.0

    def test_division_by_variable_in_finite_set(self):
        """Variable in finite_at_zero appears in denominator -> +0.5."""
        t = Tree(variables=['x', 'y'], parameters=['a'],
                 from_string='(x / y)')
        expert = AsymptoticPrior(finite_at_zero=['y'])
        assert expert.evaluate_structural(t) == 0.5

    def test_division_by_parameter_no_penalty(self):
        """Parameter in denominator (not in finite_at_zero) -> no penalty."""
        t = Tree(variables=['x'], parameters=['a'],
                 from_string='(x / _a0_)')
        expert = AsymptoticPrior(finite_at_zero=['x'])
        assert expert.evaluate_structural(t) == 0.0

    def test_multiple_violations_stack(self):
        """Two '/' nodes both dividing by target variable -> 1.0."""
        t = Tree(variables=['x', 'y', 'z'], parameters=['a'],
                 from_string='((x / y) + (z / y))')
        expert = AsymptoticPrior(finite_at_zero=['y'])
        assert expert.evaluate_structural(t) == 1.0

    def test_variable_not_in_finite_set(self):
        """Variable not declared in finite_at_zero -> no penalty."""
        t = Tree(variables=['x', 'y'], parameters=['a'],
                 from_string='(x / y)')
        expert = AsymptoticPrior(finite_at_zero=['x'])
        assert expert.evaluate_structural(t) == 0.0

    def test_nested_denominator(self):
        """Denominator is a compound expression containing target variable."""
        t = Tree(variables=['x', 'y'], parameters=['a'],
                 from_string='(x / (_a0_ + y))')
        expert = AsymptoticPrior(finite_at_zero=['y'])
        assert expert.evaluate_structural(t) == 0.5

    def test_deeply_nested_denominator(self):
        """Variable buried deep in denominator expression."""
        t = Tree(variables=['x', 'y'], parameters=['a', 'b'],
                 from_string='(x / (_a0_ * (_b0_ + y)))')
        expert = AsymptoticPrior(finite_at_zero=['y'])
        assert expert.evaluate_structural(t) == 0.5

    def test_empty_finite_at_zero_none(self):
        """finite_at_zero=None -> no opinion, always 0."""
        t = Tree(variables=['x', 'y'], parameters=['a'],
                 from_string='(x / y)')
        expert = AsymptoticPrior(finite_at_zero=None)
        assert expert.evaluate_structural(t) == 0.0

    def test_empty_finite_at_zero_list(self):
        """finite_at_zero=[] -> no opinion, always 0."""
        t = Tree(variables=['x', 'y'], parameters=['a'],
                 from_string='(x / y)')
        expert = AsymptoticPrior(finite_at_zero=[])
        assert expert.evaluate_structural(t) == 0.0


# ------------------------------------------------------------------
# Test 2: Ideal gas physics case (P -> 0, V -> infinity)
# ------------------------------------------------------------------

class TestAsymptoticPriorIdealGas:

    def test_division_by_P_penalized(self):
        """Ideal gas: V = _a0_ / P. As P->0, V diverges.
        finite_at_zero=['P'] correctly penalizes P in denominator."""
        t = Tree(variables=['P'], parameters=['a'],
                 from_string='(_a0_ / P)')
        expert = AsymptoticPrior(finite_at_zero=['P'])
        assert expert.evaluate_structural(t) == 0.5

    def test_no_division_no_penalty(self):
        """Expression _a0_ * P has no '/' -> penalty 0.
        Physically: as P->0, expression -> 0 (finite, OK)."""
        t = Tree(variables=['P'], parameters=['a'],
                 from_string='(_a0_ * P)')
        expert = AsymptoticPrior(finite_at_zero=['P'])
        assert expert.evaluate_structural(t) == 0.0

    def test_other_variable_in_denom_no_penalty(self):
        """Expression (P / V): P not in denominator -> 0.
        V in denominator but V not in finite_at_zero -> no penalty."""
        t = Tree(variables=['P', 'V'], parameters=['a'],
                 from_string='(P / V)')
        expert = AsymptoticPrior(finite_at_zero=['P'])
        assert expert.evaluate_structural(t) == 0.0

    def test_P_in_numerator_no_penalty(self):
        """P appears only in numerator (P / _a0_) -> no penalty.
        The denominator _a0_ is a parameter, not in finite_at_zero."""
        t = Tree(variables=['P'], parameters=['a'],
                 from_string='(P / _a0_)')
        expert = AsymptoticPrior(finite_at_zero=['P'])
        assert expert.evaluate_structural(t) == 0.0

    def test_mcmc_ideal_gas_both_experts(self):
        """MCMC on ideal-gas data with OccamPrior + AsymptoticPrior.
        Boyle's law: V = const / P at fixed T.
        The AsymptoticPrior(finite_at_zero=['P']) penalizes P in denominator
        -- a mildly misaligned soft prior. The data likelihood should
        still guide the chain to the correct form."""
        np.random.seed(42)
        # Generate ideal gas data: V = const / P
        const_true = 2.0
        P_vals = np.linspace(0.5, 3.0, 40)
        V_true = const_true / P_vals
        V_obs = V_true + np.random.normal(0, 0.05, 40)

        x_df = pd.DataFrame({'P': P_vals})
        y_series = pd.Series(V_obs, name='V')

        t = Tree(
            variables=['P'], parameters=['a', 'b'],
            x=x_df, y=y_series,
            parliaments=[
                OccamPrior(),
                AsymptoticPrior(finite_at_zero=['P']),
            ],
            from_string='((_a0_ / P) + _b0_)',
            max_size=20,
        )

        # Verify both experts are registered
        assert set(t.parliament_weights.keys()) == {'occam', 'asymptotic'}

        # Weights sum to 1.0
        w = t.parliament_weights
        assert abs(sum(w.values()) - 1.0) < 1e-10

        # Structural energy = weighted sum of both experts' raw penalties
        structural = t._parliament_energy_structural()
        # OccamPrior: size of ((_a0_ / P) + _b0_) = 5 nodes
        # AsymptoticPrior: P in denominator -> 0.5
        # Expected: 0.5 * 5 + 0.5 * 0.5 = 2.5 + 0.25 = 2.75
        expected = 0.5 * 5 + 0.5 * 0.5
        assert abs(structural - expected) < 1e-10

        # Run 50 MCMC steps
        for i in range(50):
            t.mcmc_step(verbose=False)
            assert np.isfinite(float(t.E)), f"Step {i}: t.E is inf"
            assert np.isfinite(float(t.EP)), f"Step {i}: t.EP is inf"

        # Cache bounded
        assert len(t._parliament_structural_cache) < 10000

        # Final state is finite
        assert np.isfinite(float(t.E))
        assert np.isfinite(float(t.EP))


# ------------------------------------------------------------------
# Test 3: Multi-expert weighted combination (KEY TEST)
# ------------------------------------------------------------------

class TestAsymptoticPriorIntegration:

    def test_weights_equal_with_two_experts(self):
        """Two experts with gamma=0 -> equal 0.5 weights."""
        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[OccamPrior(),
                              AsymptoticPrior(finite_at_zero=['x'])],
                 from_string='x')
        w = t.parliament_weights
        assert abs(w['occam'] - 0.5) < 1e-10
        assert abs(w['asymptotic'] - 0.5) < 1e-10
        assert abs(sum(w.values()) - 1.0) < 1e-10

    def test_weighted_energy_structural(self):
        """Verify weighted sum: 0.5 * size + 0.5 * asymptotic_penalty."""
        t = Tree(variables=['x', 'y'], parameters=['a'],
                 parliaments=[OccamPrior(),
                              AsymptoticPrior(finite_at_zero=['y'])],
                 from_string='(x / y)')
        # OccamPrior raw: size = 3
        # AsymptoticPrior raw: y in denominator = 0.5
        # Weighted: 0.5 * 3 + 0.5 * 0.5 = 1.5 + 0.25 = 1.75
        structural = t._parliament_energy_structural()
        expected = 0.5 * 3 + 0.5 * 0.5
        assert abs(structural - expected) < 1e-10

    def test_cache_stores_per_expert_raw(self):
        """Cache stores raw values keyed by expert name."""
        t = Tree(variables=['x', 'y'], parameters=['a'],
                 parliaments=[OccamPrior(),
                              AsymptoticPrior(finite_at_zero=['y'])],
                 from_string='(x / y)')
        t._parliament_energy_structural()
        canonical = t.canonical()
        raw = t._parliament_structural_cache[canonical]
        assert raw['occam'] == 3.0
        assert raw['asymptotic'] == 0.5

    def test_dE_et_includes_both_contributions(self):
        """dE_et delta reflects both Occam and Asymptotic changes."""
        t = Tree(variables=['x', 'y'], parameters=['a'],
                 parliaments=[OccamPrior(),
                              AsymptoticPrior(finite_at_zero=['y'])],
                 from_string='(_a0_ * x)', max_size=30)
        x_node = [n for n in t.nodes if n.value == 'x'][0]
        # Replace x with (x / y): adds / node with y in denominator,
        # tree size grows from 3 to 5
        dE, dEB, dEP, pv, nif, nfi = t.dE_et(
            x_node, ['/', ['x', 'y']])
        assert np.isfinite(dE)
        # dEP > 0: both experts contribute positive penalties
        # Occam: +2 nodes (size 3->5) * 0.5 weight = +1.0
        # Asymptotic: +0.5 (new / with y in denom) * 0.5 weight = +0.25
        # Total dEP from parliament >= 1.25 (plus any prior_par changes)
        assert dEP > 0

    def test_EP_differs_with_second_expert(self):
        """Adding AsymptoticPrior changes EP vs OccamPrior alone.

        Semantic note: with softmax(gamma=0), adding more experts dilutes
        each expert's weight (Occam goes from 1.0 -> 0.5), so EP decreases
        here (3.0 -> 1.75). This is a mathematical property of equal-weight
        softmax, not a bug. In Phase 2, gamma learning via MCMC will let
        the data tune per-expert weights, so an expert that consistently
        disagrees with the data will have its gamma lowered rather than
        silently diluting all others equally."""
        t_occam = Tree(variables=['x', 'y'], parameters=['a'],
                       parliaments=[OccamPrior()],
                       from_string='(x / y)')
        t_both = Tree(variables=['x', 'y'], parameters=['a'],
                      parliaments=[OccamPrior(),
                                   AsymptoticPrior(finite_at_zero=['y'])],
                      from_string='(x / y)')
        # With OccamPrior alone: EP includes 1.0 * size = 3.0
        # With both: EP includes 0.5 * 3 + 0.5 * 0.5 = 1.75
        assert t_both.EP < t_occam.EP, \
            "softmax(gamma=0) dilutes Occam weight from 1.0 to 0.5"
        assert abs(t_both._parliament_energy_structural() - 1.75) < 1e-10


# ------------------------------------------------------------------
# Test 4: MCMC end-to-end with both experts
# ------------------------------------------------------------------

class TestAsymptoticPriorMCMC:

    def test_mcmc_short_chain_both_experts(self):
        """50-step MCMC with OccamPrior + AsymptoticPrior doesn't crash."""
        np.random.seed(123)
        x_vals = np.linspace(-3, 3, 50)
        y_true = 2.0 * x_vals + 1.0
        y_obs = y_true + np.random.normal(0, 0.1, 50)

        x_df = pd.DataFrame({'x': x_vals})
        y_series = pd.Series(y_obs, name='y')

        t = Tree(
            variables=['x'], parameters=['a', 'b'],
            x=x_df, y=y_series,
            parliaments=[
                OccamPrior(),
                AsymptoticPrior(finite_at_zero=['x']),
            ],
            from_string='((_a0_ * x) + _b0_)',
            max_size=20,
        )

        initial_EP = t.EP
        initial_size = t.size

        for i in range(50):
            t.mcmc_step(verbose=False)
            assert np.isfinite(float(t.E)), f"Step {i}: t.E is inf"
            assert np.isfinite(float(t.EP)), f"Step {i}: t.EP is inf"

        # Larger tree -> higher EP (penalty convention)
        if t.size > initial_size:
            assert t.EP > initial_EP, \
                f"Larger tree ({t.size} > {initial_size}) should have higher EP"

        # Cache bounded
        assert len(t._parliament_structural_cache) < 10000

        assert np.isfinite(float(t.E))
        assert np.isfinite(float(t.EP))
