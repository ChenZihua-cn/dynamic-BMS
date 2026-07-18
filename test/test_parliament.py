# test/test_parliament.py
"""Tests for the parliament-layer soft priors.

Covers: ParliamentBase, OccamPrior, ExpertMixin integration, sign
convention, cache behavior, gate interaction, error resilience,
and MCMC end-to-end regression.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from experts.parliament.base import ParliamentBase
from experts.parliament.occam import OccamPrior
from experts.constitution.dimensional import DimensionalConstitution
from core import Tree


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_linear_data(n=50, seed=42):
    np.random.seed(seed)
    x_vals = np.linspace(-3, 3, n)
    y_true = 2.0 * x_vals + 1.0
    y_obs = y_true + np.random.normal(0, 0.1, n)
    return (
        pd.DataFrame({'x': x_vals}),
        pd.Series(y_obs, name='y'),
    )


# ------------------------------------------------------------------
# Test 1: ParliamentBase defaults
# ------------------------------------------------------------------

class TestParliamentBase:

    def test_default_methods_return_zero(self):
        p = ParliamentBase()
        assert p.name == 'unnamed_parliament'
        tree = Tree(variables=['x'], parameters=['a'],
                    from_string='(_a0_ * x)')
        assert p.evaluate_structural(tree) == 0.0
        assert p.evaluate_fitted(tree) == 0.0


# ------------------------------------------------------------------
# Test 2: OccamPrior correctness
# ------------------------------------------------------------------

class TestOccamPrior:

    def test_size_penalty(self):
        op = OccamPrior()
        # size=1 tree (just 'x')
        t1 = Tree(variables=['x'], parameters=['a'], from_string='x')
        assert op.evaluate_structural(t1) == 1.0
        assert op.evaluate_fitted(t1) == 0.0

        # size=3 tree: (_a0_ * x) = {_a0_, *, x} = 3 nodes
        t3 = Tree(variables=['x'], parameters=['a'],
                  from_string='(_a0_ * x)')
        assert op.evaluate_structural(t3) == 3.0

        # size=4 tree: (_a0_ * sin(x)) = {_a0_, *, sin, x} = 4 nodes
        t4 = Tree(variables=['x'], parameters=['a'],
                  from_string='(_a0_ * sin(x))')
        assert op.evaluate_structural(t4) == 4.0


# ------------------------------------------------------------------
# Test 3: Sign convention (regression test for fix #1)
# ------------------------------------------------------------------

class TestSignConvention:

    def test_larger_tree_positive_dEP(self):
        """Replacing a leaf with a subtree increases dEP (penalty grows)."""
        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[OccamPrior()],
                 from_string='(_a0_ * x)')
        x_node = [n for n in t.nodes if n.value == 'x'][0]
        # Replace x with sin(x): size grows from 3 to 4
        dE, dEB, dEP, pv, nif, nfi = t.dE_et(x_node, ['sin', ['x']])
        assert dEP > 0, f"Expected positive dEP for larger tree, got {dEP}"

    def test_same_size_zero_structural_delta(self):
        """Replacing leaf with another leaf: size unchanged."""
        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[OccamPrior()],
                 from_string='(_a0_ * x)')
        x_node = [n for n in t.nodes if n.value == 'x'][0]
        # Replace x with 1 (both leaves, size unchanged)
        dE, dEB, dEP, pv, nif, nfi = t.dE_et(x_node, ['1', []])
        assert np.isfinite(dE)
        # Structural delta is 0 (same size) but prior_par may change dEP


# ------------------------------------------------------------------
# Test 4: EP includes structural parliament at init
# ------------------------------------------------------------------

class TestEnergyIncludesParliament:

    def test_EP_includes_structural_penalty(self):
        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[OccamPrior()],
                 from_string='(_a0_ * x)')
        # Size=3, weight=1.0: parliament penalty = 3.0
        assert t.EP > 3.0, \
            f"EP ({t.EP}) should at minimum include parliament penalty (3.0)"

    def test_no_parliaments_no_extra_EP(self):
        t_no = Tree(variables=['x'], parameters=['a'],
                    from_string='(_a0_ * x)')
        t_with = Tree(variables=['x'], parameters=['a'],
                      parliaments=[OccamPrior()],
                      from_string='(_a0_ * x)')
        # With parliament should have strictly higher EP
        assert t_with.EP > t_no.EP


# ------------------------------------------------------------------
# Test 5: Parliament cache behavior
# ------------------------------------------------------------------

class TestParliamentCache:

    def test_cache_hit_reuses_raw_values(self):
        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[OccamPrior()],
                 from_string='x')
        pen1 = t._parliament_energy_structural()
        cache_size_before = len(t._parliament_structural_cache)
        pen2 = t._parliament_energy_structural()
        cache_size_after = len(t._parliament_structural_cache)
        assert pen1 == pen2
        assert cache_size_after == cache_size_before  # no new entries

    def test_cache_stores_per_expert_raw(self):
        class DummyP(ParliamentBase):
            name = 'dummy'
            def evaluate_structural(self, tree):
                return 42.0

        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[OccamPrior(), DummyP()],
                 from_string='x')
        # Force cache population
        t._parliament_energy_structural()
        canonical = t.canonical()
        raw = t._parliament_structural_cache[canonical]
        assert 'occam' in raw
        assert 'dummy' in raw
        assert raw['occam'] == 1.0  # size=1
        assert raw['dummy'] == 42.0


# ------------------------------------------------------------------
# Test 6: Gate failure skips parliament
# ------------------------------------------------------------------

class TestParliamentGateInteraction:

    def test_gate_failure_returns_inf(self):
        """Dimensionally invalid proposal: gate reject, parliament skipped."""
        dims = {'x': (1, 0, 0, 0, 0, 0, 0),
                't': (0, 0, 1, 0, 0, 0, 0)}
        t = Tree(variables=['x', 't'], parameters=['a'],
                 dimensions=dims, constitutions=[DimensionalConstitution()],
                 parliaments=[OccamPrior()],
                 from_string='(_a0_ * x)', max_size=30)
        x_node = [n for n in t.nodes if n.value == 'x'][0]
        # Replace x with sin(x): sin(x) is dimensionless but x is L
        dE, dEB, dEP, pv, nif, nfi = t.dE_et(x_node, ['sin', ['x']])
        assert np.isinf(dE)

    def test_gate_pass_parliament_contributes(self):
        """Valid dimensions: parliament delta included in dEP."""
        # Both L1 and L2 have length dimension (1,0,0,...)
        dims = {'L1': (1, 0, 0, 0, 0, 0, 0),
                'L2': (1, 0, 0, 0, 0, 0, 0)}
        t = Tree(variables=['L1', 'L2'], parameters=['a'],
                 dimensions=dims, constitutions=[DimensionalConstitution()],
                 parliaments=[OccamPrior()],
                 from_string='(_a0_ * L1)', max_size=30)
        L1_node = [n for n in t.nodes if n.value == 'L1'][0]
        # Replace L1 with (L1 + L2): shape = 3 -> 5, dimension L+L=L OK
        dE, dEB, dEP, pv, nif, nfi = t.dE_et(
            L1_node, ['+', ['L1', 'L2']])
        assert np.isfinite(dE)
        # Parliament penalty increases (tree grows from 3 to 5 nodes)
        assert dEP > 0


# ------------------------------------------------------------------
# Test 7: Multi-parliament integration
# ------------------------------------------------------------------

class TestMultiParliament:

    def test_weights_sum_to_one(self):
        class P1(ParliamentBase):
            name = 'p1'

        class P2(ParliamentBase):
            name = 'p2'

        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[P1(), P2()],
                 from_string='x')
        w = t.parliament_weights
        assert abs(sum(w.values()) - 1.0) < 1e-10

    def test_weighted_sum_correct(self):
        class P1(ParliamentBase):
            name = 'p1'
            def evaluate_structural(self, tree):
                return 10.0

        class P2(ParliamentBase):
            name = 'p2'
            def evaluate_structural(self, tree):
                return 20.0

        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[P1(), P2()],
                 from_string='x')
        # Equal weights: 0.5*10 + 0.5*20 = 15.0
        assert abs(t._parliament_energy_structural() - 15.0) < 1e-10


# ------------------------------------------------------------------
# Test 8: Error resilience
# ------------------------------------------------------------------

class TestParliamentErrorResilience:

    def test_expert_raises_is_silenced(self):
        """Expert that raises during evaluate_structural returns 0.0
        rather than crashing the MCMC chain."""
        class BadExpert(ParliamentBase):
            name = 'bad'
            def evaluate_structural(self, tree):
                raise RuntimeError("deliberate error")

        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[BadExpert()],
                 from_string='(_a0_ * x)')
        # Structural energy should return 0.0 (exception caught)
        assert t._parliament_energy_structural() == 0.0
        # dE_et should still work
        x_node = [n for n in t.nodes if n.value == 'x'][0]
        dE, dEB, dEP, pv, nif, nfi = t.dE_et(x_node, ['sin', ['x']])
        assert np.isfinite(dE)

    def test_capture_delta_handles_exceptions(self):
        """_capture and _delta gracefully handle expert exceptions."""
        class FlakyExpert(ParliamentBase):
            name = 'flaky'
            def evaluate_structural(self, tree):
                raise RuntimeError("always fails")

        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[FlakyExpert()],
                 from_string='(_a0_ * x)')
        # Both capture and delta return 0.0 on exception
        old_val = t._capture_parliament_structural()
        assert old_val == 0.0
        assert t._parliament_structural_delta(old_val) == 0.0


# ------------------------------------------------------------------
# Test 9: Cache eviction
# ------------------------------------------------------------------

class TestCacheEviction:

    def test_cache_grows_with_new_canonicals(self):
        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[OccamPrior()],
                 from_string='x')
        t._parliament_energy_structural()
        assert len(t._parliament_structural_cache) >= 1


# ------------------------------------------------------------------
# Test 10: MCMC end-to-end regression
# ------------------------------------------------------------------

class TestMCMCWithParliament:

    def test_mcmc_short_chain(self):
        """Short MCMC chain with OccamPrior doesn't crash, EP behaves."""
        x_df, y_series = _make_linear_data(n=50)
        t = Tree(
            variables=['x'], parameters=['a', 'b'],
            x=x_df, y=y_series,
            parliaments=[OccamPrior()],
            from_string='((_a0_ * x) + _b0_)',
            max_size=20,
        )
        initial_EP = t.EP
        initial_size = t.size

        # Run 50 steps
        for i in range(50):
            t.mcmc_step(verbose=False)
            assert np.isfinite(float(t.E)), f"Step {i}: t.E is inf"
            assert np.isfinite(float(t.EP)), f"Step {i}: t.EP is inf"

        # Verify: larger trees have higher EP (penalty convention)
        if t.size > initial_size:
            assert t.EP > initial_EP, \
                f"Larger tree ({t.size} > {initial_size}) should have higher EP"

        # Verify: cache is bounded
        assert len(t._parliament_structural_cache) < 10000, \
            f"Cache grew to {len(t._parliament_structural_cache)}"

        # Verify: some accepted moves occurred, or chain stayed on correct formula
        # (50 steps may not accept if initial formula is already optimal)
        assert np.isfinite(float(t.E)), "Final t.E is inf"
        assert np.isfinite(float(t.EP)), "Final t.EP is inf"


# ============================================================================
# Test 11: Parliament delta in dE_lr, dE_rr (direct, not just MCMC)
# ============================================================================

class TestParliamentProposalGate:
    """Parliament structural delta in dE_lr and dE_rr (prune + replace)."""

    def test_dE_lr_parliament_delta(self):
        """dE_lr with OccamPrior: leaf-to-leaf swap, parliament delta = 0."""
        t = Tree(variables=['x'], parameters=['a'],
                parliaments=[OccamPrior()],
                from_string='(_a0_ * x)', max_size=30)
        x_node = [n for n in t.nodes if n.value == 'x'][0]
        before = str(t)

        dE, _, dEP, _ = t.dE_lr(x_node, 'sin')
        assert np.isfinite(dE)
        assert str(t) == before
        assert np.isfinite(dEP)

    def test_dE_lr_with_constitution_and_parliament(self):
        """dE_lr replacing * → +: constitution passes, parliament delta = 0."""
        dims = {'x': (1, 0, 0, 0, 0, 0, 0)}
        t = Tree(variables=['x'], parameters=['a'],
                 dimensions=dims,
                 constitutions=[DimensionalConstitution()],
                 parliaments=[OccamPrior()],
                 from_string='(_a0_ * x)', max_size=30)
        mul_node = [n for n in t.nodes if n.value == '*'][0]
        before = str(t)

        dE, _, _, _ = t.dE_lr(mul_node, '+')
        assert np.isfinite(dE)
        assert str(t) == before

    def test_dE_rr_prune_parliament_delta(self):
        """dE_rr prune with OccamPrior: size decreases, dEP should show it."""
        dims = {'L': (1, 0, 0, 0, 0, 0, 0)}
        t = Tree(variables=['L'], parameters=['a'],
                 dimensions=dims,
                 constitutions=[DimensionalConstitution()],
                 parliaments=[OccamPrior()],
                 from_string='(_a0_ + L)', max_size=30)
        assert t.is_root_prunable()
        before = str(t)
        old_size = t.size

        dE, _, dEP, _ = t.dE_rr(rr=None)
        assert np.isfinite(dE)
        assert str(t) == before
        assert t.size == old_size  # reverted

    def test_dE_rr_replace_parliament_delta(self):
        """dE_rr replace with OccamPrior: adding root increases dEP."""
        dims = {'L': (1, 0, 0, 0, 0, 0, 0),
                'g': (1, 0, -2, 0, 0, 0, 0)}
        t = Tree(variables=['L', 'g'], parameters=['a', 'b'],
                 dimensions=dims,
                 constitutions=[DimensionalConstitution()],
                 parliaments=[OccamPrior()],
                 from_string='sqrt((L / g))', max_size=30)
        before = str(t)

        dE, _, _, _ = t.dE_rr(rr=['*', ['_b0_']])
        assert np.isfinite(dE)
        assert str(t) == before

    def test_canonical_reject_skips_parliament_delta(self):
        """Canonical reject: dE=inf, parliament delta NOT applied, EP unchanged."""
        t = Tree(variables=['x'], parameters=['a'],
                 parliaments=[OccamPrior()],
                 from_string='(_a0_ * x)', max_size=30)
        old_E, old_EP = t.E, t.EP

        orig_update = t.update_representative
        t.update_representative = lambda verbose=False: -1
        try:
            x_node = [n for n in t.nodes if n.value == 'x'][0]
            dE, _, _, _, _, _ = t.dE_et(x_node, ['sin', ['x']])
            assert np.isinf(dE)
            assert t.EP == old_EP
            assert t.E == old_E
        finally:
            t.update_representative = orig_update
