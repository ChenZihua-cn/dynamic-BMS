"""Constitution gate rejection rate observation (Phase 3 decision point).

Decision rule (per plan_origin.md):
  < 50% → keep 5-A (hard rejection)
  50-80% → keep, record data
  > 80% → upgrade to 5-D (dimension-aware proposal)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

from experts.constitution.dimensional import DimensionalConstitution
from experts.parliament.occam import OccamPrior
from core import Tree

DIM_L = (1, 0, 0, 0, 0, 0, 0)
DIM_G = (1, 0, -2, 0, 0, 0, 0)
DIM_D = (0, 0, 0, 0, 0, 0, 0)
PENDULUM_DIMS = {'L': DIM_L, 'g': DIM_G, 'm': (0, 1, 0, 0, 0, 0, 0),
                 'theta0': DIM_D}


# ---------------------------------------------------------------------------
# Instrumentation helper
# ---------------------------------------------------------------------------

def _instrument_for_rejection_tracking(tree):
    """Wrap tree's dE_* and check_constitution to capture proposal outcomes.

    Returns *(proposal_log, cleanup_fn)*. Each entry in *proposal_log* is
    ``{'method': str, 'dE': float, 'constitution_failed': bool}``.
    """
    proposal_log = []
    orig_dE_et = tree.dE_et
    orig_dE_lr = tree.dE_lr
    orig_dE_rr = tree.dE_rr
    orig_check = tree.check_constitution

    def _make_wrapper(orig_fn, name):
        def wrapper(*args, **kwargs):
            check_calls = []

            def tracking_check():
                res = orig_check()
                check_calls.append(res)
                return res

            tree.check_constitution = tracking_check
            try:
                result = orig_fn(*args, **kwargs)
            finally:
                tree.check_constitution = orig_check

            dE_val = result[0] if isinstance(result, tuple) else result
            proposal_log.append({
                'method': name,
                'dE': float(dE_val),
                'constitution_failed': any(not c for c in check_calls),
            })
            return result
        return wrapper

    tree.dE_et = _make_wrapper(orig_dE_et, 'dE_et')
    tree.dE_lr = _make_wrapper(orig_dE_lr, 'dE_lr')
    tree.dE_rr = _make_wrapper(orig_dE_rr, 'dE_rr')

    def cleanup():
        tree.dE_et = orig_dE_et
        tree.dE_lr = orig_dE_lr
        tree.dE_rr = orig_dE_rr

    return proposal_log, cleanup


def _classify_steps(tree, proposal_log, n_steps):
    """Run *n_steps* of MCMC and classify each proposal outcome.

    Returns dict with keys: total, accepted, constitution_reject,
    canonical_reject.
    """
    stats = {'total': 0, 'accepted': 0,
             'constitution_reject': 0, 'canonical_reject': 0}

    for _ in range(n_steps):
        E_before = tree.E
        tree.mcmc_step(verbose=False)
        E_after = tree.E

        entry = proposal_log.pop(0)
        stats['total'] += 1
        if E_after != E_before:
            stats['accepted'] += 1
        elif entry['constitution_failed']:
            stats['constitution_reject'] += 1
        elif np.isinf(entry['dE']):
            stats['canonical_reject'] += 1

    return stats


def _print_rejection_report(stats, n_steps):
    """Pretty-print the rejection rate statistics."""
    total = stats['total']
    fmt = lambda n: f"{n:>6d}  ({n / total:.1%})"
    const_rate = stats['constitution_reject'] / total

    print(f"\n{'='*60}")
    print(f"Constitution Gate Rejection Rate Report ({n_steps} steps)")
    print(f"{'='*60}")
    print(f"  Total proposals:       {stats['total']:>6d}")
    print(f"  Accepted:              {fmt(stats['accepted'])}")
    print(f"  Constitution-rejected: {fmt(stats['constitution_reject'])}")
    print(f"  Canonical-rejected:    {fmt(stats['canonical_reject'])}")
    other = total - stats['accepted'] - stats['constitution_reject'] - stats['canonical_reject']
    print(f"  Other (MCMC prob):     {other:>6d}")
    print(f"{'='*60}")
    if const_rate < 0.50:
        decision = "keep 5-A (hard rejection)"
    elif const_rate < 0.80:
        decision = "keep 5-A, record data"
    else:
        decision = "UPGRADE to 5-D (dimension-aware proposal)"
    print(f"  Decision: {decision}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dim_constitution():
    return DimensionalConstitution()


@pytest.fixture
def pendulum_x_y():
    np.random.seed(42)
    n = 30
    L_vals = np.linspace(0.5, 2.0, n)
    g_val = 9.81
    T_true = 2 * np.pi * np.sqrt(L_vals / g_val)
    T_obs = T_true + np.random.normal(0, 0.02, n)
    return pd.DataFrame({'L': L_vals, 'g': [g_val] * n}), pd.Series(T_obs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConstitutionRejectionRate:
    """Phase 3 efficiency observation."""

    def test_rejection_rate_statistics(self, dim_constitution, pendulum_x_y):
        x, y = pendulum_x_y
        tree = Tree(
            variables=['L', 'g'], parameters=['a', 'b'],
            dimensions=PENDULUM_DIMS,
            constitutions=[dim_constitution],
            x=x, y=y, max_size=30,
            from_string='(_a0_ * sqrt((L / g)))',
        )
        assert dim_constitution.check(tree).is_valid

        proposal_log, cleanup = _instrument_for_rejection_tracking(tree)
        n_steps = 200
        stats = _classify_steps(tree, proposal_log, n_steps)
        cleanup()

        _print_rejection_report(stats, n_steps)

        const_rate = stats['constitution_reject'] / stats['total']
        assert const_rate < 0.80, \
            f"Constitution rejection rate {const_rate:.1%} >= 80% — " \
            f"hard rejection is blocking exploration."

    def test_rejection_rate_with_parliament(self, dim_constitution,
                                             pendulum_x_y):
        """Rejection rate is unaffected by parliament (constitution runs first)."""
        x, y = pendulum_x_y
        tree = Tree(
            variables=['L', 'g'], parameters=['a', 'b'],
            dimensions=PENDULUM_DIMS,
            constitutions=[dim_constitution],
            parliaments=[OccamPrior()],
            x=x, y=y, max_size=30,
            from_string='(_a0_ * sqrt((L / g)))',
        )
        assert dim_constitution.check(tree).is_valid

        proposal_log, cleanup = _instrument_for_rejection_tracking(tree)
        stats = _classify_steps(tree, proposal_log, n_steps=100)
        cleanup()

        _print_rejection_report(stats, 100)

        const_rate = stats['constitution_reject'] / stats['total']
        assert const_rate < 0.80
