"""Tests for MCMC steps, predict, and energy consistency."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from core import Tree, OPS

PASS, FAIL = 0, 0


def t(name):
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


# ---------------------------------------------------------------------------
# 6. MCMC step on empty data (no crash)
# ---------------------------------------------------------------------------

@t("5 MCMC steps on no-data tree")
def test_mcmc_step_no_data():
    t = Tree(
        prior_par=dict([('Nopi_%s' % op, 0) for op in OPS]),
        from_string='(x + _a0_)'
    )
    for _ in range(5):
        t.mcmc_step()


# ---------------------------------------------------------------------------
# 7. Tree with data: build and run a few MCMC steps
# ---------------------------------------------------------------------------

@t("MCMC steps with data")
def test_mcmc_step_with_data():
    np.random.seed(42)
    x = pd.DataFrame({'x0': np.linspace(0, 10, 20)})
    y = 2 * x['x0'] + 1 + np.random.normal(0, 0.1, 20)

    t = Tree(
        variables=['x0'],
        parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
        max_size=10,
    )
    for _ in range(10):
        t.mcmc_step(verbose=False)
    # Should not crash and tree should have evolved
    assert t.size > 0


# ---------------------------------------------------------------------------
# 8. Predict
# ---------------------------------------------------------------------------

@t("predict() returns pd.Series")
def test_predict():
    np.random.seed(42)
    x = pd.DataFrame({'x0': np.linspace(0, 10, 20)})
    y = 2 * x['x0'] + 1 + np.random.normal(0, 0.1, 20)

    t = Tree(
        variables=['x0'],
        parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
        max_size=10,
    )
    for _ in range(10):
        t.mcmc_step(verbose=False)

    ypred = t.predict(x)
    assert isinstance(ypred, pd.Series)


# ---------------------------------------------------------------------------
# 9. Energy consistency check
# ---------------------------------------------------------------------------

@t("Energy consistency after reset")
def test_energy_consistency():
    np.random.seed(42)
    x = pd.DataFrame({'x0': np.linspace(0, 10, 20)})
    y = 2 * x['x0'] + 1 + np.random.normal(0, 0.1, 20)

    t = Tree(
        variables=['x0'],
        parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
        max_size=10,
    )
    for _ in range(10):
        t.mcmc_step(verbose=False)

    E_new, EB_new, EP_new = t.get_energy(bic=True, reset=True)
    assert abs(E_new - t.E) < 1e-6


# ============================================================================
# run
# ============================================================================

if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()

    print(f"\n{'='*60}")
    print(f"Passed: {PASS}, Failed: {FAIL}")
    print(f"{'='*60}")

    if FAIL > 0:
        sys.exit(1)
