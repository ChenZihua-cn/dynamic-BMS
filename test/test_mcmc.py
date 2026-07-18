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

@t("MCMC step without data maintains tree invariants")
def test_mcmc_step_no_data():
    np.random.seed(42)
    t = Tree(
        prior_par=dict([('Nopi_%s' % op, 0) for op in OPS]),
        from_string='(x + _a0_)'
    )
    initial_str = str(t)

    for i in range(5):
        t.mcmc_step(verbose=False)
        assert 1 <= t.size <= t.max_size, \
            f"Step {i}: tree size {t.size} out of [1, {t.max_size}]"
        assert np.isfinite(float(t.E)), \
            f"Step {i}: energy is not finite: {t.E}"
        assert np.isfinite(float(t.bic)), \
            f"Step {i}: BIC is not finite: {t.bic}"

    # After 5 steps the tree should have changed
    final_str = str(t)
    assert final_str != initial_str, \
        f"Tree did not change after 5 MCMC steps: expr={final_str}"


# ---------------------------------------------------------------------------
# 7. Tree with data: build and run a few MCMC steps
# ---------------------------------------------------------------------------

@t("MCMC steps with data: energy and BIC remain finite, tree evolves")
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
    initial_E = t.E
    initial_str = str(t)

    for i in range(10):
        t.mcmc_step(verbose=False)
        assert 1 <= t.size <= t.max_size, \
            f"Step {i}: tree size {t.size} out of [1, {t.max_size}]"
        assert np.isfinite(float(t.E)), \
            f"Step {i}: energy is not finite: {t.E}"
        assert np.isfinite(float(t.bic)), \
            f"Step {i}: BIC is not finite: {t.bic}"

    # MCMC should have explored: either the expression changed
    final_str = str(t)
    assert final_str != initial_str, \
        f"Tree did not evolve after 10 MCMC steps with data"
    # Energy should not increase substantially on simple linear data
    assert t.E <= initial_E + 1.0, \
        f"Energy increased substantially: initial={initial_E:.4f}, final={t.E:.4f}"


# ---------------------------------------------------------------------------
# 8. Predict
# ---------------------------------------------------------------------------

@t("predict() returns finite values with correct shape")
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

    # Predict on training data
    ypred = t.predict(x)
    assert isinstance(ypred, pd.Series)
    assert len(ypred) == len(x), \
        f"Prediction length {len(ypred)} != input length {len(x)}"
    assert np.all(np.isfinite(ypred.values)), \
        "Predictions contain non-finite values"
    assert ypred.nunique() > 1, \
        "Predictions are constant -- model is trivial or broken"

    # Predict on test data with different length
    x_test = pd.DataFrame({'x0': np.linspace(0, 10, 5)})
    ypred_test = t.predict(x_test)
    assert len(ypred_test) == 5
    assert np.all(np.isfinite(ypred_test.values))


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
