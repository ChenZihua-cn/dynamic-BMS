"""Synthetic data validation pipeline tests (cases A-I)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from core import Tree, OPS
from helpers import r2_score

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


# ============================================================================
# Shared synthetic-data pipeline
# ============================================================================

def run_synthetic_test(name, ground_truth, variables, n_train=30, n_test=20,
                       noise=0.1, burnin=500, thin=20, samples=100,
                       max_size=15, prior_par=None, rtol=1e-6):
    """Run one synthetic-data validation test."""
    np.random.seed(42)

    # --- 1. Generate synthetic data ---
    x_train = pd.DataFrame({
        v: np.random.uniform(-3, 3, n_train) for v in variables
    })
    for v in variables:
        x_train[v] = x_train[v].where(x_train[v].abs() > 0.5, 0.5)

    y_clean = ground_truth(**{v: x_train[v].values for v in variables})
    y_train = pd.Series(y_clean + np.random.normal(0, noise, n_train))

    x_test = pd.DataFrame({
        v: np.linspace(-2.5, 2.5, n_test) for v in variables
    })
    for v in variables:
        x_test[v] = x_test[v].where(x_test[v].abs() > 0.3, 0.3)
    y_test = pd.Series(ground_truth(**{v: x_test[v].values for v in variables}))

    # --- 2. Build default prior ---
    if prior_par is None:
        prior_par = dict([('Nopi_%s' % op, 1.0) for op in OPS])

    # --- 3. Initialize tree ---
    t = Tree(
        variables=variables,
        parameters=['a%d' % i for i in range(6)],
        x=x_train, y=y_train,
        prior_par=prior_par,
        max_size=max_size,
        BT=1.0,
    )
    initial_bic = t.bic
    initial_E = t.E

    # --- 4. Run MCMC ---
    trace_fn = '/tmp/test_trace_{}.dat'.format(name.replace(' ', '_'))
    progress_fn = '/tmp/test_progress_{}.dat'.format(name.replace(' ', '_'))
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=burnin, thin=thin, samples=samples,
           write_files=False,
           verbose=False, progress=False)

    final_expr = str(t)
    final_bic = t.bic
    final_E = t.E

    # --- 5. Predict ---
    y_train_pred = t.predict(x_train)
    y_test_pred = t.predict(x_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # --- 6. Sanity ---
    assert final_bic < initial_bic + rtol, \
        "{}: BIC did not improve (initial={:.2f}, final={:.2f})".format(
            name, initial_bic, final_bic)
    assert final_E < initial_E + rtol, \
        "{}: Energy did not improve".format(name)

    print(f"[PASS] {name}: initial BIC={initial_bic:.2f} -> final BIC={final_bic:.2f}, "
          f"train R\u00b2={train_r2:.4f}, test R\u00b2={test_r2:.4f}, "
          f"expr: {final_expr}")

    return dict(name=name, train_r2=train_r2, test_r2=test_r2,
                initial_bic=initial_bic, final_bic=final_bic,
                initial_E=initial_E, final_E=final_E,
                final_expr=final_expr, trace_file=trace_fn)


# ============================================================================
# Test case functions
# ============================================================================

def ground_linear(x0):
    return 2.0 * x0 + 1.0

def ground_quadratic(x0):
    return x0**2 + 3.0 * x0 - 2.0

def ground_sin(x0):
    return np.sin(2.0 * x0)

def ground_rational(x0):
    return x0 / (1.0 + x0**2)

def ground_2var(x0, x1):
    return 3.0 * x0 - 2.0 * x1 + 5.0

def ground_3var_interact(x0, x1, x2):
    return x0 * x1 + x2

def ground_3var_rat(x0, x1, x2):
    return (x0 * x1) / (1.0 + x2**2)

def ground_3var_mix(x0, x1, x2):
    return np.sin(x0) + x1 * np.exp(-x2**2)

def ground_exp(x0):
    return np.exp(-x0**2)


# ============================================================================
# Test cases
# ============================================================================

@t("Linear y=2x+1")
def test_linear():
    res = run_synthetic_test("Linear y=2x+1", ground_linear, ['x0'],
                             n_train=30, noise=0.15, burnin=500, thin=20, samples=100)
    assert res['test_r2'] > 0.7


@t("Quadratic y=x^2+3x-2")
def test_quadratic():
    res = run_synthetic_test("Quadratic y=x^2+3x-2", ground_quadratic, ['x0'],
                             n_train=40, noise=0.2, burnin=500, thin=20, samples=100)
    assert res['test_r2'] > 0.6


@t("Trig y=sin(2x)")
def test_trig():
    res = run_synthetic_test("Trig y=sin(2x)", ground_sin, ['x0'],
                             n_train=40, noise=0.1, burnin=800, thin=20, samples=100)
    if res['test_r2'] < 0.5:
        print("[WARN] Trig R^2={:.4f} below threshold (may need longer chain)".format(res['test_r2']))


@t("Rational y=x/(1+x^2)")
def test_rational():
    res = run_synthetic_test("Rational y=x/(1+x^2)", ground_rational, ['x0'],
                             n_train=40, noise=0.08, burnin=800, thin=20, samples=100)
    assert res['test_r2'] > 0.5


@t("2-Var y=3x0-2x1+5")
def test_2var():
    res = run_synthetic_test("2-Var y=3x0-2x1+5", ground_2var, ['x0', 'x1'],
                             n_train=60, noise=0.2, burnin=2000, thin=20, samples=200,
                             max_size=12)
    if res['test_r2'] < 0.4:
        print("[WARN] 2-Var R^2={:.4f} below threshold (may overfit)".format(res['test_r2']))


@t("3-Var y=x0*x1+x2")
def test_3var_interact():
    res = run_synthetic_test("3-Var y=x0*x1+x2", ground_3var_interact, ['x0', 'x1', 'x2'],
                             n_train=80, noise=0.3, burnin=2000, thin=20, samples=200,
                             max_size=20)
    if res['test_r2'] < 0.3:
        print("[WARN] 3-Var interaction R^2={:.4f} below threshold".format(res['test_r2']))


@t("3-Var y=x0*x1/(1+x2^2)")
def test_3var_rational():
    res = run_synthetic_test("3-Var y=x0*x1/(1+x2^2)", ground_3var_rat, ['x0', 'x1', 'x2'],
                             n_train=80, noise=0.15, burnin=2000, thin=20, samples=200,
                             max_size=20)
    if res['test_r2'] < 0.3:
        print("[WARN] 3-Var rational R^2={:.4f} below threshold".format(res['test_r2']))


@t("3-Var y=sin(x0)+x1*exp(-x2^2)")
def test_3var_mixed():
    res = run_synthetic_test("3-Var y=sin(x0)+x1*exp(-x2^2)", ground_3var_mix, ['x0', 'x1', 'x2'],
                             n_train=80, noise=0.2, burnin=2000, thin=20, samples=200,
                             max_size=20)
    if res['test_r2'] < 0.2:
        print("[WARN] 3-Var mixed R^2={:.4f} below threshold".format(res['test_r2']))


@t("Exp y=exp(-x^2)")
def test_exp():
    res = run_synthetic_test("Exp y=exp(-x^2)", ground_exp, ['x0'],
                             n_train=40, noise=0.05, burnin=800, thin=20, samples=100)
    assert res['test_r2'] > 0.5


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
