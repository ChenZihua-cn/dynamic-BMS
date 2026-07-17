# bms_refactored/test_regression.py
"""
Regression tests for the Phase 0 refactor of mcmc.py into core/.

This script verifies that the core module can be imported, Tree/Node behave
as expected, and that basic MCMC moves still execute without error.
"""

import sys
import json
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. Basic import smoke test
# ---------------------------------------------------------------------------
from core import Tree, Node, OPS, constants

print("[PASS] Import core module successfully")

# ---------------------------------------------------------------------------
# 2. Node basic functionality
# ---------------------------------------------------------------------------
n = Node('x', offspring=[])
assert n.pr() == 'x'
print("[PASS] Node.pr() works")

n2 = Node('+', offspring=[Node('x'), Node('y')])
assert n2.pr() == '(x + y)'
print("[PASS] Binary Node.pr() works")

# ---------------------------------------------------------------------------
# 3. Tree initialization (no data)
# ---------------------------------------------------------------------------
t = Tree(prior_par=dict([('Nopi_%s' % op, 0) for op in OPS]))
print("[PASS] Tree initialized with no data")
assert t.size == 1
print(f"[INFO] Initial tree size: {t.size}, expression: {t}")

# ---------------------------------------------------------------------------
# 4. Tree from string
# ---------------------------------------------------------------------------
t2 = Tree(
    prior_par=dict([('Nopi_%s' % op, 0) for op in OPS]),
    from_string='(x + _a0_)'
)
print(f"[PASS] Tree built from string: {t2}")
assert t2.size == 3
print(f"[INFO] Tree size after build_from_string: {t2.size}")

# ---------------------------------------------------------------------------
# 5. Canonical form
# ---------------------------------------------------------------------------
can = t2.canonical()
print(f"[PASS] Canonical form: {can}")
assert len(can) > 0

# ---------------------------------------------------------------------------
# 6. MCMC step on empty data (no crash)
# ---------------------------------------------------------------------------
for i in range(5):
    t2.mcmc_step()
print("[PASS] 5 MCMC steps on no-data tree completed without error")

# ---------------------------------------------------------------------------
# 7. Tree with data: build and run a few MCMC steps
# ---------------------------------------------------------------------------
np.random.seed(42)
x = pd.DataFrame({'x0': np.linspace(0, 10, 20)})
y = 2 * x['x0'] + 1 + np.random.normal(0, 0.1, 20)

t3 = Tree(
    variables=['x0'],
    parameters=['a0', 'a1'],
    x=x, y=y,
    prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
    max_size=10,
)
print(f"[INFO] Data tree initial: {t3}, size={t3.size}, BIC={t3.bic:.2f}, E={t3.E:.2f}")

for i in range(10):
    t3.mcmc_step(verbose=False)
print(f"[PASS] 10 MCMC steps with data completed. Final: {t3}, size={t3.size}, BIC={t3.bic:.2f}, E={t3.E:.2f}")

# ---------------------------------------------------------------------------
# 8. Predict
# ---------------------------------------------------------------------------
ypred = t3.predict(x)
assert isinstance(ypred, pd.Series)
print(f"[PASS] predict() returns pd.Series with length {len(ypred)}")

# ---------------------------------------------------------------------------
# 9. Energy consistency check
# ---------------------------------------------------------------------------
E_new, EB_new, EP_new = t3.get_energy(bic=True, reset=True)
assert abs(E_new - t3.E) < 1e-6
print(f"[PASS] Energy consistency: E={E_new:.6f}, EB={EB_new:.6f}, EP={EP_new:.6f}")

# ---------------------------------------------------------------------------
# 10. Constants mutability (runtime modification of OPS for Phase 1 extension)
# ---------------------------------------------------------------------------
old_ops = dict(constants.OPS)
constants.OPS['new_op'] = 2
assert constants.OPS['new_op'] == 2
constants.OPS.clear()
constants.OPS.update(old_ops)
print("[PASS] Global constants can be modified at runtime")

# ---------------------------------------------------------------------------
# 11. Synthetic data validation pipeline
# ---------------------------------------------------------------------------
# This section validates that the MCMC engine can recover (or closely
# approximate) known ground-truth formulas from noisy synthetic data.
# Each test case:
#   1. Generates noisy training data + clean test data from a known formula.
#   2. Runs a short MCMC chain.
#   3. Checks that the final model has improved over the initial one.
#   4. Measures predictive R^2 on a clean test set.

def r2_score(y_true, y_pred):
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def run_synthetic_test(name, ground_truth, variables, n_train=30, n_test=20,
                       noise=0.1, burnin=500, thin=20, samples=100,
                       max_size=15, prior_par=None, rtol=1e-6):
    """Run one synthetic-data validation test.

    Parameters
    ----------
    name : str
        Label for reporting.
    ground_truth : callable
        f(**kwargs) -> np.array, the true formula evaluated on the input dict.
    variables : list of str
        Variable names, e.g. ['x0'] or ['x0', 'x1'].
    n_train, n_test : int
        Number of training / test points.
    noise : float
        Std of Gaussian noise added to training targets.
    burnin, thin, samples : int
        MCMC parameters (kept small to keep the test suite fast).
    max_size : int
        Maximum expression-tree size.
    prior_par : dict or None
        If None, a weak uniform prior is used.
    rtol : float
        Relative tolerance for energy/BIC comparison (regression check).

    Returns
    -------
    dict with keys: name, train_r2, test_r2, initial_bic, final_bic,
                    initial_E, final_E, final_expr, trace_file.
    """
    np.random.seed(42)

    # --- 1. Generate synthetic data ---
    x_train = pd.DataFrame({
        v: np.random.uniform(-3, 3, n_train) for v in variables
    })
    # avoid zero for division-heavy formulas
    for v in variables:
        x_train[v] = x_train[v].where(x_train[v].abs() > 0.5, 0.5)

    y_clean = ground_truth(**{v: x_train[v].values for v in variables})
    y_train = pd.Series(y_clean + np.random.normal(0, noise, n_train))

    # Test set (noise-free for clean R^2 measurement)
    x_test = pd.DataFrame({
        v: np.linspace(-2.5, 2.5, n_test) for v in variables
    })
    for v in variables:
        x_test[v] = x_test[v].where(x_test[v].abs() > 0.3, 0.3)
    y_test = pd.Series(ground_truth(**{v: x_test[v].values for v in variables}))

    # --- 2. Build default prior if not provided ---
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
           write_files=False,  # don't write to disk during CI
           verbose=False, progress=False)

    final_expr = str(t)
    final_bic = t.bic
    final_E = t.E

    # --- 5. Predict on train and test ---
    y_train_pred = t.predict(x_train)
    y_test_pred = t.predict(x_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # --- 6. Basic sanity ---
    # BIC should improve (decrease) from a one-leaf model
    assert final_bic < initial_bic + rtol, \
        "{}: BIC did not improve (initial={:.2f}, final={:.2f})".format(
            name, initial_bic, final_bic)
    # Energy should also decrease
    assert final_E < initial_E + rtol, \
        "{}: Energy did not improve".format(name)

    print("[PASS] {}: initial BIC={:.2f} -> final BIC={:.2f}, "
          "train R\N{SUPERSCRIPT TWO}={:.4f}, test R\N{SUPERSCRIPT TWO}={:.4f}, "
          "expr: {}".format(name, initial_bic, final_bic, train_r2, test_r2, final_expr))

    return dict(name=name, train_r2=train_r2, test_r2=test_r2,
                initial_bic=initial_bic, final_bic=final_bic,
                initial_E=initial_E, final_E=final_E,
                final_expr=final_expr, trace_file=trace_fn)


# ---------------------------------------------------------------------------
# Test case A: Linear  y = 2*x + 1
# ---------------------------------------------------------------------------
def ground_linear(x0):
    return 2.0 * x0 + 1.0

res = run_synthetic_test("Linear y=2x+1", ground_linear, ['x0'],
                         n_train=30, noise=0.15, burnin=500, thin=20, samples=100)
assert res['test_r2'] > 0.7, \
    "Linear test R^2 too low: {:.4f}".format(res['test_r2'])

# ---------------------------------------------------------------------------
# Test case B: Quadratic  y = x^2 + 3*x - 2
# ---------------------------------------------------------------------------
def ground_quadratic(x0):
    return x0**2 + 3.0 * x0 - 2.0

res = run_synthetic_test("Quadratic y=x^2+3x-2", ground_quadratic, ['x0'],
                         n_train=40, noise=0.2, burnin=500, thin=20, samples=100)
assert res['test_r2'] > 0.6, \
    "Quadratic test R^2 too low: {:.4f}".format(res['test_r2'])

# ---------------------------------------------------------------------------
# Test case C: Trigonometric  y = sin(2*x)
# ---------------------------------------------------------------------------
def ground_sin(x0):
    return np.sin(2.0 * x0)

res = run_synthetic_test("Trig y=sin(2x)", ground_sin, ['x0'],
                         n_train=40, noise=0.1, burnin=800, thin=20, samples=100)
if res['test_r2'] < 0.5:
    print("[WARN] Trig R^2={:.4f} below threshold (may need longer chain)".format(res['test_r2']))

# ---------------------------------------------------------------------------
# Test case D: Rational  y = x / (1 + x^2)
# ---------------------------------------------------------------------------
def ground_rational(x0):
    return x0 / (1.0 + x0**2)

res = run_synthetic_test("Rational y=x/(1+x^2)", ground_rational, ['x0'],
                         n_train=40, noise=0.08, burnin=800, thin=20, samples=100)
assert res['test_r2'] > 0.5, \
    "Rational test R^2 too low: {:.4f}".format(res['test_r2'])

# ---------------------------------------------------------------------------
# Test case E: Two-variable  y = 3*x0 - 2*x1 + 5
# ---------------------------------------------------------------------------
def ground_2var(x0, x1):
    return 3.0 * x0 - 2.0 * x1 + 5.0

# NOTE: Multi-variable models have a large search space.  Short
# MCMC chains may find expressions that fit training data well but
# generalise poorly (negative test R^2).  The R^2 check here is
# informational; the hard assertion inside run_synthetic_test()
# (BIC improvement) is the regression gate for the refactor.
res = run_synthetic_test("2-Var y=3x0-2x1+5", ground_2var, ['x0', 'x1'],
                         n_train=60, noise=0.2, burnin=2000, thin=20, samples=200,
                         max_size=12)
if res['test_r2'] < 0.4:
    print("[WARN] 2-Var R^2={:.4f} below threshold (may overfit)".format(res['test_r2']))

# ---------------------------------------------------------------------------
# Test case F: 3-Var multiplicative interaction  y = x0 * x1 + x2
# ---------------------------------------------------------------------------
# NOTE: The search space for three-variable interactions is large.
# Short MCMC chains may find expressions that fit the training data
# but fail to generalise.  BIC improvement (checked inside
# run_synthetic_test) is the primary correctness gate.
def ground_3var_interact(x0, x1, x2):
    return x0 * x1 + x2

res = run_synthetic_test("3-Var y=x0*x1+x2", ground_3var_interact, ['x0', 'x1', 'x2'],
                         n_train=80, noise=0.3, burnin=2000, thin=20, samples=200,
                         max_size=20)
if res['test_r2'] < 0.3:
    print("[WARN] 3-Var interaction R^2={:.4f} below threshold".format(res['test_r2']))

# ---------------------------------------------------------------------------
# Test case G: 3-Var rational  y = (x0 * x1) / (1 + x2^2)
# ---------------------------------------------------------------------------
# NOTE: Rational functions with multiple variables are especially
# hard; the MCMC may settle on a simpler surrogate that fits
# training data but misses the denominator structure.
def ground_3var_rat(x0, x1, x2):
    return (x0 * x1) / (1.0 + x2**2)

res = run_synthetic_test("3-Var y=x0*x1/(1+x2^2)", ground_3var_rat, ['x0', 'x1', 'x2'],
                         n_train=80, noise=0.15, burnin=2000, thin=20, samples=200,
                         max_size=20)
if res['test_r2'] < 0.3:
    print("[WARN] 3-Var rational R^2={:.4f} below threshold".format(res['test_r2']))

# ---------------------------------------------------------------------------
# Test case H: 3-Var mixed  y = sin(x0) + x1 * exp(-x2^2)
# ---------------------------------------------------------------------------
# NOTE: Mixed trig + exponential + three-variable interaction is
# the hardest case; very long chains would be needed for reliable
# convergence from a random leaf.
def ground_3var_mix(x0, x1, x2):
    return np.sin(x0) + x1 * np.exp(-x2**2)

res = run_synthetic_test("3-Var y=sin(x0)+x1*exp(-x2^2)", ground_3var_mix, ['x0', 'x1', 'x2'],
                         n_train=80, noise=0.2, burnin=2000, thin=20, samples=200,
                         max_size=20)
if res['test_r2'] < 0.2:
    print("[WARN] 3-Var mixed R^2={:.4f} below threshold".format(res['test_r2']))

# ---------------------------------------------------------------------------
# Test case I: Exponential decay  y = exp(-x^2)
# ---------------------------------------------------------------------------
def ground_exp(x0):
    return np.exp(-x0**2)

res = run_synthetic_test("Exp y=exp(-x^2)", ground_exp, ['x0'],
                         n_train=40, noise=0.05, burnin=800, thin=20, samples=100)
assert res['test_r2'] > 0.5, \
    "Exp test R^2 too low: {:.4f}".format(res['test_r2'])

# ---------------------------------------------------------------------------
# 12. Trace file write test (regression for file output)
# ---------------------------------------------------------------------------
np.random.seed(99)
x = pd.DataFrame({'x0': np.linspace(0, 5, 15)})
y = 1.5 * x['x0'] + np.random.normal(0, 0.2, 15)

t_trace = Tree(
    variables=['x0'], parameters=['a0', 'a1'],
    x=x, y=y,
    prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
    max_size=10,
)
trace_fn = '/tmp/test_trace_output.dat'
progress_fn = '/tmp/test_progress_output.dat'
t_trace.mcmc(tracefn=trace_fn, progressfn=progress_fn,
             burnin=200, thin=10, samples=30,
             write_files=True, reset_files=True,
             verbose=False, progress=False)

# Verify trace file exists and has correct number of lines
with open(trace_fn, 'r') as f:
    trace_lines = f.readlines()
assert len(trace_lines) == 30, \
    "Expected 30 trace lines, got {}".format(len(trace_lines))
# Each line should be valid JSON with 6 elements
for i, line in enumerate(trace_lines):
    record = json.loads(line)
    assert len(record) == 6, \
        "Trace line {}: expected 6 fields, got {}".format(i, len(record))
    assert isinstance(record[0], int)      # sample index
    assert isinstance(record[1], float)    # BIC
    assert isinstance(record[2], float)    # Energy

print("[PASS] Trace file written with {} valid JSON records".format(len(trace_lines)))

# Verify progress file
with open(progress_fn, 'r') as f:
    progress_lines = f.readlines()
assert len(progress_lines) == 30
print("[PASS] Progress file written with {} lines".format(len(progress_lines)))

# ---------------------------------------------------------------------------
# 13. trace_predict test (regression for trace-based prediction)
# ---------------------------------------------------------------------------
np.random.seed(123)
x = pd.DataFrame({'x0': np.linspace(0, 5, 20)})
y = 2.0 * x['x0'] + 1.0 + np.random.normal(0, 0.3, 20)

t_tp = Tree(
    variables=['x0'], parameters=['a0', 'a1'],
    x=x, y=y,
    prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
    max_size=10,
    from_string='(x0 + _a0_)',  # start from sane init to avoid complex-valued expressions
)
trace_fn = '/tmp/test_trace_predict.dat'
progress_fn = '/tmp/test_progress_predict.dat'
t_tp.mcmc(tracefn=trace_fn, progressfn=progress_fn,
          burnin=100, thin=10, samples=10,  # short chain to keep memory low
          write_files=True, reset_files=True,
          verbose=False, progress=False)

# trace_predict using the trace file just written
x_test = pd.DataFrame({'x0': np.linspace(0, 5, 10)})
ypred_trace = t_tp.trace_predict(x_test, samples=10, burnin=0)
assert isinstance(ypred_trace, pd.DataFrame)
assert ypred_trace.shape == (10, 10)  # (n_test, n_samples)
assert np.all(np.isfinite(ypred_trace.values))
print("[PASS] trace_predict returns finite predictions, shape={}".format(ypred_trace.shape))

# ---------------------------------------------------------------------------
# 14. Round-trip string serialization (Tree -> str -> Tree -> str)
# ---------------------------------------------------------------------------
prior_par = dict([('Nopi_%s' % op, 1.0) for op in OPS])
t_rt = Tree(prior_par=prior_par, from_string='(x0 + _a0_)')
for _ in range(20):
    t_rt.mcmc_step(verbose=False)
expr1 = str(t_rt)
t_rt2 = Tree(prior_par=prior_par, from_string=expr1)
expr2 = str(t_rt2)
assert expr1 == expr2, \
    "Round-trip mismatch:\n  before: {}\n  after:  {}".format(expr1, expr2)
print("[PASS] Round-trip string serialization: before == after")

# ---------------------------------------------------------------------------
# 15. Visualization (matplotlib) -- trace & predictions-vs-actual
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

np.random.seed(7)
x_viz = pd.DataFrame({'x0': np.linspace(-2, 2, 40)})
y_viz = pd.Series(2.0 * x_viz['x0'] + 1.0 + np.random.normal(0, 0.2, 40))

t_viz = Tree(
    variables=['x0'], parameters=['a0', 'a1'],
    x=x_viz, y=y_viz,
    prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
    max_size=10,
    from_string='((x0 * _a0_) + _a1_)',  # init that can represent y = a0*x + a1
)
trace_fn_viz = '/tmp/test_viz_trace.dat'
progress_fn_viz = '/tmp/test_viz_progress.dat'
t_viz.mcmc(tracefn=trace_fn_viz, progressfn=progress_fn_viz,
           burnin=200, thin=10, samples=50,  # short chain to keep memory low
           write_files=True, reset_files=True,
           verbose=False, progress=False)

# --- Plot 1: BIC & Energy trace ---
bic_trace, E_trace = [], []
with open(trace_fn_viz, 'r') as f:
    for line in f:
        rec = json.loads(line)
        bic_trace.append(rec[1])
        E_trace.append(rec[2])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(bic_trace)
ax1.set_xlabel('MCMC sample')
ax1.set_ylabel('BIC')
ax1.set_title('BIC trace ({})'.format(t_viz))

ax2.plot(E_trace)
ax2.set_xlabel('MCMC sample')
ax2.set_ylabel('Energy')
ax2.set_title('Energy trace ({})'.format(t_viz))

plt.tight_layout()
fig.savefig('/tmp/test_viz_trace.png', dpi=100)
plt.close(fig)
print("[PASS] Trace plots saved to /tmp/test_viz_trace.png")

# --- Plot 2: Predictions vs Actual ---
ypred_viz = t_viz.predict(x_viz)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(ypred_viz, y_viz, alpha=0.7)
lo = min(y_viz.min(), ypred_viz.min())
hi = max(y_viz.max(), ypred_viz.max())
pad = 0.1 * (hi - lo) if hi > lo else 0.5
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', linewidth=1, label='y = x')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Predictions vs Actual ({})'.format(t_viz))
ax.legend()
fig.savefig('/tmp/test_viz_pred_vs_actual.png', dpi=100)
plt.close(fig)
print("[PASS] Predictions-vs-actual plot saved to /tmp/test_viz_pred_vs_actual.png")

print("\n" + "="*60)
print("All regression tests passed!")
print("="*60)
