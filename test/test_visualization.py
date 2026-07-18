"""Tests for matplotlib visualization (trace & predictions-vs-actual)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pandas as pd
import numpy as np
from core import Tree, OPS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
# 15. Visualization (matplotlib)
# ---------------------------------------------------------------------------

@t("BIC/Energy trace plots saved")
def test_trace_plots():
    np.random.seed(7)
    x = pd.DataFrame({'x0': np.linspace(-2, 2, 40)})
    y = pd.Series(2.0 * x['x0'] + 1.0 + np.random.normal(0, 0.2, 40))

    t = Tree(
        variables=['x0'], parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
        max_size=10,
        from_string='((x0 * _a0_) + _a1_)',
    )
    trace_fn = '/tmp/test_viz_trace.dat'
    progress_fn = '/tmp/test_viz_progress.dat'
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=200, thin=10, samples=50,
           write_files=True, reset_files=True,
           verbose=False, progress=False)

    bic_trace, E_trace = [], []
    with open(trace_fn, 'r') as f:
        for line in f:
            rec = json.loads(line)
            bic_trace.append(rec[1])
            E_trace.append(rec[2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(bic_trace)
    ax1.set_xlabel('MCMC sample')
    ax1.set_ylabel('BIC')
    ax1.set_title('BIC trace ({})'.format(t))
    ax2.plot(E_trace)
    ax2.set_xlabel('MCMC sample')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy trace ({})'.format(t))
    plt.tight_layout()
    fig.savefig('/tmp/test_viz_trace.png', dpi=100)
    plt.close(fig)


@t("Predictions-vs-actual plot saved")
def test_pred_vs_actual_plot():
    np.random.seed(7)
    x = pd.DataFrame({'x0': np.linspace(-2, 2, 40)})
    y = pd.Series(2.0 * x['x0'] + 1.0 + np.random.normal(0, 0.2, 40))

    t = Tree(
        variables=['x0'], parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
        max_size=10,
        from_string='((x0 * _a0_) + _a1_)',
    )
    trace_fn = '/tmp/test_viz_trace.dat'
    progress_fn = '/tmp/test_viz_progress.dat'
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=200, thin=10, samples=50,
           write_files=True, reset_files=True,
           verbose=False, progress=False)

    ypred = t.predict(x)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ypred, y, alpha=0.7)
    lo = min(y.min(), ypred.min())
    hi = max(y.max(), ypred.max())
    pad = 0.1 * (hi - lo) if hi > lo else 0.5
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', linewidth=1, label='y = x')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Predictions vs Actual ({})'.format(t))
    ax.legend()
    fig.savefig('/tmp/test_viz_pred_vs_actual.png', dpi=100)
    plt.close(fig)


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
