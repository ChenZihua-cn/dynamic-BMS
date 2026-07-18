"""Tests for trace file I/O and trace_predict."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
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
# 12. Trace file write test
# ---------------------------------------------------------------------------

@t("Trace file written correctly")
def test_trace_file_write():
    np.random.seed(99)
    x = pd.DataFrame({'x0': np.linspace(0, 5, 15)})
    y = 1.5 * x['x0'] + np.random.normal(0, 0.2, 15)

    t = Tree(
        variables=['x0'], parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
        max_size=10,
    )
    trace_fn = '/tmp/test_trace_output.dat'
    progress_fn = '/tmp/test_progress_output.dat'
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=200, thin=10, samples=30,
           write_files=True, reset_files=True,
           verbose=False, progress=False)

    with open(trace_fn, 'r') as f:
        trace_lines = f.readlines()
    assert len(trace_lines) == 30

    for i, line in enumerate(trace_lines):
        record = json.loads(line)
        assert len(record) == 6
        assert isinstance(record[0], int)
        assert isinstance(record[1], float)
        assert isinstance(record[2], float)

    with open(progress_fn, 'r') as f:
        progress_lines = f.readlines()
    assert len(progress_lines) == 30


# ---------------------------------------------------------------------------
# 13. trace_predict test
# ---------------------------------------------------------------------------

@t("trace_predict returns finite predictions")
def test_trace_predict():
    np.random.seed(123)
    x = pd.DataFrame({'x0': np.linspace(0, 5, 20)})
    y = 2.0 * x['x0'] + 1.0 + np.random.normal(0, 0.3, 20)

    t = Tree(
        variables=['x0'], parameters=['a0', 'a1'],
        x=x, y=y,
        prior_par=dict([('Nopi_%s' % op, 1.0) for op in OPS]),
        max_size=10,
        from_string='(x0 + _a0_)',
    )
    trace_fn = '/tmp/test_trace_predict.dat'
    progress_fn = '/tmp/test_progress_predict.dat'
    t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
           burnin=100, thin=10, samples=10,
           write_files=True, reset_files=True,
           verbose=False, progress=False)

    x_test = pd.DataFrame({'x0': np.linspace(0, 5, 10)})
    ypred_trace = t.trace_predict(x_test, samples=10, burnin=0)
    assert isinstance(ypred_trace, pd.DataFrame)
    assert ypred_trace.shape == (10, 10)
    assert np.all(np.isfinite(ypred_trace.values))


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
