"""Tests for round-trip string serialization of Tree."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
# 14. Round-trip string serialization
# ---------------------------------------------------------------------------

@t("Round-trip string serialization")
def test_round_trip_serialization():
    prior_par = dict([('Nopi_%s' % op, 1.0) for op in OPS])
    t_rt = Tree(prior_par=prior_par, from_string='(x0 + _a0_)')
    for _ in range(20):
        t_rt.mcmc_step(verbose=False)
    expr1 = str(t_rt)
    t_rt2 = Tree(prior_par=prior_par, from_string=expr1)
    expr2 = str(t_rt2)
    assert expr1 == expr2, \
        "Round-trip mismatch:\n  before: {}\n  after:  {}".format(expr1, expr2)


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
