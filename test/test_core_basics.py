"""Tests for core import, Node, Tree basics, and constants."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core import Tree, Node, OPS

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
# 1. Basic import smoke test
# ---------------------------------------------------------------------------

@t("Import core module")
def test_import():
    from core import Tree, Node, OPS, constants
    assert Tree is not None
    assert Node is not None


# ---------------------------------------------------------------------------
# 2. Node basic functionality
# ---------------------------------------------------------------------------

@t("Node.pr() for leaf")
def test_node_pr_leaf():
    n = Node('x', offspring=[])
    assert n.pr() == 'x'


@t("Node.pr() for binary operator")
def test_node_pr_binary():
    n2 = Node('+', offspring=[Node('x'), Node('y')])
    assert n2.pr() == '(x + y)'


# ---------------------------------------------------------------------------
# 3. Tree initialization (no data)
# ---------------------------------------------------------------------------

@t("Tree init with no data")
def test_tree_init_no_data():
    t = Tree(prior_par=dict([('Nopi_%s' % op, 0) for op in OPS]))
    assert t.size == 1


# ---------------------------------------------------------------------------
# 4. Tree from string
# ---------------------------------------------------------------------------

@t("Tree built from string")
def test_tree_from_string():
    t2 = Tree(
        prior_par=dict([('Nopi_%s' % op, 0) for op in OPS]),
        from_string='(x + _a0_)'
    )
    assert t2.size == 3


# ---------------------------------------------------------------------------
# 5. Canonical form
# ---------------------------------------------------------------------------

@t("Tree canonical form: parameter renaming")
def test_canonical_param_rename():
    prior = dict([('Nopi_%s' % op, 0) for op in OPS])
    t1 = Tree(prior_par=prior, from_string='(x + _a0_)')
    can1 = t1.canonical()
    assert 'c1' in can1, \
        f"Expected canonical form to rename parameters to c1, got: {can1}"
    assert '_a0_' not in can1, \
        f"Canonical form should not contain original parameter name _a0_: {can1}"


@t("Tree canonical form: idempotent")
def test_canonical_idempotent():
    prior = dict([('Nopi_%s' % op, 0) for op in OPS])
    t1 = Tree(prior_par=prior, from_string='(x + _a0_)')
    can1 = t1.canonical()
    can1_again = t1.canonical()
    assert can1 == can1_again, \
        f"Canonical form is not idempotent: {can1} != {can1_again}"


@t("Tree canonical form: commutative equivalence")
def test_canonical_commutative():
    prior = dict([('Nopi_%s' % op, 0) for op in OPS])
    t1 = Tree(prior_par=prior, from_string='(x + _a0_)')
    t2 = Tree(prior_par=prior, from_string='(_a0_ + x)')
    can1 = t1.canonical()
    can2 = t2.canonical()
    assert can1 == can2, \
        f"Equivalent trees should have same canonical form:\n  {can1}\n  {can2}"


@t("Tree canonical form: different expressions distinct")
def test_canonical_distinct():
    prior = dict([('Nopi_%s' % op, 0) for op in OPS])
    t1 = Tree(prior_par=prior, from_string='(x + _a0_)')
    t3 = Tree(prior_par=prior, from_string='(x * _a0_)')
    can1 = t1.canonical()
    can3 = t3.canonical()
    assert can1 != can3, \
        "Different expressions should have distinct canonical forms"


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
