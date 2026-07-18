"""Tests for core import, Node, Tree basics, and constants."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core import Tree, Node, OPS, constants

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

@t("Tree canonical form")
def test_canonical():
    t2 = Tree(
        prior_par=dict([('Nopi_%s' % op, 0) for op in OPS]),
        from_string='(x + _a0_)'
    )
    can = t2.canonical()
    assert len(can) > 0


# ---------------------------------------------------------------------------
# 10. Constants mutability
# ---------------------------------------------------------------------------

@t("Global constants can be modified at runtime")
def test_constants_mutability():
    old_ops = dict(constants.OPS)
    constants.OPS['new_op'] = 2
    assert constants.OPS['new_op'] == 2
    constants.OPS.clear()
    constants.OPS.update(old_ops)


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
