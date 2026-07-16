# bms_refactored/test_regression.py
"""
Regression tests for the Phase 0 refactor of mcmc.py into core/.

This script verifies that the core module can be imported, Tree/Node behave
as expected, and that basic MCMC moves still execute without error.
"""

import sys
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
t = Tree(prior_par={'Nopi_+': 0, 'Nopi_*': 0, 'Nopi_-': 0, 'Nopi_/': 0})
print("[PASS] Tree initialized with no data")
assert t.size == 1
print(f"[INFO] Initial tree size: {t.size}, expression: {t}")

# ---------------------------------------------------------------------------
# 4. Tree from string
# ---------------------------------------------------------------------------
t2 = Tree(
    prior_par={'Nopi_+': 0, 'Nopi_*': 0, 'Nopi_-': 0, 'Nopi_/': 0},
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
    prior_par={'Nopi_+': 1.0, 'Nopi_*': 1.0, 'Nopi_-': 1.0, 'Nopi_/': 1.0},
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

print("\n" + "="*60)
print("All regression tests passed!")
print("="*60)
