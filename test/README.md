# Test Suite Overview

## Files

| File | Style | Tests | Focus |
|------|-------|-------|-------|
| [test_core_basics.py](test_core_basics.py) | decorator `@t()` | 9 | Core import, Node, Tree init, canonical form (param rename, idempotence, commutation, distinctness) |
| [test_asymptotic.py](test_asymptotic.py) | pytest classes | 16 | AsymptoticPrior expert: unit, ideal gas, multi-expert weighted combo, MCMC regression |
| [test_dimensional.py](test_dimensional.py) | decorator `@t()` | 18 | DimExpr algebra, constraint collection, dimensional consistency, DimensionalConstitution gate + MCMC |
| [test_mcmc.py](test_mcmc.py) | decorator `@t()` | 4 | MCMC steps (invariants, tree evolution, energy bounds), predict (length, finiteness, non-trivial), energy consistency |
| [test_serialization.py](test_serialization.py) | decorator `@t()` | 1 | Round-trip string serialization of Tree |
| [test_synthetic.py](test_synthetic.py) | decorator `@t()` | 9 | End-to-end synthetic data recovery: linear, quadratic, trig, rational, multi-var, exp — all hard assertions |
| [test_trace.py](test_trace.py) | decorator `@t()` | 2 | Trace file I/O, trace_predict |
| [test_visualization.py](test_visualization.py) | decorator `@t()` | 2 | Matplotlib: BIC/Energy trace data validation + plots, predictions-vs-actual correlation + scatter |
| [test_parliament.py](test_parliament.py) | pytest classes | 16 | ParliamentBase, OccamPrior, sign convention, cache, gate interaction, multi-expert, error resilience, dE_lr/dE_rr deltas |
| [test_pendulum.py](test_pendulum.py) | pytest classes | 31 | DimExpr algebra, DimensionCheckResult, valid/invalid pendulum dimension checks, EP computation, proposal gate, MCMC integration, fixed_term dimensions |
| [test_rejection_rate.py](test_rejection_rate.py) | pytest classes | 2 | Phase 3 observability: measure constitution gate rejection rate in MCMC (threshold < 80%) |

---

## Per-File Details

### test_core_basics.py
Smoke tests for the core module.
- **test_import** — verifies `Tree` and `Node` can be imported
- **test_node_pr_leaf / test_node_pr_binary** — pretty-print format for leaf and binary nodes
- **test_tree_init_no_data** — Tree initialization with prior_par only, checks `size == 1`
- **test_tree_from_string** — Tree built from `(_a0_ * x)`, checks `size == 3`
- **test_canonical_param_rename** — canonical form renames parameters (`_a0_` → `c1`)
- **test_canonical_idempotent** — applying canonical twice yields the same result
- **test_canonical_commutative** — `(x + _a0_)` and `(_a0_ + x)` produce the same canonical form
- **test_canonical_distinct** — different expressions produce distinct canonical forms

### test_asymptotic.py
Unit and integration tests for the `AsymptoticPrior` parliament expert (penalizes division by variables that should remain finite at zero).
- **TestAsymptoticPriorUnit** (8 tests) — no division → 0 penalty; division by target variable → +0.5; multiple violations stack; parameter denominators safe; nested/deeply-nested denominators; empty/None finite_at_zero → no opinion
- **TestAsymptoticPriorIdealGas** (5 tests) — Boyle's law physics: P in denominator penalized; no division or P in numerator passes; MCMC with OccamPrior + AsymptoticPrior on ideal gas data
- **TestAsymptoticPriorIntegration** (5 tests) — equal weights with two experts; weighted structural energy correctness; cache stores per-expert raw values; dE_et includes both contributions; EP differs with second expert
- **TestAsymptoticPriorMCMC** (1 test) — 50-step MCMC doesn't crash, EP and cache behave correctly

### test_dimensional.py
Unit and integration tests for the dimensional analysis constitution layer.
- **DimExpr unit** (5 tests) — creation, arithmetic (add/sub/mul), equality/hashing, all_even check
- **Constraint collection** (10 tests) — leaf with known dim; parameter leaf (symbolic); sin(dimensionless) valid; sin(L) invalid; L+T invalid; linear model (a\*x+b) valid; shared parameter conflict; dimension cancellation; sqrt(x/x) valid; sqrt(a*x) conservative reject; nested exp(sin(x/x)) valid; pow2 scales dim; sqrt(pow2(x)) valid; x\*\*y both must be dimensionless
- **DimensionalConstitution integration** (3 tests) — valid tree passes; invalid tree raises/rejected; backward compat (no constitution = no error)
- **MCMC with constitution** (2 tests) — all accepted formulas valid after MCMC; empty constitution returns True; multiple constitutions all must pass

### test_mcmc.py
MCMC invariants and regression tests.
- **test_mcmc_step_no_data** — 5 MCMC steps without data: asserts tree size in [1, max_size], finite energy/BIC after each step, tree structure changes
- **test_mcmc_step_with_data** — 10 MCMC steps with linear data: asserts size bounds, finite energy/BIC, tree evolves, energy doesn't increase substantially
- **test_predict** — predict() returns correct-length Series with all finite, non-constant values; works on different-sized test data
- **test_energy_consistency** — `get_energy(reset=True)` matches `t.E`

### test_serialization.py
- **test_round_trip_serialization** — runs MCMC, serializes tree to string, rebuilds from string, asserts expressions match

### test_synthetic.py
End-to-end synthetic data recovery pipeline. Each test generates data from a known ground-truth function, runs MCMC, and checks that BIC improves and test R² exceeds a hard threshold. All assertions are mandatory (no soft warnings).
- **test_linear** (R² > 0.7), **test_quadratic** (R² > 0.6), **test_rational** (R² > 0.5), **test_exp** (R² > 0.5)
- **test_trig** (R² > 0.3), **test_2var** (R² > 0.3), **test_3var_interact** (R² > 0.15), **test_3var_rational** (R² > 0.1), **test_3var_mixed** (R² > 0.05)

### test_trace.py
- **test_trace_file_write** — writes 30-sample trace, verifies line count and JSON schema
- **test_trace_predict** — trace_predict returns DataFrame with expected shape, all finite values

### test_visualization.py
Visualization tests with data validation assertions.
- **test_trace_plots** — runs MCMC, asserts 50 trace samples, all BIC/Energy finite, non-zero variance (chain explores), saves PNG
- **test_pred_vs_actual_plot** — runs MCMC, asserts prediction length matches input, all finite, correlation > 0.5 with actual, saves scatter PNG

### test_parliament.py
Comprehensive tests for the parliament (soft prior) layer.
- **TestParliamentBase** (1 test) — default methods return 0.0
- **TestOccamPrior** (1 test) — size penalty: 1 node → 1.0, 3 nodes → 3.0, 4 nodes → 4.0
- **TestSignConvention** (2 tests) — larger tree gives positive dEP; same-size swap gives finite dE
- **TestEnergyIncludesParliament** (2 tests) — EP includes structural penalty; with parliament > without
- **TestParliamentCache** (2 tests) — cache hit reuses values; per-expert raw values stored
- **TestParliamentGateInteraction** (2 tests) — dimensional fail → dE=inf; dimensional pass → parliament contributes
- **TestMultiParliament** (2 tests) — weights sum to 1; weighted sum correct
- **TestParliamentErrorResilience** (2 tests) — expert exception silenced → 0.0; capture/delta handle exceptions
- **TestCacheEviction** (1 test) — cache grows with new canonicals
- **TestMCMCWithParliament** (1 test) — 50-step MCMC, EP behaves, cache bounded
- **TestParliamentProposalGate** (5 tests) — dE_lr/replace/prune with parliament deltas; canonical reject skips parliament

### test_pendulum.py
Thorough dimensional analysis tests using pendulum physics as a running example.
- **TestDimExprArithmetic** (12 tests) — add, cancel, sub, scalar mul, zero mul, neg, rmul, is_zero, all_even variants, eq/hash, repr
- **TestDimensionCheckResult** (3 tests) — valid result, invalid with failures, default conflicting_exprs
- **TestDimensionalValid** (8 tests) — T=2π√(L/g), L/g, sqrt(L)/sqrt(g) (conservative reject), with offset, with angle offset, large-amplitude series, ratio form, pow2 scales, cancellation
- **TestDimensionalInvalid** (7 tests) — sin(L), L+g, sqrt(L), shared param conflict, cos(T²), pow with dims, shared param dim vs dimless, sin of dimensional with param
- **TestConstitutionIntegration** (2 tests) — valid tree passes check; invalid rejected in build
- **TestPriorEnergy** (7 tests) — single leaf = 0; binary op EP; correct pendulum EP; with offset; transcendental EP; custom prior_par; simpler < complex ordering
- **TestProposalGate** (8 tests) — dE_et reject/accept; dE_lr reject/accept; dE_rr reject/accept root add; dE_rr prune accept; dE_et transcendental reject
- **TestMCMCIntegration** (3 tests) — all steps pass constitution; invalid proposals always rejected; converges to correct formula
- **TestFixedTermDimensional** (4 tests) — valid dimensionless; valid same-dim; invalid dim mismatch; None falls through

### test_rejection_rate.py
Phase 3 observability: measures what fraction of proposals are rejected by the constitution gate.
- **test_rejection_rate_statistics** — 200 MCMC steps, instrumented dE_\* wrappers, asserts rejection rate < 80%
- **test_rejection_rate_with_parliament** — same with OccamPrior active, asserts rejection rate < 80%
