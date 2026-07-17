# Experts: Constitution & Parliament

Two-layer physical prior formalization. The constitution layer enforces hard constraints (0/1 infinite penalty) at tree generation/mutation time; the parliament layer provides soft priors whose relative weights \(g_k\) are inferred from data via hierarchical Bayes.

## Constitution Layer (Hard Constraints)

Always enforced — expressions failing these checks are rejected before likelihood evaluation.

| Hard Constraint | Physical Intuition | Implementation |
|-----------------|--------------------|----------------|
| Dimensional Consistency | Physical equations must be dimensionally homogeneous | Each node carries dimensional labels (M, L, T). Addition/subtraction requires identical dimensions; multiplication/division composes/cancels dimensions; elementary functions (sin, exp, log) require dimensionless arguments. Illegal trees are rejected at generation/mutation. |

## Parliament Layer (Soft Priors, participate in \(g_k\) weight competition)

No free hyperparameters — relative importance inferred from data via hierarchical Bayes (see `gate/`).

| Expert Prior | Physical Intuition | Formalization |
|-------------|--------------------|---------------|
| Occam's Razor | Simpler expressions preferred | Description-length prior based on tree node count / depth |
| Asymptotic Matching | Model must match known perturbative / limit theories (\(x \to 0\), \(x \to \infty\)) | Symbolic computation (Sympy) of asymptotic behavior, probability adjusted by deviation from theoretical expectation |
| Parameter Range | Physical parameters must stay within valid domain | Detect predictions falling outside physical region (e.g., \(q \notin (0,1]\)), lower prior probability |

## Directory Structure

```
experts/
├── __init__.py
├── base.py                     # ExpertBase abstract class
├── registry.py                 # Expert registry
├── constitution/
│   ├── __init__.py
│   ├── base.py                 # ConstitutionBase (hard constraint, 0/1 rejection)
│   └── dimensional.py          # Dimensional consistency via node labels (M, L, T)
├── parliament/
│   ├── __init__.py
│   ├── base.py                 # ParliamentBase (soft prior, returns log-probability)
│   ├── occam.py                # Occam's razor: description-length prior
│   ├── asymptotic.py           # Asymptotic matching: symbolic limit computation
│   └── parameter_range.py      # Physical parameter range: boundary sampling
└── domain_knowledge/
    ├── __init__.py
    ├── gw_merger.py            # GW BBH final-state priors (Smarr limit, test-particle limit)
    └── units.py                # Unit system definitions (M, L, T)
```
