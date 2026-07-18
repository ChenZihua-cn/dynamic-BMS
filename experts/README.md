# Experts: Constitution & Parliament

Two-layer physical prior formalization. The constitution layer enforces hard constraints (0/1 infinite penalty) at tree generation/mutation time; the parliament layer provides soft priors whose relative weights \(g_k\) are inferred from data via hierarchical Bayes.

**Responsibility boundary:** Expert registration and constitution checks live in `experts/base.py` (``ExpertMixin``). Parliament weight inference (gamma, softmax, energy computation) lives in `gate/base.py` (``GateMixin``). The two are connected via ``ExpertMixin(GateMixin)`` — see `gate/README.md` for the hyperprior design.

## Constitution Layer (Hard Constraints)

Always enforced — expressions failing these checks are rejected before likelihood evaluation.

| Hard Constraint | Physical Intuition | Implementation |
|-----------------|--------------------|----------------|
| Dimensional Consistency | Physical equations must be dimensionally homogeneous | Each node carries dimensional labels (M, L, T). Addition/subtraction requires identical dimensions; multiplication/division composes/cancels dimensions; elementary functions (sin, exp, log) require dimensionless arguments. Illegal trees are rejected at generation/mutation. |

## Parliament Layer (Soft Priors, participate in \(g_k\) weight competition)

No free hyperparameters — relative importance inferred from data via hierarchical Bayes (see `gate/`).

| Expert Prior | Physical Intuition | Formalization |
|-------------|--------------------|---------------|
| Occam's Razor | Simpler expressions preferred | Description-length prior based on tree node count |
| Asymptotic Matching | Model must match known perturbative / limit theories (\(x \to 0\), \(x \to \infty\)) | **Phase 1:** simplified AST check penalizing division by variables that should be finite at zero. **Phase 2 (planned):** Sympy symbolic limit computation with continuous deviation penalty. |
| Parameter Range | Physical parameters must stay within valid domain | **待做 (Phase 2):** detect predictions falling outside physical region (e.g., \(q \notin (0,1]\)). |

## Directory Structure

```
experts/
├── __init__.py
├── base.py                     # ExpertMixin — expert registration + constitution checks
├── README.md
├── plan_origin.md
├── registry.py                 # 待做 (Phase 2)
├── constitution/
│   ├── __init__.py
│   ├── base.py                 # ConstitutionBase (hard constraint, 0/1 rejection)
│   └── dimensional.py          # Dimensional consistency via constraint solving
├── parliament/
│   ├── __init__.py
│   ├── base.py                 # ParliamentBase (soft prior, two-phase evaluation)
│   ├── occam.py                # Occam's razor: size-based penalty
│   ├── asymptotic.py           # Asymptotic matching: Phase 1 denominator check
│   └── parameter_range.py      # 待做 (Phase 2): physical parameter range
└── domain_knowledge/           # 待做 (Phase 2)
    ├── __init__.py
    ├── gw_merger.py            # GW BBH final-state priors (Smarr limit, test-particle limit)
    └── units.py                # Unit system definitions (M, L, T)
```
