# Experts prior expression formalization and construction

Here we define several experts and try to formalize different prior that statistically consistent with the corpus.

```
main
|-- e.t.c...
`-- experts/
    |-- __init__.py               #
    |-- base.py                   # abstract base class 'ExpertBase`
    |--- registry.py              # experts registry table for dynamic load
    |-- dimensional.py            # 
    |-- limit.py                  # 
    |-- symmetry.py               # 
    |-- parameter_range.py        # 
    |-- complexity.py             # inhert from prior_par
    |-- asymptotic.py             # 
    `-- domain_knowledge_experts/ # specialize for specific domain
        |-- __init__.py           #
        |-- gw_waveform.py        # 
        |-- thermodynamics.py     # migrate from sym_themo_constraint
        `-- units.py              # units system definition (M, L, T or dimensionless)

```
