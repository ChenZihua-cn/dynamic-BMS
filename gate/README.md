# dynamic gating network design 

The essential component of MoE approach

```
main
|-- e.t.c...
`-- gate/ 
    |-- __init__.py               # 
    |-- base.py                   # abstract base class 'GateBase'
    |-- neural_gate.py            # lighty NN gating network
    |-- attention_gate.py (Additional DLC)  
    |-- feature_extractor.py      # extract data & tree structure feature
    |-- hard_gate.py              # Top-k Hard gating strategy
    `-- configs/                  #
        `-- default.yaml          # pre-train config

```

