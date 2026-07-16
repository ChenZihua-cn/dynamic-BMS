# refactor the original mcmc.py 

The directory try to spilit the logic of `mcmc.py`(~1760 line) into 5 module and an init entry file, which is used to provide a clear interface for next phase study.

```
main/
|-- e.t.c... 
|-- mcmc.py
|-- test_regression.py
`-- core/
    |--__init__.py   # combine 'Tree' class
    |-- constants.py # global const and config
    |-- node.py      # define 'Node' class 
    |-- tree_base.py # define 'TreeBase'
    |-- energy.py    # define 'EnergyMixin'
    |-- proposal.py  # define 'ProsposalMixin'
    `-- mcmc.py      # define 'MCMCMixin'
```

