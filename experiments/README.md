# Experiments and reproduce

This folder is used to test the reliabality of experts. 
- Phase1: test the complexity and dimension experts with pedulum case( except formula like 'T = k * L +b' )
- Phase2: add/test the limit expert with theory gas case( P-> 0, V-> \infinity )
- Phase3: add/test the symmetry expert with gravity wave exchange mass case( q<-->1/q )
- Phase4: add/test the parameter_range and asymptotic expert with BBH case( post-newtownian and ringdown )

```
main
|-- e.t.c...
`-- experiments/
    |-- single_experiment.py
    |-- sweep.py
    |-- configs/
        |-- phase1_pendulum.yaml
        |-- phase2_ideal_gas.yaml
        |-- phase3_gravity_wave.yaml
        `-- phase4_gw_merger.yaml
    |-- logs/
        |-- e.t.c...

```

