# Reason/ train closed circuit system

...

```
main
|-- e.t.c...
`-- inference/
    |-- __init.py__           # 
    |-- dynamic_prior.py      # p_total = prod p_k ^ g_k
    |-- mcmc_with_gate.py     # iteration with gatewa (revise the probality of accept)
    |-- meta_trainer.py       # gating network trainer
    |-- online_finetuner.py   # stratagy
    |-- cache_manager.py      # model cache manager (replace the present 'representative` machanism)
    `-- trace.py              # MCMC trace log and analyse

```


