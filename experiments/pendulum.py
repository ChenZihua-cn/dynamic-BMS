import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json

from core import Tree, Node, OPS
from experts.constitution.dimensional import (
    DimensionalConstitution,
    DimExpr,
    DimensionCheckResult,
    DimensionFailure,
    ZERO,
    collect_constraints, solve_constraints,
)

np.random.seed(0)

n = 100
Length = np.linspace(0.1, 3.0, n)
g = 9.81
T_true = 2 * np.pi * np.sqrt(Length / g)
T_observe = T_true + np.random.normal(0, 0.1, n)

PENDULUM_DIMS ={
    'L': (1, 0, 0, 0, 0, 0, 0),
    'T': (0, 0, 1, 0, 0, 0, 0),

}


data = {"Length": Length, "Time": T_observe}
df = pd.DataFrame(data)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

plt.subplot(2, 2, 1)
df.plot.scatter(x= "Length", y= "Time", label= "Generated_Data")

t = Tree(
    variables=['L'], parameters=['a'],
    dimensions=PENDULUM_DIMS, constitutions=[DimensionalConstitution()],
    x=pd.DataFrame({"L": Length}), y=pd.Series(T_observe),
    prior_par= {f'Nopi_{op}':1.0 for op in OPS}

)
trace_fn = '/tmp/pendulum_trace.dat'
progress_fn = '/tmp/pendulum_progress.dat'

t.mcmc(tracefn=trace_fn, progressfn=progress_fn,
       burnin=200,
       thin=10,
       samples=50,
       )

y_predict = t.predict(pd.DataFrame({"L": Length}))

plt.subplot(2, 2, 2)




plt.show()




