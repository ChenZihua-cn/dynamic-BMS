# core/mcmc.py

import sys
import numpy as np
import pandas as pd
import scipy
import json
from copy import deepcopy
from random import random, choice
from sympy import sympify, lambdify


class MCMCMixin:
    """MCMC sampling, prediction, and trace generation."""

    # -------------------------------------------------------------------------
    def mcmc_step(self, verbose=False, p_rr=0.05, p_long=.45):
        """Make a single MCMC step.

        """
        topDice = random()
        # Root replacement move
        if topDice < p_rr:
            if random() < .5:
                # Try to prune the root
                dE, dEB, dEP, par_valuesNew = self.dE_rr(rr=None,
                                                         verbose=verbose)
                if -dEB / self.BT - dEP / self.PT > 300:
                    paccept = 1
                else:
                    paccept = np.exp(-dEB / self.BT - dEP / self.PT) / \
                              float(self.num_rr)
                dice = random()
                if dice < paccept:
                    # Accept move
                    self.prune_root(update_gof=False, verbose=verbose)
                    self.par_values = par_valuesNew
                    self.get_bic(reset=True, fit=False, verbose=verbose)
                    self.E += dE
                    self.EB += dEB
                    self.EP += dEP
            else:
                # Try to replace the root
                newrr = choice(self.rr_space)
                dE, dEB, dEP, par_valuesNew = self.dE_rr(rr=newrr,
                                                         verbose=verbose)
                if self.num_rr > 0 and -dEB / self.BT - dEP / self.PT > 0:
                    paccept = 1.
                elif self.num_rr == 0:
                    paccept = 0.
                else:
                    paccept = self.num_rr * np.exp(-dEB / self.BT - \
                                                   dEP / self.PT)
                dice = random()
                if dice < paccept:
                    # Accept move
                    self.replace_root(rr=newrr, update_gof=False,
                                      verbose=verbose)
                    self.par_values = par_valuesNew
                    self.get_bic(reset=True, fit=False, verbose=verbose)
                    self.E += dE
                    self.EB += dEB
                    self.EP += dEP

        # Long-range move
        elif topDice < (p_rr + p_long):
            # Choose a random node in the tree, and a random new operation
            target = choice(self.nodes)
            nready = False
            while not nready:
                if len(target.offspring) == 0:
                    new = choice(self.variables + self.parameters)
                    nready = True
                else:
                    new = choice(list(self.ops.keys()))
                    if self.ops[new] == self.ops[target.value]:
                        nready = True
            dE, dEB, dEP, par_valuesNew = self.dE_lr(target, new,
                                                     verbose=verbose)
            try:
                paccept = np.exp(-dEB / self.BT - dEP / self.PT)
            except:
                if (dEB / self.BT + dEP / self.PT) < 0:
                    paccept = 1.
            # Accept move, if necessary
            dice = random()
            if dice < paccept:
                # update number of operations
                if target.offspring != []:
                    self.nops[target.value] -= 1
                    self.nops[new] += 1
                # move
                target.value = new
                # recalculate distinct parameters
                self.dist_par = list(set([n.value for n in self.ets[0]
                                          if n.value in self.parameters]))
                self.n_dist_par = len(self.dist_par)
                # update others
                self.par_values = deepcopy(par_valuesNew)
                self.get_bic(reset=True, fit=False, verbose=verbose)
                self.E += dE
                self.EB += dEB
                self.EP += dEP

        # Elementary tree (short-range) move
        else:
            # Choose a feasible move (doable and keeping size<=max_size)
            while True:
                oini, ofin = choice(self.move_types)
                if (len(self.ets[oini]) > 0 and
                    (self.size - oini + ofin <= self.max_size)):
                    break
            # target and new ETs
            target = choice(self.ets[oini])
            new = choice(self.et_space[ofin])
            # omegai and omegaf
            omegai = len(self.ets[oini])
            omegaf = len(self.ets[ofin]) + 1
            if ofin == 0:
                omegaf -= oini
            if oini == 0 and target.parent in self.ets[ofin]:
                omegaf -= 1
            # size of et_space of each type
            si = len(self.et_space[oini])
            sf = len(self.et_space[ofin])
            # Probability of acceptance
            dE, dEB, dEP, par_valuesNew, nif, nfi = self.dE_et(target, new,
                                                               verbose=verbose)
            try:
                paccept = (float(nif) * omegai * sf *
                           np.exp(-dEB / self.BT - dEP / self.PT)) / \
                           (float(nfi) * omegaf * si)
            except:
                if (dEB / self.BT + dEP / self.PT) < -200:
                    paccept = 1.
            # Accept / reject
            dice = random()
            if dice < paccept:
                # Accept move
                self.et_replace(target, new, verbose=verbose)
                self.par_values = par_valuesNew
                self.get_bic(verbose=verbose)
                self.E += dE
                self.EB += dEB
                self.EP += dEP

        # Done
        return

    # -------------------------------------------------------------------------
    def mcmc(self, tracefn='trace.dat', progressfn='progress.dat',
             write_files=True, reset_files=True,
             burnin=2000, thin=10, samples=10000, verbose=False, progress=True):
        """Sample the space of formula trees using MCMC, and write the trace and some progress information to files (unless write_files is False).

        """
        self.get_energy(reset=True, verbose=verbose)

        # Burnin
        if progress:
            sys.stdout.write('# Burning in\t')
            sys.stdout.write('[%s]' % (' ' * 50))
            sys.stdout.flush()
            sys.stdout.write('\b' * (50+1))
        for i in range(burnin):
            self.mcmc_step(verbose=verbose)
            if progress and (i % (burnin / 50) == 0):
                sys.stdout.write('=')
                sys.stdout.flush()
        # Sample
        if write_files:
            if reset_files:
                tracef = open(tracefn, 'w')
                progressf = open(progressfn, 'w')
            else:
                tracef = open(tracefn, 'a')
                progressf = open(progressfn, 'a')
        if progress:
            sys.stdout.write('\n# Sampling\t')
            sys.stdout.write('[%s]' % (' ' * 50))
            sys.stdout.flush()
            sys.stdout.write('\b' * (50+1))
        for s in range(samples):
            for i in range(thin):
                self.mcmc_step(verbose=verbose)
            if progress and (s % (samples / 50) == 0):
                sys.stdout.write('=')
                sys.stdout.flush()
            if write_files:
                json.dump([s, float(self.bic), float(self.E),
                           str(self.get_energy(verbose=verbose)),
                           str(self), self.par_values], tracef)
                tracef.write('\n')
                tracef.flush()
                progressf.write('%d %lf %lf\n' % (s, self.E, self.bic))
                progressf.flush()
        # Done
        if progress:
            sys.stdout.write('\n')
        return

    # -------------------------------------------------------------------------
    def predict(self, x):
        """Calculate the value of the formula at the given data x. The data x
must have the same format as the training data and, in particular, it
it must specify to which dataset the test data belongs, if multiple
datasets where used for training.

        """
        if isinstance(x, pd.DataFrame):
            this_x = {'d0' : x}
            input_type = 'df'
        elif isinstance(x, dict):
            this_x = x
            input_type = 'dict'
        else:
            raise TypeError('x must be either a dict or a pandas.DataFrame')

        # Convert the Tree into a SymPy expression
        ex = sympify(str(self))
        # Convert the expression to a function
        atomd = dict([(a.name, a) for a in ex.atoms() if a.is_Symbol])
        variables = [atomd[v] for v in self.variables + self.extra_variables
                     if v in list(atomd.keys())]
        parameters = [atomd[p] for p in self.parameters + self.fixed_parameters
                      if p in list(atomd.keys())]
        flam = lambdify(
            variables + parameters, ex, [
                "numpy",
                {'fac' : scipy.special.factorial}
            ])
        # Loop over datasets
        predictions = {}
        for ds in this_x:
            # Prepare variables and parameters
            xmat = [this_x[ds][v.name] for v in variables]
            params = [self.par_values[ds][p.name] for p in parameters]
            args = [xi for xi in xmat] + [p for p in params]
            # Predict
            try:
                prediction = flam(*args)
            except:
                # Do it point by point
                prediction = [np.nan for i in range(len(this_x[ds]))]
                """
                # Do it point by point NOT WORKING!!!
                prediction = []
                for xi in xmat:
                    args = [xi] + [p for p in params]
                    try:
                        this_prediction = flam(*args)
                    except:
                        this_prediction = [np.nan]
                    prediction += this_prediction
                """
            predictions[ds] = pd.Series(prediction, index=list(this_x[ds].index))

        if input_type == 'df':
            return predictions['d0']
        else:
            return predictions

    # -------------------------------------------------------------------------
    def trace_predict(
            self,
            x,
            burnin=1000, thin=2000, samples=1000,
            tracefn='trace.dat', progressfn='progress.dat',
            write_files=True, reset_files=True, verbose=False, progress=True,
    ):
        """Sample the space of formula trees using MCMC, and predict y(x) for each of the sampled formula trees.

        """
        ypred = {}
        # Burnin
        if progress:
            sys.stdout.write('# Burning in\t')
            sys.stdout.write('[%s]' % (' ' * 50))
            sys.stdout.flush()
            sys.stdout.write('\b' * (50+1))
        for i in range(burnin):
            self.mcmc_step(verbose=verbose)
            if progress and (i % (burnin / 50) == 0):
                sys.stdout.write('=')
                sys.stdout.flush()
        # Sample
        if write_files:
            if reset_files:
                tracef = open(tracefn, 'w')
                progressf = open(progressfn, 'w')
            else:
                tracef = open(tracefn, 'a')
                progressf = open(progressfn, 'a')
        if progress:
            sys.stdout.write('\n# Sampling\t')
            sys.stdout.write('[%s]' % (' ' * 50))
            sys.stdout.flush()
            sys.stdout.write('\b' * (50+1))

        for s in range(samples):
            """
            # Warm up the BIC heavily to escape deep wells
            self.BT = 1.e100
            self.get_energy(bic=True, reset=True, verbose=verbose)
            for kk in range(thin/4):
                self.mcmc_step(verbose=verbose)
            # Back to thermalization
            self.BT = 1.
            self.get_energy(bic=True, reset=True, verbose=verbose)
            """
            for kk in range(thin):
                self.mcmc_step(verbose=verbose)
            # Make prediction
            ypred[s] = self.predict(x)
            # Output
            if progress and (s % (samples / 50) == 0):
                sys.stdout.write('=')
                sys.stdout.flush()
            if write_files:
                json.dump([s, float(self.bic), float(self.E),
                           float(self.get_energy(verbose=verbose)[0]),
                           str(self), self.par_values], tracef)
                tracef.write('\n')
                tracef.flush()
                progressf.write('%d %lf %lf\n' % (s, self.E, self.bic))
                progressf.flush()
        # Done
        if progress:
            sys.stdout.write('\n')
        return pd.DataFrame.from_dict(ypred)
