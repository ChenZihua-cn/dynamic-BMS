# core/energy.py

import sys
import numpy as np
import scipy
import pandas as pd
from sympy import sympify, lambdify, log
from scipy.optimize import curve_fit
from copy import deepcopy


class EnergyMixin:
    """Energy and goodness-of-fit calculations for Tree."""

    # -------------------------------------------------------------------------
    def get_sse(self, fit=True, verbose=False):
        """Get the sum of squared errors, fitting the expression represented by the Tree to the existing data, if specified (by default, yes).

        """
        # Return 0 if there is no data
        if list(self.x.values())[0].empty or list(self.y.values())[0].empty:
            self.sse = 0
            return 0
        # Convert the Tree into a SymPy expression
        ex = sympify(str(self))
        # Convert the expression to a function that can be used by
        # curve_fit, i.e. that takes as arguments (x, a0, a1, ..., an)
        atomd = dict([(a.name, a) for a in ex.atoms() if a.is_Symbol])
        variables = [atomd[v] for v in self.variables + self.extra_variables
                     if v in list(atomd.keys())]
        parameters = [atomd[p] for p in self.parameters + self.fixed_parameters
                      if p in list(atomd.keys())]
        try:
            flam = lambdify(
                variables + parameters, ex, [
                    "numpy",
                    {'fac' : scipy.special.factorial}
                ])
        except:
            self.sse = dict([(ds, np.inf) for ds in self.x])
            return self.sse
        if fit:
            if len(parameters) == 0: # Nothing to fit
                for ds in self.x:
                    for p in self.parameters:
                        self.par_values[ds][p] = 1.
                    for p in self.fixed_parameters:
                        self.par_values[ds][p] = 1.
            elif str(self) in self.fit_par: # Recover previously fit parameters
                self.par_values = self.fit_par[str(self)]
            else:                    # Do the fit for all datasets
                self.fit_par[str(self)] = {}
                for ds in self.x:
                    this_x, this_y = self.x[ds], self.y[ds]
                    xmat = [this_x[v.name] for v in variables]
                    def feval(x, *params):
                        args = [xi for xi in x] + [p for p in params]
                        return flam(*args)
                    try:
                        # Fit the parameters
                        res = curve_fit(
                            feval, xmat, this_y,
                            p0=[self.par_values[ds][p.name]
                                for p in parameters],
                            maxfev=10000,
                        )
                        # Reassign the values of the parameters
                        self.par_values[ds] = dict(
                            [(parameters[i].name, res[0][i])
                             for i in range(len(res[0]))]
                        )
                        for p in self.parameters + self.fixed_parameters:
                            if p not in self.par_values[ds]:
                                self.par_values[ds][p] = 1.
                        # Save this fit
                        self.fit_par[str(self)][ds] = deepcopy(
                            self.par_values[ds]
                        )
                    except:
                        # Save this (unsuccessful) fit and print warning
                        self.fit_par[str(self)][ds] = deepcopy(
                            self.par_values[ds]
                        )
                        if verbose:
                            print('#Cannot_fit:%s # # # # #' % str(self).replace(' ', ''), file=sys.stderr)

        # Sum of squared errors
        self.sse = {}
        for ds in self.x:
            this_x, this_y = self.x[ds], self.y[ds]
            xmat = [this_x[v.name] for v in variables]
            ar = [np.array(xi) for xi in xmat] + \
                 [self.par_values[ds][p.name] for p in parameters]
            try:
                se = np.square(this_y - flam(*ar))
                if sum(np.isnan(se)) > 0:
                    raise ValueError
                else:
                    self.sse[ds] = np.sum(se)
            except:
                if verbose:
                    print('> Cannot calculate SSE for %s: inf' % self, file=sys.stderr)
                self.sse[ds] = np.inf

        # Done
        return self.sse

    # -------------------------------------------------------------------------
    def get_bic(self, reset=True, fit=False, verbose=False):
        """Calculate the Bayesian information criterion (BIC) of the current expression, given the data. If reset==False, the value of self.bic will not be updated (by default, it will).

        """
        if list(self.x.values())[0].empty or list(self.y.values())[0].empty:
            if reset:
                self.bic = 0
            return 0
        # Get the sum of squared errors (fitting, if required)
        sse = self.get_sse(fit=fit, verbose=verbose)
        # Calculate the BIC
        parameters = set(
            [p.value for p in self.ets[0] if p.value in self.parameters] + \
            self.fixed_parameters
        )
        k = 1 + len(parameters)
        BIC = 0.
        for ds in self.y:
            n = len(self.y[ds])
            BIC += (k - n) * np.log(n) + n * (np.log(2. * np.pi) + log(sse[ds]) + 1)
        if reset == True:
            self.bic = BIC
        return BIC

    # -------------------------------------------------------------------------
    def get_energy(self, bic=False, reset=False, verbose=False):
        """Calculate the "energy" of a given formula, that is, approximate minus log-posterior of the formula given the data (the approximation coming from the use of the BIC instead of the exactly integrated likelihood).

        """
        # Contribtution of the data (recalculating BIC if necessary)
        if bic == True:
            EB = self.get_bic(reset=reset, verbose=verbose) / 2.
        else:
            EB = self.bic / 2.
        # Contribution from the prior
        EP = 0.0
        for op, nop in list(self.nops.items()):
            try:
                EP += self.prior_par['Nopi_%s' % op] * nop
            except KeyError:
                pass
            try:
                EP += self.prior_par['Nopi2_%s' % op] * nop**2
            except KeyError:
                pass
        # Parliament structural penalty (Phase 1: structural only)
        if self.parliaments:
            EP += self._parliament_energy_structural()
        # Reset the value, if necessary
        if reset:
            self.EB = EB
            self.EP = EP
            self.E = EB + EP
        # Done
        return EB + EP, EB, EP

    # -------------------------------------------------------------------------
    def update_representative(self, verbose=False):
        """
Check if we've seen this formula before, either in its current form or in another form.

*If we haven't seen it, save it and return 1.

*If we have seen it and this IS the representative, just return 0.

*If we have seen it and the representative has smaller energy, just return -1.

*If we have seen it and the representative has higher energy, update
the representatitve and return -2.

        """
        # Check for canonical representative
        canonical = self.canonical(verbose=verbose)
        try:             # We've seen this canonical before!
            rep, rep_energy, rep_par_values = self.representative[canonical]
        except KeyError: # Never seen this canonical formula before:
                         # save it and return 1
            self.get_bic(reset=True, fit=True, verbose=verbose)
            new_energy = self.get_energy(bic=False, verbose=verbose)
            self.representative[canonical] = (str(self), new_energy,
                                              deepcopy(self.par_values))
            return 1

        # If we've seen this canonical before, check if the
        # representative needs to be updated
        if rep == str(self): # This IS the representative: return 0
            return 0
        else:
            # CAUTION: CHANGED TO NEVER UPDATE REPRESENTATIVE!!!!!!!!
            return -1
            # END OF CAUTION ZONE
            """
            self.get_bic(reset=True, fit=True, verbose=verbose)
            new_energy = self.get_energy(bic=False, verbose=verbose)
            if (new_energy - rep_energy) < -1.e-6: # Update
                                                   # representative &
                                                   # return -2
                print >> sys.stdout, 'Updating rep: ||', canonical, '||', rep, '||', str(self), '||', rep_energy, '||', new_energy
                print >> sys.stderr, 'Updating rep: ||', canonical, '||',  rep, '||', str(self), '||', rep_energy, '||', new_energy
                self.representative[canonical] = (str(self),
                                                  new_energy,
                                                  deepcopy(self.par_values))
                return -2
            else: # Not the representative: return -1
                return -1
            """
