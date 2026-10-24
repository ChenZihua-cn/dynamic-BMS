# core/energy.py

import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy
import pandas as pd
from sympy import Symbol, lambdify, log, sympify
from scipy.optimize import curve_fit
from copy import deepcopy


class EnergyMixin:
    """Energy and goodness-of-fit calculations for Tree."""

    # -------------------------------------------------------------------------
    def get_sse(
        self,
        fit: bool = True,
        verbose: bool = False,
    ) -> dict[str, float] | int:
        """Return the sum of squared errors for each dataset.

        If ``fit`` is true, the free parameters of the expression are fitted
        with ``scipy.optimize.curve_fit`` before the errors are evaluated.
        """
        # Return 0 if there is no data
        if list(self.x.values())[0].empty or list(self.y.values())[0].empty:
            self.sse = 0
            return 0
        # Convert the Tree into a SymPy expression
        ex = sympify(str(self))
        # Convert the expression to a function that can be used by
        # curve_fit, i.e. that takes as arguments (x, a0, a1, ..., an)
        atomd = {a.name: a for a in ex.atoms() if a.is_Symbol}
        variables = [
            atomd[v] for v in self.variables + self.extra_variables
            if v in atomd
        ]
        parameters = [
            atomd[p] for p in self.parameters + self.fixed_parameters
            if p in atomd
        ]
        try:
            flam = lambdify(
                variables + parameters,
                ex,
                ["numpy", {'fac': scipy.special.factorial}],
            )
        except Exception:
            self.sse = {ds: np.inf for ds in self.x}
            return self.sse

        if fit:
            self.estimate_parameters(parameters, variables, flam, verbose=verbose)

        # Sum of squared errors
        self.sse = {}
        for ds in self.x:
            this_x, this_y = self.x[ds], self.y[ds]
            xmat = [this_x[v.name] for v in variables]
            args = [np.array(xi) for xi in xmat] + [
                self.par_values[ds][p.name] for p in parameters
            ]
            try:
                se = np.square(this_y - flam(*args))
                if np.any(np.isnan(se)):
                    raise ValueError
                self.sse[ds] = np.sum(se)
            except Exception:
                if verbose:
                    print(
                        f'> Cannot calculate SSE for {self}: inf',
                        file=sys.stderr,
                    )
                self.sse[ds] = np.inf

        # Done
        return self.sse

    def estimate_parameters(
        self,
        parameters: list[Symbol],
        variables: list[Symbol],
        flam: Callable[..., Any],
        verbose: bool = False,
    ) -> None:
        """Estimate the expression's free parameters for every dataset.

        Parameter values are stored in ``self.par_values`` and cached in
        ``self.fit_par`` under the expression string. If estimation fails,
        the current values are cached and, when ``verbose`` is true, reported.
        """
        if not parameters:  # Nothing to fit
            for ds in self.x:
                for p in self.parameters + self.fixed_parameters:
                    self.par_values[ds][p] = 1.0
            return

        expression = str(self)
        if expression in self.fit_par:  # Recover previously fit parameters
            self.par_values = self.fit_par[expression]
            return

        # Do the fit for all datasets
        self.fit_par[expression] = {}
        for ds in self.x:
            this_x, this_y = self.x[ds], self.y[ds]
            xmat = [this_x[v.name] for v in variables]

            def feval(x, *params):
                return flam(*x, *params)

            try:
                # Fit the parameters
                res = curve_fit(
                    feval,
                    xmat,
                    this_y,
                    p0=[self.par_values[ds][p.name] for p in parameters],
                    maxfev=10000,
                )
                # Reassign the values of the parameters
                self.par_values[ds] = {
                    parameters[i].name: res[0][i]
                    for i in range(len(res[0]))
                }
                for p in self.parameters + self.fixed_parameters:
                    if p not in self.par_values[ds]:
                        self.par_values[ds][p] = 1.0
                # Save this fit
                self.fit_par[expression][ds] = deepcopy(self.par_values[ds])
            except Exception:
                # Save this (unsuccessful) fit and print warning
                self.fit_par[expression][ds] = deepcopy(self.par_values[ds])
                if verbose:
                    print(
                        f"#Cannot_fit:{str(self).replace(' ', '')} # # # # #",
                        file=sys.stderr,
                    )

    # -------------------------------------------------------------------------
    def get_bic(
        self,
        reset: bool = True,
        fit: bool = False,
        verbose: bool = False,
    ) -> float:
        """Return the Bayesian information criterion summed over all datasets.

        ``fit`` is forwarded to :meth:`get_sse`. The parameter count is one
        plus the number of distinct parameters, including fixed ones. If
        ``reset`` is true, ``self.bic`` is updated.
        """
        if list(self.x.values())[0].empty or list(self.y.values())[0].empty:
            if reset:
                self.bic = 0
            return 0
        # Get the sum of squared errors (fitting, if required)
        sse = self.get_sse(fit=fit, verbose=verbose)
        # Calculate the BIC
        parameters = {
            p.value for p in self.ets[0] if p.value in self.parameters
        } | set(self.fixed_parameters)
        k = 1 + len(parameters)

        bic = 0.0
        for ds in self.y:
            n = len(self.y[ds])
            bic += (k - n) * np.log(n) + n * (
                np.log(2.0 * np.pi) + log(sse[ds]) + 1
            )

        if reset:
            self.bic = bic
        return bic

    # -------------------------------------------------------------------------
    def get_energy(
        self,
        bic: bool = False,
        reset: bool = False,
        verbose: bool = False,
    ) -> tuple[float, float, float]:
        """Return the energy as an approximation to minus the log-posterior.

        The data term uses the BIC in place of the exactly integrated
        likelihood. If ``bic`` is true, the BIC is recalculated first; if
        ``reset`` is true, ``self.E``, ``self.EB``, and ``self.EP`` are
        updated.
        """
        # Contribtution of the data (recalculating BIC if necessary)
        if bic:
            data_energy = self.get_bic(reset=reset, verbose=verbose) / 2.0
        else:
            data_energy = self.bic / 2.0
        # Contribution from the prior
        prior_energy = 0.0
        for op, nop in self.nops.items():
            try:
                prior_energy += self.prior_par[f"Nopi_{op}"] * nop
            except KeyError:
                pass
            try:
                prior_energy += self.prior_par[f"Nopi2_{op}"] * nop**2
            except KeyError:
                pass
        # Parliament structural penalty (Phase 1: structural only)
        if self.parliaments:
            prior_energy += self._parliament_energy_structural()

        total_energy = data_energy + prior_energy
        # Reset the value, if necessary
        if reset:
            self.EB = data_energy
            self.EP = prior_energy
            self.E = total_energy
        # Done
        return total_energy, data_energy, prior_energy

    # -------------------------------------------------------------------------
    def update_representative(self, verbose: bool = False) -> int:
        """Register this formula's canonical form or classify it against the representative.

        Returns 1 for a newly seen canonical form (after fitting and storing
        it), 0 if this tree is the stored representative, and -1 if a
        different tree is. Once registered, a representative is never replaced.
        """
        # Check for canonical representative
        canonical = self.canonical(verbose=verbose)

        if canonical not in self.representative:
            # Never seen this canonical formula before:
            # save it and return 1
            self.get_bic(reset=True, fit=True, verbose=verbose)
            new_energy = self.get_energy(bic=False, verbose=verbose)
            self.representative[canonical] = (
                str(self),
                new_energy,
                deepcopy(self.par_values),
            )
            return 1

        # If we've seen this canonical before, check if the
        # representative needs to be updated
        # We've seen this canonical before!
        representative, _, _ = self.representative[canonical]
        if representative == str(self):  # This IS the representative: return 0
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
