# core/mcmc.py

import sys
from contextlib import ExitStack
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
    def mcmc_step(
        self,
        verbose: bool = False,
        p_rr: float = 0.05,
        p_long: float = 0.45,
    ) -> None:
        """Perform one MCMC update of the expression tree.

        With probability ``p_rr`` the move targets the root (pruning or
        replacing it), with probability ``p_long`` it changes a random node,
        and otherwise it replaces an elementary tree. Proposals are accepted
        with the temperature-scaled Metropolis probability.
        """

        # rr = replace root
        # et = elementary tree move
        # long = long-range move
        top_dice = random()

        match top_dice:
            # Root replacement move
            case dice if dice < p_rr:
                if random() < 0.5:
                    # Try to prune the root
                    dE, dEB, dEP, par_values_new = self.dE_rr(
                        rr=None,
                        verbose=verbose,
                    )
                    if -dEB / self.BT - dEP / self.PT > 300:
                        p_accept = 1.0
                    else:
                        p_accept = np.exp(
                            -dEB / self.BT - dEP / self.PT
                        ) / float(self.num_rr)

                    if random() < p_accept:
                        # Accept move
                        self.prune_root(update_gof=False, verbose=verbose)
                        self.par_values = par_values_new
                        self.get_bic(reset=True, fit=False, verbose=verbose)
                        self.E += dE
                        self.EB += dEB
                        self.EP += dEP
                else:
                    # Try to replace the root
                    new_rr = choice(self.rr_space)
                    dE, dEB, dEP, par_values_new = self.dE_rr(
                        rr=new_rr,
                        verbose=verbose,
                    )
                    if (
                        self.num_rr > 0
                        and -dEB / self.BT - dEP / self.PT > 0
                    ):
                        p_accept = 1.0
                    elif self.num_rr == 0:
                        p_accept = 0.0
                    else:
                        p_accept = self.num_rr * np.exp(
                            -dEB / self.BT - dEP / self.PT
                        )

                    if random() < p_accept:
                        # Accept move
                        self.replace_root(
                            rr=new_rr,
                            update_gof=False,
                            verbose=verbose,
                        )
                        self.par_values = par_values_new
                        self.get_bic(reset=True, fit=False, verbose=verbose)
                        self.E += dE
                        self.EB += dEB
                        self.EP += dEP

            # Long-range move
            case dice if dice < p_rr + p_long:
                # Choose a random node in the tree, and a random new operation
                target = choice(self.nodes)
                while True:
                    if not target.offspring:
                        new = choice(self.variables + self.parameters)
                        break
                    new = choice(list(self.ops))
                    if self.ops[new] == self.ops[target.value]:
                        break

                dE, dEB, dEP, par_values_new = self.dE_lr(
                    target,
                    new,
                    verbose=verbose,
                )
                try:
                    p_accept = np.exp(-dEB / self.BT - dEP / self.PT)
                except Exception:
                    if dEB / self.BT + dEP / self.PT < 0:
                        p_accept = 1.0

                # Accept move, if necessary
                if random() < p_accept:
                    # update number of operations
                    if target.offspring:
                        self.nops[target.value] -= 1
                        self.nops[new] += 1
                    # move
                    target.value = new
                    # recalculate distinct parameters
                    self.dist_par = list({
                        node.value for node in self.ets[0]
                        if node.value in self.parameters
                    })
                    self.n_dist_par = len(self.dist_par)
                    # update others
                    self.par_values = deepcopy(par_values_new)
                    self.get_bic(reset=True, fit=False, verbose=verbose)
                    self.E += dE
                    self.EB += dEB
                    self.EP += dEP

            # Elementary tree (short-range) move
            case _:
                # Choose a feasible move (doable and keeping size<=max_size)
                while True:
                    oini, ofin = choice(self.move_types)
                    if (
                        self.ets[oini]
                        and self.size - oini + ofin <= self.max_size
                    ):
                        break

                # target and new ETs
                target = choice(self.ets[oini])
                new = choice(self.et_space[ofin])
                # omegai and omegaf
                omega_i = len(self.ets[oini])
                omega_f = len(self.ets[ofin]) + 1
                if ofin == 0:
                    omega_f -= oini
                if oini == 0 and target.parent in self.ets[ofin]:
                    omega_f -= 1
                # size of et_space of each type
                size_i = len(self.et_space[oini])
                size_f = len(self.et_space[ofin])
                # Probability of acceptance
                dE, dEB, dEP, par_values_new, nif, nfi = self.dE_et(
                    target,
                    new,
                    verbose=verbose,
                )
                try:
                    p_accept = (
                        float(nif) * omega_i * size_f
                        * np.exp(-dEB / self.BT - dEP / self.PT)
                    ) / (float(nfi) * omega_f * size_i)
                except Exception:
                    if dEB / self.BT + dEP / self.PT < -200:
                        p_accept = 1.0

                # Accept / reject
                if random() < p_accept:
                    # Accept move
                    self.et_replace(target, new, verbose=verbose)
                    self.par_values = par_values_new
                    self.get_bic(verbose=verbose)
                    self.E += dE
                    self.EB += dEB
                    self.EP += dEP

        # Done

    # -------------------------------------------------------------------------
    def mcmc(
        self,
        tracefn: str = "trace.dat",
        progressfn: str = "progress.dat",
        write_files: bool = True,
        reset_files: bool = True,
        burnin: int = 2000,
        thin: int = 10,
        samples: int = 10000,
        verbose: bool = False,
        progress: bool = True,
    ) -> None:
        """Sample formula trees with MCMC and record the chain.

        Run ``burnin`` steps before sampling. During sampling, take ``thin``
        MCMC steps between successive records. If ``write_files`` is true,
        write one JSON record per sample to ``tracefn`` and one numeric
        progress line to ``progressfn``; ``reset_files`` truncates these
        files instead of appending. If ``progress`` is true, draw a progress
        bar on standard output.
        """
        self.get_energy(reset=True, verbose=verbose)

        # Burnin
        if progress:
            print("# Burning in\t", end="")
            print(f"[{' ' * 50}]", end="", flush=True)
            print("\b" * (50 + 1), end="")
        for step in range(burnin):
            self.mcmc_step(verbose=verbose)
            if progress and step % (burnin / 50) == 0:
                print("=", end="", flush=True)

        # Sample
        with ExitStack() as stack:
            if write_files:
                mode = "w" if reset_files else "a"
                tracef = stack.enter_context(open(tracefn, mode))
                progressf = stack.enter_context(open(progressfn, mode))
            else:
                tracef = progressf = None

            if progress:
                print("\n# Sampling\t", end="")
                print(f"[{' ' * 50}]", end="", flush=True)
                print("\b" * (50 + 1), end="")

            for sample in range(samples):
                for _ in range(thin):
                    self.mcmc_step(verbose=verbose)
                if progress and sample % (samples / 50) == 0:
                    print("=", end="", flush=True)
                if write_files:
                    json.dump(
                        [
                            sample,
                            float(self.bic),
                            float(self.E),
                            str(self.get_energy(verbose=verbose)),
                            str(self),
                            self.par_values,
                        ],
                        tracef,
                    )
                    tracef.write("\n")
                    tracef.flush()
                    progressf.write(
                        f"{sample} {float(self.E):f} {float(self.bic):f}\n"
                    )
                    progressf.flush()

        # Done
        if progress:
            print()

    # -------------------------------------------------------------------------
    def predict(
        self,
        x: pd.DataFrame | dict,
    ) -> pd.Series | dict:
        """Evaluate the formula on new data.

        ``x`` must have the same layout as the training data: a DataFrame, or
        a mapping from dataset labels to DataFrames when several datasets were
        used for training. A DataFrame input returns a Series; a mapping
        returns a mapping from dataset label to Series.

        If the vectorized evaluation fails, the prediction for that dataset is
        filled with NaN.
        """
        is_dataframe = isinstance(x, pd.DataFrame)
        if is_dataframe:
            datasets = {"d0": x}
        elif isinstance(x, dict):
            datasets = x
        else:
            raise TypeError("x must be either a dict or a pandas.DataFrame")

        # Convert the Tree into a SymPy expression
        expression = sympify(str(self))
        # Convert the expression to a function
        atomd = {
            atom.name: atom for atom in expression.atoms() if atom.is_Symbol
        }
        variables = [
            atomd[name]
            for name in self.variables + self.extra_variables
            if name in atomd
        ]
        parameters = [
            atomd[name]
            for name in self.parameters + self.fixed_parameters
            if name in atomd
        ]
        flam = lambdify(
            variables + parameters,
            expression,
            ["numpy", {"fac": scipy.special.factorial}],
        )

        # Loop over datasets
        predictions = {}
        for dataset, frame in datasets.items():
            # Prepare variables and parameters
            xmat = [frame[v.name] for v in variables]
            params = [self.par_values[dataset][p.name] for p in parameters]
            # Predict
            try:
                prediction = flam(*xmat, *params)
            except Exception:
                # Do it point by point
                prediction = [np.nan] * len(frame)
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
            predictions[dataset] = pd.Series(prediction, index=frame.index)

        return predictions["d0"] if is_dataframe else predictions

    # -------------------------------------------------------------------------
    def trace_predict(
        self,
        x: pd.DataFrame | dict,
        burnin: int = 1000,
        thin: int = 2000,
        samples: int = 1000,
        tracefn: str = "trace.dat",
        progressfn: str = "progress.dat",
        write_files: bool = True,
        reset_files: bool = True,
        verbose: bool = False,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Sample formula trees with MCMC and predict y(x) for each sample.

        Run ``burnin`` MCMC steps first, then take ``thin`` steps between
        successive predictions. Predictions are stored under the sample
        number and returned as a DataFrame.

        If ``write_files`` is true, write one JSON record per sample to
        ``tracefn`` and one numeric progress line to ``progressfn``;
        ``reset_files`` truncates these files instead of appending. If
        ``progress`` is true, draw a progress bar on standard output.
        """
        ypred = {}
        # Burnin
        if progress:
            print("# Burning in\t", end="")
            print(f"[{' ' * 50}]", end="", flush=True)
            print("\b" * (50 + 1), end="", flush=True)
        for step in range(burnin):
            self.mcmc_step(verbose=verbose)
            if progress and step % (burnin / 50) == 0:
                print("=", end="", flush=True)

        # Sample
        with ExitStack() as stack:
            if write_files:
                mode = "w" if reset_files else "a"
                tracef = stack.enter_context(open(tracefn, mode))
                progressf = stack.enter_context(open(progressfn, mode))
            else:
                tracef = progressf = None

            if progress:
                print("\n# Sampling\t", end="")
                print(f"[{' ' * 50}]", end="", flush=True)
                print("\b" * (50 + 1), end="", flush=True)

            for sample in range(samples):
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
                for _ in range(thin):
                    self.mcmc_step(verbose=verbose)
                # Make prediction
                ypred[sample] = self.predict(x)
                # Output
                if progress and sample % (samples / 50) == 0:
                    print("=", end="", flush=True)
                if write_files:
                    json.dump(
                        [
                            sample,
                            float(self.bic),
                            float(self.E),
                            str(self.get_energy(verbose=verbose)),
                            str(self),
                            self.par_values,
                        ],
                        tracef,
                    )
                    tracef.write("\n")
                    tracef.flush()
                    progressf.write(
                        f"{sample} {float(self.E):f} {float(self.bic):f}\n"
                    )
                    progressf.flush()

        # Done
        if progress:
            print()
        return pd.DataFrame.from_dict(ypred)
