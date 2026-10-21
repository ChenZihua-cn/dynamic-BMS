# core/parallel.py

"""Parallel tempering (replica exchange) over the expression-space posterior.

Ported from upstream BMS ``parallel.py`` and adapted to this repository's
``Tree``, which carries extra constructor arguments (``dimensions``,
``target_dim``, ``constitutions``, ``parliaments``) that must reach every
replica.
"""

import sys
import json
import numpy as np
from random import random, randint


class Parallel:
    """A ladder of ``Tree`` replicas, one per BIC temperature.

    ``mcmc_step`` updates every replica; ``tree_swap`` attempts to exchange
    the expressions sitting at two adjacent temperatures.  The replica at
    ``BT = 1`` is the one whose samples are meaningful, and is kept in
    ``self.t1``.
    """

    # -------------------------------------------------------------------------
    def __init__(self, Ts, tree_cls=None, **tree_kwargs):
        from . import Tree          # late import: core/__init__ assembles Tree
        Ts = sorted(set(float(T) for T in Ts))
        if len(Ts) < 2:
            raise ValueError(
                'parallel tempering needs at least 2 temperatures, got %s; '
                'call Tree.mcmc directly for a single-temperature run' % (Ts,))
        if Ts[0] != 1.0:
            raise ValueError(
                'the ladder must include BT = 1.0, since only that replica '
                'samples the posterior (got min BT = %g)' % Ts[0])
        self.tree_cls = tree_cls or Tree
        self.Ts = Ts
        self.t1 = self.tree_cls(BT=1.0, **tree_kwargs)
        self.trees = {1.0: self.t1}
        for BT in Ts[1:]:
            t = self.tree_cls(BT=BT, root_value=str(self.t1), **tree_kwargs)
            # Fits and degeneracy bookkeeping are properties of the formula,
            # not of the replica holding it, so they are shared by reference.
            t.fit_par = self.t1.fit_par
            t.representative = self.t1.representative
            self.trees[BT] = t
        return

    # -------------------------------------------------------------------------
    def __getitem__(self, BT):
        return self.trees[float(BT)]

    # -------------------------------------------------------------------------
    def mcmc_step(self, verbose=False, p_rr=0.05, p_long=.45):
        """Make one MCMC step in each replica."""
        for tree in list(self.trees.values()):
            tree.mcmc_step(verbose=verbose, p_rr=p_rr, p_long=p_long)
        return

    # -------------------------------------------------------------------------
    def tree_swap(self):
        """Attempt to exchange the expressions at a random adjacent pair.

        Returns the two temperatures involved, or ``(None, None)``.
        """
        nT1 = randint(0, len(self.Ts) - 2)
        nT2 = nT1 + 1
        BT1, BT2 = self.Ts[nT1], self.Ts[nT2]
        t1, t2 = self.trees[BT1], self.trees[BT2]
        # Only EB enters here, and that is exact rather than an approximation:
        # every replica shares one PT, so the EP terms cancel identically in
        # the exchange ratio.  Giving the replicas different PT would silently
        # break detailed balance under this acceptance rule.
        dE = float(t1.EB) * (1. / BT2 - 1. / BT1) + \
             float(t2.EB) * (1. / BT1 - 1. / BT2)
        paccept = 1. if dE <= 0 else float(np.exp(-dE))
        if random() < paccept:
            self.trees[BT1], self.trees[BT2] = t2, t1
            t1.BT, t2.BT = BT2, BT1
            self.t1 = self.trees[1.0]
            return BT1, BT2
        return None, None

    # -------------------------------------------------------------------------
    def anneal(self, n=1000, factor=5, verbose=False):
        """Heat the whole ladder, mix, then cool it back to the declared Ts.

        Used when a pair has gone too long without accepting an exchange; the
        cold replica is otherwise trapped in whichever well it fell into during
        burnin.
        """
        for t in self.trees.values():
            t.BT *= factor
        for steps, name in [(n, 'heating'), (2 * n, 'cooling')]:
            for kk in range(steps):
                if verbose and kk % max(1, steps // 20) == 0:
                    print('# Annealing %s: %d / %d' % (name, kk, steps),
                          file=sys.stderr, flush=True)
                self.mcmc_step()
                self.tree_swap()
        for BT, t in self.trees.items():
            t.BT = BT
        self.t1 = self.trees[1.0]
        return

    # -------------------------------------------------------------------------
    def mcmc(self, tracefn='trace.dat', progressfn='progress.dat',
             write_files=True, reset_files=True,
             burnin=2000, thin=10, samples=10000, verbose=False,
             progress=True, anneal_after=0, anneal_factor=5.):
        """Sample with replica exchange, writing the same files as ``Tree.mcmc``.

        Traces are written for the BT = 1 replica only, so anything that parses
        a single-temperature trace reads these unchanged.
        """
        for t in self.trees.values():
            t.get_energy(reset=True, verbose=verbose)
        # Unlike upstream's Parallel.trace_predict, swaps are attempted during
        # burnin too: without them each replica equilibrates at its own
        # temperature and the cold one keeps whatever well it fell into.
        # Stall is measured in MCMC steps, not samples, so that anneal_after
        # means the same thing as it does in experiments/*/upstream_*.py.
        step = 0
        last_swap = dict([(T, 0) for T in self.Ts[:-1]])
        n_annealed = 0

        if progress:
            sys.stdout.write('# Burning in\t[%s]\n' % (' ' * 50))
        for i in range(burnin):
            self.mcmc_step(verbose=verbose)
            BT1, BT2 = self.tree_swap()
            step += 1
            if BT1 is not None and BT1 in last_swap:
                last_swap[BT1] = step
            if progress and i % max(1, burnin // 50) == 0:
                sys.stdout.write('\b=')
                sys.stdout.flush()

        if write_files:
            mode = 'w' if reset_files else 'a'
            tracef = open(tracefn, mode)
            progressf = open(progressfn, mode)
        if progress:
            sys.stdout.write('\n# Sampling\t[%s]\n' % (' ' * 50))

        for s in range(samples):
            for i in range(thin):
                self.mcmc_step(verbose=verbose)
                BT1, BT2 = self.tree_swap()
                step += 1
                if BT1 is not None and BT1 in last_swap:
                    last_swap[BT1] = step
            t = self.t1
            if write_files:
                json.dump([s, float(t.bic), float(t.E),
                           str(t.get_energy(verbose=verbose)),
                           str(t), t.par_values], tracef)
                tracef.write('\n')
                tracef.flush()
                progressf.write('%d %lf %lf\n' % (s, t.E, t.bic))
                progressf.flush()
            if progress and s % max(1, samples // 50) == 0:
                sys.stdout.write('\b=')
                sys.stdout.flush()
            if anneal_after:
                inactive = max([step - last_swap[T] for T in last_swap])
                if inactive > anneal_after:
                    self.anneal(n=anneal_after, factor=anneal_factor,
                                verbose=progress)
                    step += 3 * anneal_after      # anneal 本身消耗 n+2n 步
                    last_swap = dict([(T, step) for T in last_swap])
                    n_annealed += 1

        if write_files:
            tracef.close()
            progressf.close()
        if progress:
            sys.stdout.write('\n')
        return n_annealed
