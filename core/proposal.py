# core/proposal.py

from .node import Node
from copy import deepcopy
import numpy as np
import sys
from random import choice


class ProposalMixin:
    """Tree manipulation and proposal distribution moves."""

    # -------------------------------------------------------------------------
    def replace_root(
        self,
        rr: list | None = None,
        update_gof: bool = True,
        verbose: bool = False,
    ) -> Node | None:
        """Replace the root with a root replacement ``rr``.

        Parameters
        ----------
        rr : list or None, optional
            Root replacement ``[value, [offspring, ...]]``. If it is
            ``None``, it is chosen with ``random.choice(self.rr_space)``,
            i.e. exactly one random draw is made.
        update_gof : bool, optional
            Recompute ``sse``, ``bic`` and ``E`` afterwards.
        verbose : bool, optional
            Passed on to the goodness-of-fit computations.

        Returns
        -------
        Node or None
            The new root, or ``None`` if the replacement would lead to a
            tree larger than ``self.max_size`` (the tree is then left
            unchanged).

        Raises
        ------
        RuntimeError
            If the order of the new root does not match
            ``self.ops[rr[0]]``, a bare ``raise`` is executed (which
            surfaces as ``RuntimeError`` outside an active exception).
        """
        # If no RR is provided, randomly choose one
        if rr is None:
            rr = choice(self.rr_space)
        # Return None if the replacement is too big
        if (self.size + self.ops[rr[0]]) > self.max_size:
            return None
        # Create the new root and replace exisiting root
        newRoot = Node(rr[0], offspring=[], parent=None)
        newRoot.order = 1 + len(rr[1])
        if newRoot.order != self.ops[rr[0]]:
            raise
        newRoot.offspring.append(self.root)
        self.root.parent = newRoot
        self.root = newRoot
        self.nops[self.root.value] += 1
        self.nodes.append(self.root)
        self.size += 1
        oldRoot = self.root.offspring[0]
        for leaf in rr[1]:
            self.root.offspring.append(Node(leaf, offspring=[],
                                            parent=self.root))
            self.nodes.append(self.root.offspring[-1])
            self.ets[0].append(self.root.offspring[-1])
            self.size += 1
        # Add new root to elementary trees if necessary (that is, iff
        # the old root was a leaf)
        if oldRoot.offspring == []:
            self.ets[self.root.order].append(self.root)
        # Update list of distinct parameters
        self.dist_par = list({n.value for n in self.ets[0]
                              if n.value in self.parameters})
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        return self.root

    # -------------------------------------------------------------------------
    def is_root_prunable(self) -> bool:
        """Check whether the root is "prunable".

        The root is prunable when it has two or more offspring and all of
        them except the leftmost one are leaves, so that ``prune_root`` can
        leave the leftmost branch as the new tree. A tree of size 1 is never
        prunable and a tree of size 2 always is.

        Returns
        -------
        bool
            ``True`` if the root can be pruned, ``False`` otherwise.
        """
        if self.size == 1:
            is_prunable = False
        elif self.size == 2:
            is_prunable = True
        else:
            is_prunable = True
            for o in self.root.offspring[1:]:
                if o.offspring != []:
                    is_prunable = False
                    break
        return is_prunable

    # -------------------------------------------------------------------------
    def prune_root(self, update_gof: bool = True,
                   verbose: bool = False) -> list | None:
        """Cut the root and its rightmost leaves, leaving the leftmost branch.

        The rightmost offspring of the root must all be leaves (see
        ``is_root_prunable``); the leftmost one becomes the new root.

        Parameters
        ----------
        update_gof : bool, optional
            Recompute ``sse``, ``bic`` and ``E`` afterwards.
        verbose : bool, optional
            Passed on to the goodness-of-fit computations.

        Returns
        -------
        list or None
            The pruned root as ``[value, [offspring, ...]]``, i.e. in the
            same format as the replacements in ``self.rr_space``, or
            ``None`` if the root is not prunable (the tree is then left
            unchanged and no random draw is made).
        """
        # Check if the root is "prunable" (and return None if not)
        if not self.is_root_prunable():
            return None
        # Let's do it!
        rr = [self.root.value, []]
        self.nodes.remove(self.root)
        try:
            self.ets[len(self.root.offspring)].remove(self.root)
        except ValueError:
            pass
        self.nops[self.root.value] -= 1
        self.size -= 1
        for o in self.root.offspring[1:]:
            rr[1].append(o.value)
            self.nodes.remove(o)
            self.size -= 1
            self.ets[0].remove(o)
        self.root = self.root.offspring[0]
        self.root.parent = None
        # Update list of distinct parameters
        self.dist_par = list({n.value for n in self.ets[0]
                              if n.value in self.parameters})
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        # Done
        return rr

    # -------------------------------------------------------------------------
    def _add_et(self, node: Node, et_order: int | None = None,
                et: list | None = None, update_gof: bool = True,
                verbose: bool = False) -> Node:
        """Add an elementary tree replacing the node, which must be a leaf.

        Parameters
        ----------
        node : Node
            Leaf whose value and offspring are overwritten by the
            elementary tree; it is moved to ``self.ets[et_order]``.
        et_order : int or None, optional
            Order of the elementary tree to draw when ``et`` is not given.
            Ignored (recomputed from ``et``) when ``et`` is given. If both
            are ``None``, the ET is drawn from all elementary trees of
            positive order and its order is then read off the drawn ET.
        et : list or None, optional
            Elementary tree ``[value, [offspring, ...]]`` to insert. When
            it is ``None`` the ET is chosen with ``random.choice``: from
            ``self.et_space[et_order]`` if ``et_order`` is given, otherwise
            from the concatenation of ``self.et_space[o]`` over the entries
            ``o > 0`` of ``self.op_orders``. Exactly one draw is made.
        update_gof : bool, optional
            Recompute ``sse``, ``bic`` and ``E`` afterwards.
        verbose : bool, optional
            Passed on to the goodness-of-fit computations.

        Returns
        -------
        Node
            The node now holding the inserted elementary tree.

        Raises
        ------
        RuntimeError
            If ``node`` is not a leaf, a bare ``raise`` is executed (which
            surfaces as ``RuntimeError`` outside an active exception).
        """
        if node.offspring != []:
            raise
        # If no ET is provided, randomly choose one (of the specified
        # order if given, or totally at random otherwise)
        if et is None:
            if et_order is not None:
                et = choice(self.et_space[et_order])
            else:
                all_ets = [
                    candidate
                    for o in self.op_orders
                    if o > 0
                    for candidate in self.et_space[o]
                ]
                et = choice(all_ets)
                et_order = len(et[1])
        else:
            et_order = len(et[1])
        # Update the node and its offspring
        node.value = et[0]
        try:
            self.nops[node.value] += 1
        except KeyError:
            pass
        node.offspring = [Node(v, parent=node, offspring=[]) for v in et[1]]
        self.ets[et_order].append(node)
        try:
            self.ets[len(node.parent.offspring)].remove(node.parent)
        except ValueError:
            pass
        except AttributeError:
            pass
        # Add the offspring to the list of nodes
        for n in node.offspring:
            self.nodes.append(n)
        # Remove the node from the list of leaves and add its offspring
        self.ets[0].remove(node)
        for o in node.offspring:
            self.ets[0].append(o)
            self.size += 1
        # Update list of distinct parameters
        self.dist_par = list({n.value for n in self.ets[0]
                              if n.value in self.parameters})
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        return node

    # -------------------------------------------------------------------------
    def _del_et(self, node: Node, leaf: str | None = None,
                update_gof: bool = True, verbose: bool = False) -> Node | None:
        """Remove an elementary tree, replacing it by a leaf.

        Parameters
        ----------
        node : Node
            Node holding the elementary tree to remove; its offspring are
            dropped and its value becomes ``leaf``.
        leaf : str or None, optional
            Value of the replacing leaf. If it is ``None``, the leaf is
            chosen with ``random.choice(self.et_space[0])[0]``, i.e. a
            single draw from all variables and parameters.
        update_gof : bool, optional
            Recompute ``sse``, ``bic`` and ``E`` afterwards.
        verbose : bool, optional
            Passed on to the goodness-of-fit computations.

        Returns
        -------
        Node or None
            ``node``, now a leaf, or ``None`` when ``self.size == 1``: the
            tree would be left empty, so the removal is refused and no
            random draw is made.
        """
        if self.size == 1:
            return None
        if leaf is None:
            leaf = choice(self.et_space[0])[0]
        self.nops[node.value] -= 1
        node.value = leaf
        self.ets[len(node.offspring)].remove(node)
        self.ets[0].append(node)
        for o in node.offspring:
            self.ets[0].remove(o)
            self.nodes.remove(o)
            self.size -= 1
        node.offspring = []
        if node.parent is not None:
            is_parent_et = True
            for o in node.parent.offspring:
                if o not in self.ets[0]:
                    is_parent_et = False
                    break
            if is_parent_et:
                self.ets[len(node.parent.offspring)].append(node.parent)
        # Update list of distinct parameters
        self.dist_par = list({n.value for n in self.ets[0]
                              if n.value in self.parameters})
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        return node

    # -------------------------------------------------------------------------
    def et_replace(self, target: Node, new: list, update_gof: bool = True,
                   verbose: bool = False) -> Node | None:
        """Replace one elementary tree by another one, both of any order.

        Three paths are possible. If ``target`` is a leaf, ``new`` is added
        with ``_add_et``. If ``target`` is not a leaf and ``new`` is a leaf,
        the elementary tree of ``target`` is removed with ``_del_et`` using
        ``new[0]`` as the replacing leaf. Otherwise the elementary tree is
        removed first (which draws the value of the intermediate leaf with
        ``random.choice``, i.e. one random draw) and then ``new`` is added.

        Parameters
        ----------
        target : Node
            Node whose elementary tree is replaced.
        new : list
            Replacement elementary tree ``[value, [offspring, ...]]``.
        update_gof : bool, optional
            Recompute ``sse`` and ``bic`` afterwards (``E`` is not updated).
        verbose : bool, optional
            Passed on to the goodness-of-fit computations.

        Returns
        -------
        Node or None
            The node holding the new elementary tree, which is ``target``
            on every path, or ``None`` if the removal itself was refused.
        """
        oini, ofin = len(target.offspring), len(new[1])
        if oini == 0:
            added = self._add_et(target, et=new, update_gof=False,
                                 verbose=verbose)
        else:
            if ofin == 0:
                added = self._del_et(target, leaf=new[0], update_gof=False,
                                     verbose=verbose)
            else:
                self._del_et(target, update_gof=False, verbose=verbose)
                added = self._add_et(target, et=new, update_gof=False,
                                     verbose=verbose)
        # Update goodness of fit measures, if necessary
        if update_gof:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
        # Done
        return added

    # -------------------------------------------------------------------------
    def dE_et(
        self,
        target: Node,
        new: list,
        verbose: bool = False,
    ) -> tuple[float, float, float, dict, int, int]:
        """Return the energy change of replacing one ET with another.

        ``target`` is a ``Node`` and ``new`` is ``[value, [offspring, ...]]``.
        The replacement is applied temporarily and then reverted, so the tree
        is left unchanged. The result is
        ``(dE, dEB, dEP, par_values, nif, nfi)``, where ``nif`` and ``nfi``
        are the numbers of feasible move types before and after the
        replacement. A canonical rejection returns infinite energies.
        """
        dEB, dEP = 0.0, 0.0

        # Some terms of the acceptance (number of possible move types
        # from initial and final configurations), as well as checking
        # if the tree is canonically acceptable.

        # number of possible move types from initial
        nif = sum(
            int(
                len(self.ets[oi]) > 0
                and (self.size + of - oi) <= self.max_size
            )
            for oi, of in self.move_types
        )
        # replace
        old = [target.value, [o.value for o in target.offspring]]
        old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
        old_par_values = deepcopy(self.par_values)
        added = self.et_replace(target, new, update_gof=False, verbose=verbose)
        # number of possible move types from final
        nfi = sum(
            int(
                len(self.ets[oi]) > 0
                and (self.size + of - oi) <= self.max_size
            )
            for oi, of in self.move_types
        )
        # check/update canonical representative
        rep_res = self.update_representative(verbose=verbose)
        if rep_res == -1:
            # this formula is forbidden
            self.et_replace(added, old, update_gof=False, verbose=verbose)
            self.bic, self.sse, self.E = (
                old_bic, deepcopy(old_sse), old_energy
            )
            self.par_values = old_par_values
            return (
                np.inf,
                np.inf,
                np.inf,
                deepcopy(self.par_values),
                nif,
                nfi,
            )
        # leave the whole thing as it was before the back & fore
        self.et_replace(added, old, update_gof=False, verbose=verbose)
        self.bic, self.sse, self.E = (
            old_bic, deepcopy(old_sse), old_energy
        )
        self.par_values = old_par_values
        # Prior: change due to the numbers of each operation
        try:
            dEP -= self.prior_par[f"Nopi_{target.value}"]
        except KeyError:
            pass
        try:
            dEP += self.prior_par[f"Nopi_{new[0]}"]
        except KeyError:
            pass
        try:
            dEP += (
                self.prior_par[f"Nopi2_{target.value}"]
                * (
                    (self.nops[target.value] - 1) ** 2
                    - (self.nops[target.value]) ** 2
                )
            )
        except KeyError:
            pass
        try:
            dEP += (
                self.prior_par[f"Nopi2_{new[0]}"]
                * (
                    (self.nops[new[0]] + 1) ** 2
                    - (self.nops[new[0]]) ** 2
                )
            )
        except KeyError:
            pass

        # Data
        if not list(self.x.values())[0].empty:
            bicOld = self.bic
            sseOld = deepcopy(self.sse)
            par_valuesOld = deepcopy(self.par_values)
            old = [target.value, [o.value for o in target.offspring]]
            # replace
            added = self.et_replace(
                target, new, update_gof=True, verbose=verbose
            )
            bicNew = self.bic
            par_valuesNew = deepcopy(self.par_values)
            # leave the whole thing as it was before the back & fore
            self.et_replace(added, old, update_gof=False, verbose=verbose)
            self.bic = bicOld
            self.sse = deepcopy(sseOld)
            self.par_values = par_valuesOld
            dEB += (bicNew - bicOld) / 2.0
        else:
            par_valuesNew = deepcopy(self.par_values)
        # Done
        try:
            dEB = float(dEB)
            dEP = float(dEP)
            dE = dEB + dEP
        except Exception:
            dEB, dEP, dE = np.inf, np.inf, np.inf
        return dE, dEB, dEP, par_valuesNew, nif, nfi

    # -------------------------------------------------------------------------
    def dE_lr(
        self,
        target: Node,
        new: str,
        verbose: bool = False,
    ) -> tuple[float, float, float, dict | None]:
        """Return the energy change of a long-range move.

        A long-range move replaces the value of ``target`` with ``new``. The
        change is evaluated on a temporary state and then reverted, so the tree
        is left unchanged. The result is ``(dE, dEB, dEP, par_values)``; a
        canonical rejection returns infinite energies and ``None`` parameter
        values.
        """
        dEB, dEP = 0.0, 0.0
        par_valuesNew = deepcopy(self.par_values)

        if target.value != new:

            # Check if the new tree is canonically acceptable.
            old = target.value
            old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
            old_par_values = deepcopy(self.par_values)
            target.value = new
            try:
                self.nops[old] -= 1
                self.nops[new] += 1
            except KeyError:
                pass
            # check/update canonical representative
            rep_res = self.update_representative(verbose=verbose)
            if rep_res == -1:
                # this formula is forbidden
                target.value = old
                try:
                    self.nops[old] += 1
                    self.nops[new] -= 1
                except KeyError:
                    pass
                self.bic, self.sse, self.E = (
                    old_bic, deepcopy(old_sse), old_energy
                )
                self.par_values = old_par_values
                return np.inf, np.inf, np.inf, None
            # leave the whole thing as it was before the back & fore
            target.value = old
            try:
                self.nops[old] += 1
                self.nops[new] -= 1
            except KeyError:
                pass
            self.bic, self.sse, self.E = (
                old_bic, deepcopy(old_sse), old_energy
            )
            self.par_values = old_par_values

            # Prior: change due to the numbers of each operation
            try:
                dEP -= self.prior_par[f"Nopi_{target.value}"]
            except KeyError:
                pass
            try:
                dEP += self.prior_par[f"Nopi_{new}"]
            except KeyError:
                pass
            try:
                dEP += (
                    self.prior_par[f"Nopi2_{target.value}"]
                    * (
                        (self.nops[target.value] - 1) ** 2
                        - (self.nops[target.value]) ** 2
                    )
                )
            except KeyError:
                pass
            try:
                dEP += (
                    self.prior_par[f"Nopi2_{new}"]
                    * (
                        (self.nops[new] + 1) ** 2
                        - (self.nops[new]) ** 2
                    )
                )
            except KeyError:
                pass

            # Data
            if not list(self.x.values())[0].empty:
                bicOld = self.bic
                sseOld = deepcopy(self.sse)
                par_valuesOld = deepcopy(self.par_values)
                old = target.value
                target.value = new
                bicNew = self.get_bic(
                    reset=True, fit=True, verbose=verbose
                )
                par_valuesNew = deepcopy(self.par_values)
                # leave the whole thing as it was before the back & fore
                target.value = old
                self.bic = bicOld
                self.sse = deepcopy(sseOld)
                self.par_values = par_valuesOld
                dEB += (bicNew - bicOld) / 2.0
            else:
                par_valuesNew = deepcopy(self.par_values)

        # Done
        try:
            dEB = float(dEB)
            dEP = float(dEP)
            dE = dEB + dEP
            return dE, dEB, dEP, par_valuesNew
        except Exception:
            return np.inf, np.inf, np.inf, None

    # -------------------------------------------------------------------------
    def dE_rr(
        self,
        rr: list | None = None,
        verbose: bool = False,
    ) -> tuple[float, float, float, dict]:
        """Return the energy change of a root replacement move.

        With ``rr=None`` this evaluates pruning the root. Otherwise it
        evaluates replacing the root with ``rr = [value, [offspring, ...]]``.
        The tree is left unchanged. The result is
        ``(dE, dEB, dEP, par_values)``; an impossible or canonically
        forbidden move returns infinite energies.
        """
        dEB, dEP = 0.0, 0.0

        # Root pruning
        if rr is None:
            if not self.is_root_prunable():
                return np.inf, np.inf, np.inf, self.par_values

            # Check if the new tree is canonically acceptable.
            # replace
            old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
            old_par_values = deepcopy(self.par_values)
            oldrr = [
                self.root.value,
                [o.value for o in self.root.offspring[1:]],
            ]
            self.prune_root(update_gof=False, verbose=verbose)
            # check/update canonical representative
            rep_res = self.update_representative(verbose=verbose)
            if rep_res == -1:
                # this formula is forbidden
                self.replace_root(
                    rr=oldrr, update_gof=False, verbose=verbose
                )
                self.bic, self.sse, self.E = (
                    old_bic, deepcopy(old_sse), old_energy
                )
                self.par_values = old_par_values
                return np.inf, np.inf, np.inf, deepcopy(self.par_values)
            # leave the whole thing as it was before the back & fore
            self.replace_root(rr=oldrr, update_gof=False, verbose=verbose)
            self.bic, self.sse, self.E = (
                old_bic, deepcopy(old_sse), old_energy
            )
            self.par_values = old_par_values

            # Prior: change due to the numbers of each operation
            dEP -= self.prior_par[f"Nopi_{self.root.value}"]
            try:
                dEP += (
                    self.prior_par[f"Nopi2_{self.root.value}"]
                    * (
                        (self.nops[self.root.value] - 1) ** 2
                        - (self.nops[self.root.value]) ** 2
                    )
                )
            except KeyError:
                pass

            # Data correction
            if not list(self.x.values())[0].empty:
                bicOld = self.bic
                sseOld = deepcopy(self.sse)
                par_valuesOld = deepcopy(self.par_values)
                oldrr = [
                    self.root.value,
                    [o.value for o in self.root.offspring[1:]],
                ]
                # replace
                self.prune_root(update_gof=False, verbose=verbose)
                bicNew = self.get_bic(
                    reset=True, fit=True, verbose=verbose
                )
                par_valuesNew = deepcopy(self.par_values)
                # leave the whole thing as it was before the back & fore
                self.replace_root(
                    rr=oldrr, update_gof=False, verbose=verbose
                )
                self.bic = bicOld
                self.sse = deepcopy(sseOld)
                self.par_values = par_valuesOld
                dEB += (bicNew - bicOld) / 2.0
            else:
                par_valuesNew = deepcopy(self.par_values)
            # Done
            try:
                dEB = float(dEB)
                dEP = float(dEP)
                dE = dEB + dEP
            except Exception:
                dEB, dEP, dE = np.inf, np.inf, np.inf
            return dE, dEB, dEP, par_valuesNew

        # Root replacement
        else:
            # Check if the new tree is canonically acceptable.
            # replace
            old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
            old_par_values = deepcopy(self.par_values)
            newroot = self.replace_root(
                rr=rr, update_gof=False, verbose=verbose
            )
            if newroot is None: # Root cannot be replaced (due to max_size)
                return np.inf, np.inf, np.inf, deepcopy(self.par_values)
            # check/update canonical representative
            rep_res = self.update_representative(verbose=verbose)
            if rep_res == -1:
                # this formula is forbidden
                self.prune_root(update_gof=False, verbose=verbose)
                self.bic, self.sse, self.E = (
                    old_bic, deepcopy(old_sse), old_energy
                )
                self.par_values = old_par_values
                return np.inf, np.inf, np.inf, deepcopy(self.par_values)
            # leave the whole thing as it was before the back & fore
            self.prune_root(update_gof=False, verbose=verbose)
            self.bic, self.sse, self.E = (
                old_bic, deepcopy(old_sse), old_energy
            )
            self.par_values = old_par_values

            # Prior: change due to the numbers of each operation
            dEP += self.prior_par[f"Nopi_{rr[0]}"]
            try:
                dEP += (
                    self.prior_par[f"Nopi2_{rr[0]}"]
                    * (
                        (self.nops[rr[0]] + 1) ** 2
                        - (self.nops[rr[0]]) ** 2
                    )
                )
            except KeyError:
                pass

            # Data
            if not list(self.x.values())[0].empty:
                bicOld = self.bic
                sseOld = deepcopy(self.sse)
                par_valuesOld = deepcopy(self.par_values)
                # replace
                newroot = self.replace_root(
                    rr=rr, update_gof=False, verbose=verbose
                )
                if newroot is None:
                    return np.inf, np.inf, np.inf, self.par_values
                bicNew = self.get_bic(
                    reset=True, fit=True, verbose=verbose
                )
                par_valuesNew = deepcopy(self.par_values)
                # leave the whole thing as it was before the back & fore
                self.prune_root(update_gof=False, verbose=verbose)
                self.bic = bicOld
                self.sse = deepcopy(sseOld)
                self.par_values = par_valuesOld
                dEB += (bicNew - bicOld) / 2.0
            else:
                par_valuesNew = deepcopy(self.par_values)
            # Done
            try:
                dEB = float(dEB)
                dEP = float(dEP)
                dE = dEB + dEP
            except Exception:
                dEB, dEP, dE = np.inf, np.inf, np.inf
            return dE, dEB, dEP, par_valuesNew
