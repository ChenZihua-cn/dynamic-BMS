# core/proposal.py

from .node import Node
from copy import deepcopy
import numpy as np
import sys
from random import choice


class ProposalMixin:
    """Tree manipulation and proposal distribution moves."""

    # -------------------------------------------------------------------------
    def replace_root(self, rr=None, update_gof=True, verbose=False):
        """Replace the root with a "root replacement" rr (if provided; otherwise choose one at random from self.rr_space). Returns the new root if the move was possible, and None if not (because the replacement would lead to a tree larger than self.max_size."

        """
        # If no RR is provided, randomly choose one
        if rr == None:
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
        self.dist_par = list(set([n.value for n in self.ets[0]
                                  if n.value in self.parameters]))
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof == True:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        return self.root

    # -------------------------------------------------------------------------
    def is_root_prunable(self):
        """ Check if the root is "prunable".

        """
        if self.size == 1:
            isPrunable = False
        elif self.size == 2:
            isPrunable = True
        else:
            isPrunable = True
            for o in self.root.offspring[1:]:
                if o.offspring != []:
                    isPrunable = False
                    break
        return isPrunable

    # -------------------------------------------------------------------------
    def prune_root(self, update_gof=True, verbose=False):
        """Cut the root and its rightmost leaves (provided they are, indeed, leaves), leaving the leftmost branch as the new tree. Returns the pruned root with the same format as the replacement roots in self.rr_space (or None if pruning was impossible).

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
        self.dist_par = list(set([n.value for n in self.ets[0]
                                  if n.value in self.parameters]))
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof == True:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        # Done
        return rr

    # -------------------------------------------------------------------------
    def _add_et(self, node, et_order=None, et=None, update_gof=True,
                verbose=False):
        """Add an elementary tree replacing the node, which must be a leaf.

        """
        if node.offspring != []:
            raise
        # If no ET is provided, randomly choose one (of the specified
        # order if given, or totally at random otherwise)
        if et == None:
            if et_order != None:
                et = choice(self.et_space[et_order])
            else:
                all_ets = []
                for o in [o for o in self.op_orders if o > 0]:
                    all_ets += self.et_space[o]
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
        self.dist_par = list(set([n.value for n in self.ets[0]
                                  if n.value in self.parameters]))
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof == True:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        return node

    # -------------------------------------------------------------------------
    def _del_et(self, node, leaf=None, update_gof=True, verbose=False):
        """Remove an elementary tree, replacing it by a leaf.

        """
        if self.size == 1:
            return None
        if leaf == None:
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
        if (node.parent != None):
            is_parent_et = True
            for o in node.parent.offspring:
                if o not in self.ets[0]:
                    is_parent_et = False
                    break
            if is_parent_et == True:
                self.ets[len(node.parent.offspring)].append(node.parent)
        # Update list of distinct parameters
        self.dist_par = list(set([n.value for n in self.ets[0]
                                  if n.value in self.parameters]))
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof == True:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        return node

    # -------------------------------------------------------------------------
    def et_replace(self, target, new, update_gof=True, verbose=False):
        """Replace one ET by another one, both of arbitrary order. target is a
Node and new is a tuple [node_value, [list, of, offspring, values]]

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
        if update_gof == True:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
        # Done
        return added

    # -------------------------------------------------------------------------
    def dE_et(self, target, new, verbose=False):
        """Calculate the energy change associated to the replacement of one ET
by another, both of arbitrary order. "target" is a Node() and "new" is
a tuple [node_value, [list, of, offspring, values]].

        """
        dEB, dEP = 0.0, 0.0

        # Some terms of the acceptance (number of possible move types
        # from initial and final configurations), as well as checking
        # if the tree is canonically acceptable.

        # number of possible move types from initial
        nif = sum([int(len(self.ets[oi]) > 0 and
                       (self.size + of - oi) <= self.max_size)
                   for oi, of in self.move_types])
        # replace
        old = [target.value, [o.value for o in target.offspring]]
        old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
        old_par_values = deepcopy(self.par_values)
        added = self.et_replace(target, new, update_gof=False, verbose=verbose)
        # number of possible move types from final
        nfi = sum([int(len(self.ets[oi]) > 0 and
                       (self.size + of - oi) <= self.max_size)
                   for oi, of in self.move_types])
        # === merged gate: constitution + canonical ===
        gate_failed = False
        if hasattr(self, 'check_constitution') and \
           not self.check_constitution():
            gate_failed = True
        elif self.update_representative(verbose=verbose) == -1:
            gate_failed = True

        if gate_failed:
            # this formula is forbidden
            self.et_replace(added, old, update_gof=False, verbose=verbose)
            self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
            self.par_values = old_par_values
            return np.inf, np.inf, np.inf, deepcopy(self.par_values), nif, nfi
        # leave the whole thing as it was before the back & fore
        self.et_replace(added, old, update_gof=False, verbose=verbose)
        self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
        self.par_values = old_par_values
        # Prior: change due to the numbers of each operation
        try:
            dEP -= self.prior_par['Nopi_%s' % target.value]
        except KeyError:
            pass
        try:
            dEP += self.prior_par['Nopi_%s' % new[0]]
        except KeyError:
            pass
        try:
            dEP += (self.prior_par['Nopi2_%s' % target.value] *
                   ((self.nops[target.value] - 1)**2 -
                    (self.nops[target.value])**2))
        except KeyError:
            pass
        try:
            dEP += (self.prior_par['Nopi2_%s' % new[0]] *
                   ((self.nops[new[0]] + 1)**2 -
                    (self.nops[new[0]])**2))
        except KeyError:
            pass

        # Data
        if not list(self.x.values())[0].empty:
            bicOld = self.bic
            sseOld = deepcopy(self.sse)
            par_valuesOld = deepcopy(self.par_values)
            old = [target.value, [o.value for o in target.offspring]]
            # replace
            added = self.et_replace(target, new, update_gof=True,
                                    verbose=verbose)
            bicNew = self.bic
            par_valuesNew = deepcopy(self.par_values)
            # leave the whole thing as it was before the back & fore
            self.et_replace(added, old, update_gof=False, verbose=verbose)
            self.bic = bicOld
            self.sse = deepcopy(sseOld)
            self.par_values = par_valuesOld
            dEB += (bicNew - bicOld) / 2.
        else:
            par_valuesNew = deepcopy(self.par_values)
        # Done
        try:
            dEB = float(dEB)
            dEP = float(dEP)
            dE = dEB + dEP
        except:
            dEB, dEP, dE = np.inf, np.inf, np.inf
        return dE, dEB, dEP, par_valuesNew, nif, nfi

    # -------------------------------------------------------------------------
    def dE_lr(self, target, new, verbose=False):
        """Calculate the energy change associated to a long-range move (the replacement of the value of a node. "target" is a Node() and "new" is a node_value.
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
            # === merged gate: constitution + canonical ===
            gate_failed = False
            if hasattr(self, 'check_constitution') and \
               not self.check_constitution():
                gate_failed = True
            elif self.update_representative(verbose=verbose) == -1:
                gate_failed = True

            if gate_failed:
                # this formula is forbidden
                target.value = old
                try:
                    self.nops[old] += 1
                    self.nops[new] -= 1
                except KeyError:
                    pass
                self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
                self.par_values = old_par_values
                return np.inf, np.inf, np.inf, None
            # leave the whole thing as it was before the back & fore
            target.value = old
            try:
                self.nops[old] += 1
                self.nops[new] -= 1
            except KeyError:
                pass
            self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
            self.par_values = old_par_values

            # Prior: change due to the numbers of each operation
            try:
                dEP -= self.prior_par['Nopi_%s' % target.value]
            except KeyError:
                pass
            try:
                dEP += self.prior_par['Nopi_%s' % new]
            except KeyError:
                pass
            try:
                dEP += (self.prior_par['Nopi2_%s' % target.value] *
                       ((self.nops[target.value] - 1)**2 -
                        (self.nops[target.value])**2))
            except KeyError:
                pass
            try:
                dEP += (self.prior_par['Nopi2_%s' % new] *
                       ((self.nops[new] + 1)**2 -
                        (self.nops[new])**2))
            except KeyError:
                pass

            # Data
            if not list(self.x.values())[0].empty:
                bicOld = self.bic
                sseOld = deepcopy(self.sse)
                par_valuesOld = deepcopy(self.par_values)
                old = target.value
                target.value = new
                bicNew = self.get_bic(reset=True, fit=True, verbose=verbose)
                par_valuesNew = deepcopy(self.par_values)
                # leave the whole thing as it was before the back & fore
                target.value = old
                self.bic = bicOld
                self.sse = deepcopy(sseOld)
                self.par_values = par_valuesOld
                dEB += (bicNew - bicOld) / 2.
            else:
                par_valuesNew = deepcopy(self.par_values)

        # Done
        try:
            dEB = float(dEB)
            dEP = float(dEP)
            dE = dEB + dEP
            return dE, dEB, dEP, par_valuesNew
        except:
            return np.inf, np.inf, np.inf, None

    # -------------------------------------------------------------------------
    def dE_rr(self, rr=None, verbose=False):
        """Calculate the energy change associated to a root replacement move. If rr==None, then it returns the energy change associated to pruning the root; otherwise, it returns the dE associated to adding the root replacement "rr".

        """
        dEB, dEP = 0.0, 0.0

        # Root pruning
        if rr == None:
            if not self.is_root_prunable():
                return np.inf, np.inf, np.inf, self.par_values

            # Check if the new tree is canonically acceptable.
            # replace
            old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
            old_par_values = deepcopy(self.par_values)
            oldrr = [self.root.value,
                     [o.value for o in self.root.offspring[1:]]]
            self.prune_root(update_gof=False, verbose=verbose)
            # === merged gate: constitution + canonical ===
            gate_failed = False
            if hasattr(self, 'check_constitution') and \
               not self.check_constitution():
                gate_failed = True
            elif self.update_representative(verbose=verbose) == -1:
                gate_failed = True

            if gate_failed:
                # this formula is forbidden
                self.replace_root(rr=oldrr, update_gof=False, verbose=verbose)
                self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
                self.par_values = old_par_values
                return np.inf, np.inf, np.inf, deepcopy(self.par_values)
            # leave the whole thing as it was before the back & fore
            self.replace_root(rr=oldrr, update_gof=False, verbose=verbose)
            self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
            self.par_values = old_par_values

            # Prior: change due to the numbers of each operation
            dEP -= self.prior_par['Nopi_%s' % self.root.value]
            try:
                dEP += (self.prior_par['Nopi2_%s' % self.root.value] *
                        ((self.nops[self.root.value] - 1)**2 -
                         (self.nops[self.root.value])**2))
            except KeyError:
                pass

            # Data correction
            if not list(self.x.values())[0].empty:
                bicOld = self.bic
                sseOld = deepcopy(self.sse)
                par_valuesOld = deepcopy(self.par_values)
                oldrr = [self.root.value,
                         [o.value for o in self.root.offspring[1:]]]
                # replace
                self.prune_root(update_gof=False, verbose=verbose)
                bicNew = self.get_bic(reset=True, fit=True, verbose=verbose)
                par_valuesNew = deepcopy(self.par_values)
                # leave the whole thing as it was before the back & fore
                self.replace_root(rr=oldrr, update_gof=False, verbose=verbose)
                self.bic = bicOld
                self.sse = deepcopy(sseOld)
                self.par_values = par_valuesOld
                dEB += (bicNew - bicOld) / 2.
            else:
                par_valuesNew = deepcopy(self.par_values)
            # Done
            try:
                dEB = float(dEB)
                dEP = float(dEP)
                dE = dEB + dEP
            except:
                dEB, dEP, dE = np.inf, np.inf, np.inf
            return dE, dEB, dEP, par_valuesNew

        # Root replacement
        else:
            # Check if the new tree is canonically acceptable.
            # replace
            old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
            old_par_values = deepcopy(self.par_values)
            newroot = self.replace_root(rr=rr, update_gof=False,
                                        verbose=verbose)
            if newroot == None: # Root cannot be replaced (due to max_size)
                return np.inf, np.inf, np.inf, deepcopy(self.par_values)
            # === merged gate: constitution + canonical ===
            gate_failed = False
            if hasattr(self, 'check_constitution') and \
               not self.check_constitution():
                gate_failed = True
            elif self.update_representative(verbose=verbose) == -1:
                gate_failed = True

            if gate_failed:
                # this formula is forbidden
                self.prune_root(update_gof=False, verbose=verbose)
                self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
                self.par_values = old_par_values
                return np.inf, np.inf, np.inf, deepcopy(self.par_values)
            # leave the whole thing as it was before the back & fore
            self.prune_root(update_gof=False, verbose=verbose)
            self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
            self.par_values = old_par_values

            # Prior: change due to the numbers of each operation
            dEP += self.prior_par['Nopi_%s' % rr[0]]
            try:
                dEP += (self.prior_par['Nopi2_%s' % rr[0]] *
                        ((self.nops[rr[0]] + 1)**2 -
                         (self.nops[rr[0]])**2))
            except KeyError:
                pass

            # Data
            if not list(self.x.values())[0].empty:
                bicOld = self.bic
                sseOld = deepcopy(self.sse)
                par_valuesOld = deepcopy(self.par_values)
                # replace
                newroot = self.replace_root(rr=rr, update_gof=False,
                                            verbose=verbose)
                if newroot == None:
                    return np.inf, np.inf, np.inf, self.par_values
                bicNew = self.get_bic(reset=True, fit=True, verbose=verbose)
                par_valuesNew = deepcopy(self.par_values)
                # leave the whole thing as it was before the back & fore
                self.prune_root(update_gof=False, verbose=verbose)
                self.bic = bicOld
                self.sse = deepcopy(sseOld)
                self.par_values = par_valuesOld
                dEB += (bicNew - bicOld) / 2.
            else:
                par_valuesNew = deepcopy(self.par_values)
            # Done
            try:
                dEB = float(dEB)
                dEP = float(dEP)
                dE = dEB + dEP
            except:
                dEB, dEP, dE = np.inf, np.inf, np.inf
            return dE, dEB, dEP, par_valuesNew
