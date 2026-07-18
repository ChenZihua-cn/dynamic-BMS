# core/tree_base.py

from .node import Node
from .constants import OPS
from random import choice
from itertools import product, permutations
from copy import deepcopy
import sys
import numpy as np
import pandas as pd
from sympy import sympify, latex


class TreeBase:
    """ The Tree class - base structure and initialization."""

    # -------------------------------------------------------------------------
    def __init__(self, ops=OPS, variables=['x'], parameters=['a'],
                 prior_par={}, x=None, y=None, BT=1., PT=1.,
                 max_size=50,
                 root_value=None, from_string=None,
                 fixed_term=None, fixed_term_op='*', extra_variables=[],
                 dimensions=None, constitutions=None, parliaments=None,
                 ):
        # The variables and parameters
        self.variables = variables
        self.parameters = [p if p.startswith('_') and p.endswith('_')
                           else '_%s_' % p
                           for p in parameters]
        # The root
        if root_value == None:
            self.root = Node(choice(self.variables+self.parameters),
                             offspring=[],
                             parent=None)
        else:
            self.root = Node(root_value,
                             offspring=[],
                             parent=None)
        # The possible operations
        self.ops = ops
        # The possible orders of the operations, move types, and move
        # type probabilities
        self.op_orders = list(set([0] + [n for n in list(ops.values())]))
        self.move_types = [p for p in permutations(self.op_orders, 2)]
        # Elementary trees (including leaves), indexed by order
        self.ets = dict([(o, []) for o in self.op_orders])
        self.ets[0] = [self.root]
        # Distinct parameters used
        self.dist_par = list(set([n.value for n in self.ets[0]
                                  if n.value in self.parameters]))
        self.n_dist_par = len(self.dist_par)
        # Nodes of the tree (operations + leaves)
        self.nodes = [self.root]
        # Tree size and other properties of the model
        self.size = 1
        self.max_size = max_size
        # Space of all possible leaves and elementary trees
        # (dict. indexed by order)
        self.et_space = self.build_et_space()
        # Space of all possible root replacement trees
        self.rr_space = self.build_rr_space()
        self.num_rr = len(self.rr_space)
        # Number of operations of each type
        self.nops = dict([[o, 0] for o in ops])
        # The parameters of the prior propability (default: 5 everywhere)
        if prior_par == {}:
            self.prior_par = dict([('Nopi_%s' % t, 10.) for t in self.ops])
        else:
            self.prior_par = prior_par
        # The datasets
        if x is None:
            self.x = {'d0' : pd.DataFrame()}
            self.y = {'d0' : pd.Series(dtype=float)}
        elif isinstance(x, pd.DataFrame):
            self.x = {'d0' : x}
            self.y = {'d0' : y}
        elif isinstance(x, dict):
            self.x = x
            if y is None:
                self.y = dict([(ds, pd.Series(dtype=float)) for ds in self.x])
            else:
                self.y = y
        else:
            raise TypeError('x must be either a dict or a pandas.DataFrame')
        # To specify a fixed-form prefactor. This must have proper BMS
        # syntax, and will be multiplied by whatever the actual tree
        # is. If necessary, extra variables can be added. These will
        # be treated as variables for parameter fitting, prediction,
        # and so on, but will not be used in the actual tree. Of
        # course, the fixed_term may include regular
        # (i.e. non-extra) variables, as well.
        self.fixed_term = fixed_term
        self.fixed_term_op = fixed_term_op
        self.extra_variables = extra_variables
        if fixed_term == None:
            self.fixed_parameters = []
        else:
            self.fixed_parameters = [
                p for p in
                fixed_term.replace('(', ' ').replace(')', ' ').split()
                if p.startswith('_') and p.endswith('_')
                and p not in self.parameters
            ]
        # The values of the model parameters (one set of values for each dataset)
        self.par_values = dict([
            (ds, deepcopy(dict([(p, 1.) for p in
                                self.parameters + self.fixed_parameters])))
            for ds in self.x
        ])
        # BIC and prior temperature
        self.BT = float(BT)
        self.PT = float(PT)
        # Dimension declarations and expert system
        self.dimensions = dimensions or {}
        if hasattr(self, '_init_experts'):
            self._init_experts(constitutions, parliaments)
        # Build from string
        if from_string != None:
            self.build_from_string(from_string)
        # For fast fitting, we save past successful fits to this formula
        self.fit_par = {}
        # Goodness of fit measures
        self.sse = self.get_sse()
        self.bic = self.get_bic()
        self.E, self.EB, self.EP = self.get_energy()
        # Clear cache if the expression was created from string
        if from_string != None:
            self.fit_par = {}
        # To control formula degeneracy (i.e. different trees that
        # correspond to the same cannoninal formula), we store the
        # representative tree for each canonical formula
        self.representative = {}
        self.representative[self.canonical()] = (
            str(self), self.E, deepcopy(self.par_values)
        )
        # Done
        return

    # -------------------------------------------------------------------------
    def __repr__(self):
        if self.fixed_term == None:
            return self.root.pr()
        else:
            return '(%s %s %s)' % (self.root.pr(), self.fixed_term_op, self.fixed_term)

    # -------------------------------------------------------------------------
    def pr(self, show_pow=True):
        if self.fixed_term == None:
            return self.root.pr(show_pow=show_pow)
        else:
            return '(%s %s %s)' % (self.root.pr(show_pow=show_pow),
                                   self.fixed_term_op, self.fixed_term)

    # -------------------------------------------------------------------------
    # NEED TO DOUBLE CHECK THIS METHOD!!!!
    def set_par_values(self, par_values):
        if set(par_values.keys()) == set(self.x.keys()):
            # Parameter sets match the data: simply overwrite
            self.par_values = deepcopy(par_values)
        elif (set(self.parameters + self.fixed_parameters) <= set(par_values.keys()) and
              len(list(self.x.keys())) == 1):
            # The par_values provided are enough to specify all model
            # parameters (self.parameters is a subset of
            # par_values.keys()) and there is only one dataset: use
            # the data to specify the dataset label.
            self.par_values = {list(self.x.keys())[0] : deepcopy(par_values)}
        else:
            raise ValueError('Parameter datasets do not match x/y datasets.')
        # Recalculate goodness of fit measures
        self.bic = self.get_bic(fit=False, reset=True)
        self.get_energy(bic=False, reset=True)
        # Save fitted parameters
        self.fit_par[str(self)] = self.par_values

        return

    # -------------------------------------------------------------------------
    def canonical(self, verbose=False):
        """Return the canonical form of a tree.

        """
        try:
            cansp = sympify(str(self).replace(' ', ''))
            can = str(cansp)
            ps = list([str(s) for s in cansp.free_symbols])
            positions = []
            for p in ps:
                if p.startswith('_') and p.endswith('_'):
                    positions.append((can.find(p), p))
            positions.sort()
            pcount = 1
            for pos, p in positions:
                can = can.replace(p, 'c%d' % pcount)
                pcount += 1
        except:
            if verbose:
                print('WARNING: Could not get canonical form for', \
                    str(self), '(using full form!)', file=sys.stderr)
            can = str(self)
        return can.replace(' ', '')

    # -------------------------------------------------------------------------
    def latex(self):
        return latex(sympify(self.canonical()))

    # -------------------------------------------------------------------------
    def __parse_recursive(self, string, variables=None, parameters=None,
                          vpreturn=False):
        """ Parse a string obtained from Tree.__repr__() so that it can be used by build_from_string.

        """
        if variables == None:
            variables = []
        if parameters == None:
            parameters = []
        # Leaf
        if '(' not in string:
            if string.startswith('_'):
                if string not in parameters:
                    parameters.append(string)
            else:
                if string not in variables:
                    variables.append(string)
            rval = [string, []]
        # Not a leaf: parse the expression
        else:
            ready = False
            while not ready:
                nterm, terms, nopenpar, op, opactive = 0, [''], 0, '', True
                for c in string:
                    if opactive and c == '(':
                        opactive = False
                    if opactive and c != ' ':
                        op += c
                    elif opactive and c == ' ':
                        opactive = False
                        nterm += 1
                        terms.append('')
                    elif nopenpar == 1 and c == ' ':
                        opactive = True
                    elif c == '(':
                        if nopenpar > 0:
                            terms[nterm] += c
                        nopenpar += 1
                    elif c == ')':
                        nopenpar -= 1
                        if nopenpar > 0:
                            terms[nterm] += c
                    else:
                        terms[nterm] += c
                if op != '':
                    ready = True
                    rval = [op, [self.__parse_recursive(t,
                                                        variables=variables,
                                                        parameters=parameters)
                                 for t in terms]]
                else:
                    if string[0] == '(' and string[-1] == ')':
                        string = string[1:-1]
                    else:
                        raise
        # Done parsing
        if vpreturn:
            return rval, parameters, variables
        else:
            return rval

    # -------------------------------------------------------------------------
    def __grow_tree(self, target, value, offspring):
        """Auxiliary function used to recursively grow a tree from an expression parsed with __parse_recursive().

        """
        try:
            tmpoff = [self.variables[0] for i in range(len(offspring))]
        except IndexError:
            tmpoff = [self.parameters[0] for i in range(len(offspring))]
        self.et_replace(target, [value, tmpoff], verbose=False)
        for i in range(len(offspring)):
            self.__grow_tree(target.offspring[i],
                             offspring[i][0], offspring[i][1])
        return

    # -------------------------------------------------------------------------
    def build_from_string(self, string, verbose=False):
        """Build the tree from an expression formatted according to Tree.__repr__().

        """
        tlist, parameters, variables = self.__parse_recursive(string,
                                                              vpreturn=True)
        self.__init__(ops=self.ops, prior_par=self.prior_par,
                      x=self.x, y=self.y, BT=self.BT, PT=self.PT,
                      max_size=self.max_size,
                      parameters=parameters, variables=variables,
                      dimensions=self.dimensions,
                      constitutions=self.constitutions
                          if hasattr(self, 'constitutions') else None,
                      parliaments=self.parliaments
                          if hasattr(self, 'parliaments') else None)
        self.__grow_tree(self.root, tlist[0], tlist[1])
        # Constitution verification (one-shot after full tree construction)
        if hasattr(self, 'check_constitution') and \
           not self.check_constitution():
            raise ValueError(
                f"Expression '{string}' violates dimensional constraints."
            )
        self.get_sse(verbose=verbose)
        self.get_bic(verbose=verbose)
        self.fit_par = {}  # Forget all values fitted so far
        return

    # -------------------------------------------------------------------------
    def build_et_space(self):
        """Build the space of possible elementary trees, which is a dictionary indexed by the order of the elementary tree.

        """
        et_space = dict([(o, []) for o in self.op_orders])
        et_space[0] = [[x, []] for x in self.variables + self.parameters]
        for op, noff in list(self.ops.items()):
            for vs in product(et_space[0], repeat=noff):
                et_space[noff].append([op, [v[0] for v in vs]])
        return et_space

    # -------------------------------------------------------------------------
    def build_rr_space(self):
        """Build the space of possible trees for the root replacement move.

        """
        rr_space = []
        for op, noff in list(self.ops.items()):
            if noff == 1:
                rr_space.append([op, []])
            else:
                for vs in product(self.et_space[0], repeat=(noff-1)):
                    rr_space.append([op, [v[0] for v in vs]])
        return rr_space
