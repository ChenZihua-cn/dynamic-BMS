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


# `-` is unary in OPS (the sampler only ever builds `- x`), but hand-written and
# corpus formulas use it as binary subtraction.  Accept both at parse time
# without widening the sampler's move space.
_POLYARITY_OPS = {'-': (1, 2)}


def _validate_parse(parsed, ops, string):
    """Check that a parsed expression is structurally buildable.

    ``__parse_recursive`` is a permissive character scanner: a malformed string
    like ``(sqrt L)`` produces nodes whose "operator" is unknown or whose child
    count matches no operator, and those used to surface as a bare ``KeyError``
    from the elementary-tree bookkeeping several frames deeper.
    """
    op, children = parsed
    if not children:
        return
    arity = ops.get(op)
    if arity is None:
        raise ValueError(
            f"cannot build {string!r}: '{op}' is not an operator. Unary ops "
            f"must be written op(arg), e.g. sqrt(L) not (sqrt L)")
    allowed = _POLYARITY_OPS.get(op, (arity,))
    if len(children) not in allowed:
        raise ValueError(
            f"cannot build {string!r}: '{op}' takes {allowed} argument(s) but "
            f"got {len(children)}. Every binary operand needs its own "
            f"parentheses, e.g. (A * (B / C)) not (A * B / C)")
    for child in children:
        _validate_parse(child, ops, string)


class TreeBase:
    """ The Tree class - base structure and initialization."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        ops: dict[str, int] = OPS,
        variables: list[str] = ["x"],
        parameters: list[str] = ["a"],
        prior_par: dict[str, float] = {},
        x: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
        y: pd.Series | dict[str, pd.Series] | None = None,
        BT: float = 1.0,
        PT: float = 1.0,
        max_size: int = 50,
        root_value: str | None = None,
        from_string: str | None = None,
        fixed_term: str | None = None,
        fixed_term_op: str = "*",
        extra_variables: list[str] = [],
        dimensions: dict[str, tuple[int, ...]] | None = None,
        target_dim: tuple[int, ...] | None = None,
        constitutions: list | None = None,
        parliaments: list | None = None,
    ):
        """Initialize a formula tree and its goodness-of-fit state.

        ``ops`` maps operation names to their number of offspring, and
        ``variables``/``parameters`` list the allowed leaves. ``prior_par``
        holds the operation-count prior; when empty, each operation gets the
        default prior. ``max_size`` bounds the tree during moves, while ``BT``
        and ``PT`` set the temperatures of the data and prior terms.

        The data are either a DataFrame (stored under ``d0``) or a mapping
        from dataset labels to DataFrames. ``root_value`` fixes the root,
        otherwise it is drawn at random; ``from_string`` builds the full tree
        from a BMS expression. A fixed prefactor can be supplied through
        ``fixed_term`` and ``fixed_term_op``, with ``extra_variables``
        available to that term only.

        ``dimensions`` declares the dimension vector of each variable,
        ``target_dim`` fixes the dimension of the root (``None`` leaves it
        free), and ``constitutions``/``parliaments`` configure the expert
        gates.
        """
        # The variables and parameters
        self.variables = variables
        self.parameters = [
            p if p.startswith("_") and p.endswith("_") else f"_{p}_"
            for p in parameters
        ]
        # The root
        if root_value is None:
            self.root = Node(
                choice(self.variables + self.parameters),
                offspring=[],
                parent=None,
            )
        else:
            self.root = Node(root_value, offspring=[], parent=None)
        # The possible operations
        self.ops = ops
        # The possible orders of the operations, move types, and move
        # type probabilities
        self.op_orders = list(set([0, *ops.values()]))
        self.move_types = list(permutations(self.op_orders, 2))
        # Elementary trees (including leaves), indexed by order
        self.ets = {order: [] for order in self.op_orders}
        self.ets[0] = [self.root]
        # Distinct parameters used
        self.dist_par = list(
            {
                node.value
                for node in self.ets[0]
                if node.value in self.parameters
            }
        )
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
        self.nops = {op: 0 for op in ops}
        # The parameters of the prior propability (default: 5 everywhere)
        if not prior_par:
            self.prior_par = {f"Nopi_{op}": 10.0 for op in self.ops}
        else:
            self.prior_par = prior_par
        # The datasets
        if x is None:
            self.x = {"d0": pd.DataFrame()}
            self.y = {"d0": pd.Series(dtype=float)}
        elif isinstance(x, pd.DataFrame):
            self.x = {"d0": x}
            self.y = {"d0": y}
        elif isinstance(x, dict):
            self.x = x
            if y is None:
                self.y = {ds: pd.Series(dtype=float) for ds in self.x}
            else:
                self.y = y
        else:
            raise TypeError("x must be either a dict or a pandas.DataFrame")
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
        if fixed_term is None:
            self.fixed_parameters = []
        else:
            self.fixed_parameters = [
                p
                for p in fixed_term.replace("(", " ").replace(")", " ").split()
                if p.startswith("_") and p.endswith("_")
                and p not in self.parameters
            ]
        # The values of the model parameters (one set of values for each dataset)
        self.par_values = {
            ds: deepcopy(
                {p: 1.0 for p in self.parameters + self.fixed_parameters}
            )
            for ds in self.x
        }
        # BIC and prior temperature
        self.BT = float(BT)
        self.PT = float(PT)
        # Dimension declarations and expert system
        self.dimensions = dimensions or {}
        # Root must equal this dimension; None leaves the root dimension free
        self.target_dim = target_dim
        self.parliaments = []  # default; GateMixin._init_gate overwrites
        if hasattr(self, '_init_experts'):
            self._init_experts(constitutions, parliaments)
        # Build from string
        if from_string is not None:
            self.build_from_string(from_string)
        # For fast fitting, we save past successful fits to this formula
        self.fit_par = {}
        # Goodness of fit measures
        self.sse = self.get_sse()
        self.bic = self.get_bic()
        self.E, self.EB, self.EP = self.get_energy()
        # Clear cache if the expression was created from string
        if from_string is not None:
            self.fit_par = {}
        # To control formula degeneracy (i.e. different trees that
        # correspond to the same cannoninal formula), we store the
        # representative tree for each canonical formula
        self.representative = {
            self.canonical(): (str(self), self.E, deepcopy(self.par_values))
        }
        # Done
        return

    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return the tree expression as a BMS string.

        If a fixed prefactor is set, the tree expression is combined with
        ``fixed_term`` using ``fixed_term_op``.
        """
        if self.fixed_term is None:
            return self.root.pr()
        return f"({self.root.pr()} {self.fixed_term_op} {self.fixed_term})"

    # -------------------------------------------------------------------------
    def pr(self, show_pow: bool = True) -> str:
        """Return the tree expression as a BMS string.

        ``show_pow`` is forwarded to the root and keeps pow2/pow3 in function
        form when true. If a fixed prefactor is set, it is combined with the
        tree expression using ``fixed_term_op``.
        """
        if self.fixed_term is None:
            return self.root.pr(show_pow=show_pow)
        return (
            f"({self.root.pr(show_pow=show_pow)} "
            f"{self.fixed_term_op} {self.fixed_term})"
        )

    # -------------------------------------------------------------------------
    # NEED TO DOUBLE CHECK THIS METHOD!!!!
    def set_par_values(
        self,
        par_values: dict[str, float] | dict[str, dict[str, float]],
    ) -> None:
        """Set parameter values and refresh the goodness-of-fit state.

        ``par_values`` is either a mapping from dataset labels to parameter
        dictionaries, or (with a single dataset) a parameter dictionary. The
        BIC and energy are recalculated afterwards, and the values are cached
        under the current expression.

        Raise ``ValueError`` if the provided layout matches neither the
        training datasets nor all model parameters of a single dataset.
        """
        if set(par_values) == set(self.x):
            # Parameter sets match the data: simply overwrite
            self.par_values = deepcopy(par_values)
        elif (
            set(self.parameters + self.fixed_parameters) <= set(par_values)
            and len(self.x) == 1
        ):
            # The par_values provided are enough to specify all model
            # parameters (self.parameters is a subset of
            # par_values.keys()) and there is only one dataset: use
            # the data to specify the dataset label.
            self.par_values = {next(iter(self.x)): deepcopy(par_values)}
        else:
            raise ValueError("Parameter datasets do not match x/y datasets.")
        # Recalculate goodness of fit measures
        self.bic = self.get_bic(fit=False, reset=True)
        self.get_energy(bic=False, reset=True)
        # Save fitted parameters
        self.fit_par[str(self)] = self.par_values

    # -------------------------------------------------------------------------
    def canonical(self, verbose: bool = False) -> str:
        """Return the canonical form of the tree expression.

        Parameters are replaced by ``c1``, ``c2``, ... in order of their first
        appearance. If the expression cannot be parsed, fall back to its full
        string form and, when ``verbose`` is true, report a warning.
        """
        try:
            cansp = sympify(str(self).replace(" ", ""))
            can = str(cansp)
            symbols = [str(symbol) for symbol in cansp.free_symbols]
            positions = []
            for symbol in symbols:
                if symbol.startswith("_") and symbol.endswith("_"):
                    positions.append((can.find(symbol), symbol))
            positions.sort()
            parameter_count = 1
            for _, symbol in positions:
                can = can.replace(symbol, f"c{parameter_count}")
                parameter_count += 1
        except Exception:
            if verbose:
                print(
                    f"WARNING: Could not get canonical form for {self} "
                    "(using full form!)",
                    file=sys.stderr,
                )
            can = str(self)
        return can.replace(" ", "")

    # -------------------------------------------------------------------------
    def latex(self) -> str:
        """Return the tree expression in LaTeX form."""
        return latex(sympify(self.canonical()))

    # -------------------------------------------------------------------------
    def __parse_recursive(
        self,
        string: str,
        variables: list[str] | None = None,
        parameters: list[str] | None = None,
        vpreturn: bool = False,
    ) -> list | tuple[list, list[str], list[str]]:
        """Parse a string obtained from ``Tree.__repr__()`` into a nested list.

        A leaf becomes ``[value, []]`` and an operation becomes
        ``[operation, [parsed offspring, ...]]``. Leaves named like ``_name_``
        are appended to ``parameters`` and all other leaves to ``variables``;
        when the corresponding argument is omitted, a new list is used.

        If ``vpreturn`` is true, return ``(parsed, parameters, variables)``;
        otherwise return only ``parsed``.
        """
        if variables is None:
            variables = []
        if parameters is None:
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
            op, terms = self.__parse_head(string)
            rval = [op, [self.__parse_recursive(t,
                                                variables=variables,
                                                parameters=parameters)
                         for t in terms]]
        # Done parsing
        if vpreturn:
            return rval, parameters, variables
        else:
            return rval

    # -------------------------------------------------------------------------
    @staticmethod
    def __parse_head(string: str) -> tuple[str, list[str]]:
        """Split an expression into its head operation and raw term strings.

        Outer parentheses are stripped until an operation is found; raise
        when the expression contains none.
        """
        while True:
            nterm, terms, nopenpar, op, opactive = 0, [''], 0, '', True
            for char in string:
                match char:
                    case '(':
                        opactive = False
                        if nopenpar > 0:
                            terms[nterm] += char
                        nopenpar += 1
                    case ')' if opactive:
                        op += char
                    case ')':
                        nopenpar -= 1
                        if nopenpar > 0:
                            terms[nterm] += char
                    case ' ' if opactive:
                        opactive = False
                        nterm += 1
                        terms.append('')
                    case ' ' if nopenpar == 1:
                        opactive = True
                    case _ if opactive:
                        op += char
                    case _:
                        terms[nterm] += char
            if op != '':
                return op, terms
            if string[0] == '(' and string[-1] == ')':
                string = string[1:-1]
            else:
                raise

    # -------------------------------------------------------------------------
    def __grow_tree(
        self,
        target: Node,
        value: str,
        offspring: list[list],
    ) -> None:
        """Recursively grow a tree from an expression parsed by ``__parse_recursive``.

        Replace ``target`` with ``value`` and the parsed ``offspring``, then grow
        each child in turn. ``et_replace`` needs a temporary offspring list,
        which is filled with the first variable, or with the first parameter
        when the tree has no variables.
        """
        try:
            tmpoff = [self.variables[0] for _ in range(len(offspring))]
        except IndexError:
            tmpoff = [self.parameters[0] for _ in range(len(offspring))]
        self.et_replace(target, [value, tmpoff], verbose=False)
        for i, subtree in enumerate(offspring):
            self.__grow_tree(target.offspring[i], subtree[0], subtree[1])

    # -------------------------------------------------------------------------
    def build_from_string(self, string: str, verbose: bool = False) -> None:
        """Build the tree from an expression formatted like ``Tree.__repr__()``.

        The expression is parsed into a tree, the tree is reinitialized with
        the variables and parameters found in it, and the goodness-of-fit
        state is recomputed. Cached fits are discarded.
        """
        tlist, parameters, variables = self.__parse_recursive(
            string,
            vpreturn=True,
        )
        _validate_parse(tlist, self.ops, string)
        self.__init__(
            ops=self.ops,
            prior_par=self.prior_par,
            x=self.x,
            y=self.y,
            BT=self.BT,
            PT=self.PT,
            max_size=self.max_size,
            parameters=parameters,
            variables=variables,
            fixed_term=self.fixed_term,
            fixed_term_op=self.fixed_term_op,
            extra_variables=self.extra_variables,
            dimensions=self.dimensions,
            target_dim=self.target_dim,
            constitutions=(
                self.constitutions if hasattr(self, "constitutions") else None
            ),
            parliaments=(
                self.parliaments if hasattr(self, "parliaments") else None
            ),
        )
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

    # -------------------------------------------------------------------------
    def build_et_space(self) -> dict[int, list]:
        """Build the space of possible elementary trees, indexed by order.

        Order 0 holds the leaves (variables and parameters); higher orders
        hold each operation combined with every tuple of lower-order subtrees.
        """
        et_space = {order: [] for order in self.op_orders}
        et_space[0] = [[leaf, []] for leaf in self.variables + self.parameters]
        for op, noff in self.ops.items():
            for subtrees in product(et_space[0], repeat=noff):
                et_space[noff].append(
                    [op, [subtree[0] for subtree in subtrees]]
                )
        return et_space

    # -------------------------------------------------------------------------
    def build_rr_space(self) -> list[list]:
        """Build the space of possible trees for the root replacement move.

        Unary operations get an empty offspring list; higher-order operations
        combine with every tuple of leaves except one fixed slot.
        """
        rr_space = []
        for op, noff in self.ops.items():
            if noff == 1:
                rr_space.append([op, []])
            else:
                for subtrees in product(self.et_space[0], repeat=noff - 1):
                    rr_space.append(
                        [op, [subtree[0] for subtree in subtrees]]
                    )
        return rr_space
