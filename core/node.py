# core/node.py

class Node:
    """ The Node class."""
    def __init__(self, value, parent=None, offspring=[]):
        self.parent = parent
        self.offspring = offspring
        self.value = value
        self.order = len(self.offspring)
        return

    def pr(self, show_pow: bool = False) -> str:
        """Render this subtree as an expression string.

        ``show_pow`` keeps pow2/pow3 in function form; otherwise they are
        written as ``** 2`` and ``** 3``.
        """
        match self.offspring:
            case []:
                return str(self.value)
            case [left, right]:
                return f"({left.pr(show_pow)} {self.value} {right.pr(show_pow)})"
            case children if not show_pow and self.value in {"pow2", "pow3"}:
                return f"({children[0].pr(show_pow)} ** {self.value[-1]})"
            case children:
                args = ",".join(child.pr(show_pow) for child in children)
                return f"{self.value}({args})"
