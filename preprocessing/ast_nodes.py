# filepath: preprocessing/ast_nodes.py
"""Nodos del AST y utilidades de representación (sin ACG aquí)."""
from __future__ import annotations


class ParseNode:
    def __str__(self):
        return self.to_formula()

    def to_formula(self):
        return ""

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}{self.__class__.__name__}\n"

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __hash__(self):
        def make_hashable(value):
            if isinstance(value, list):
                return tuple(value)
            elif isinstance(value, dict):
                return tuple(sorted(value.items()))
            elif isinstance(value, set):
                return frozenset(value)
            elif isinstance(value, ParseNode):
                return hash(value)
            return value

        items = tuple(sorted((k, make_hashable(v)) for k, v in self.__dict__.items()))
        return hash((self.__class__.__name__, items))


class T(ParseNode):
    def __eq__(self, other):
        return isinstance(other, T)

    def __hash__(self):
        return hash("T")

    def to_formula(self):
        return "⊤"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}T"


class F(ParseNode):
    def __eq__(self, other):
        return isinstance(other, F)

    def __hash__(self):
        return hash("F")

    def to_formula(self):
        return "⊥"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}F"


class Var(ParseNode):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(("Var", self.name))

    def to_formula(self):
        return self.name

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Var('{self.name}')"


class And(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        lhs_str = f"({self.lhs})" if isinstance(self.lhs, (And, Or)) else str(self.lhs)
        rhs_str = f"({self.rhs})" if isinstance(self.rhs, (And, Or)) else str(self.rhs)
        return f"{lhs_str} ∧ {rhs_str}"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}And\n{self.lhs.to_tree(level+1)}\n{self.rhs.to_tree(level+1)}"


class Or(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        lhs_str = f"({self.lhs})" if isinstance(self.lhs, (And, Or)) else str(self.lhs)
        rhs_str = f"({self.rhs})" if isinstance(self.rhs, (And, Or)) else str(self.rhs)
        return f"{lhs_str} ∨ {rhs_str}"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Or\n{self.lhs.to_tree(level+1)}\n{self.rhs.to_tree(level+1)}"


class Not(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"(¬ {self.sub})"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Not\n{self.sub.to_tree(level+1)}"


class Next(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"◯ {self.sub}" if isinstance(self.sub, Var) else f"◯ ({self.sub})"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Next\n{self.sub.to_tree(level+1)}"


class Until(ParseNode):
    def __init__(self, lhs, rhs, generated_from_eventually=False):
        self.lhs = lhs
        self.rhs = rhs
        self.generated_from_eventually = generated_from_eventually

    def to_formula(self):
        return f"({self.lhs} U {self.rhs})"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Until\n{self.lhs.to_tree(level+1)}\n{self.rhs.to_tree(level+1)}"


class Release(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} R {self.rhs})"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Release\n{self.lhs.to_tree(level+1)}\n{self.rhs.to_tree(level+1)}"


class Globally(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"□ {self.sub}" if isinstance(self.sub, Var) else f"□ ({self.sub})"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Globally\n{self.sub.to_tree(level+1)}"


class Eventually(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"◇ {self.sub}" if isinstance(self.sub, Var) else f"◇ ({self.sub})"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Eventually\n{self.sub.to_tree(level+1)}"


class Implies(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} -> {self.rhs})"

    def __str__(self):
        return self.to_formula()


class Iff(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} <-> {self.rhs})"

    def __str__(self):
        return self.to_formula()


class Modality(ParseNode):
    def __init__(self, agents, sub):
        self.agents = agents
        self.sub = sub

    def to_formula(self):
        agents_str = ", ".join(self.agents)
        return f"<{agents_str}> {self.sub}"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        agents_str = ", ".join(self.agents)
        return f"{indent}Modality ({agents_str})\n{self.sub.to_tree(level+1)}"


class DualModality(ParseNode):
    def __init__(self, agents, sub):
        self.agents = agents
        self.sub = sub

    def to_formula(self):
        agents_str = ", ".join(self.agents)
        return f"[{agents_str}] {self.sub}"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        agents_str = ", ".join(self.agents)
        return f"{indent}DualModality ({agents_str})\n{self.sub.to_tree(level+1)}"

    def __eq__(self, other):
        return isinstance(other, DualModality) and self.agents == other.agents and self.sub == other.sub

    def __hash__(self):
        return hash(("DualModality", frozenset(self.agents), self.sub))


class Top:
    def __eq__(self, other):
        return isinstance(other, Top)

    def __hash__(self):
        return hash("Top")

    def __str__(self):
        return "⊤"

    def to_formula(self):
        return "⊤"


class Bottom:
    def __eq__(self, other):
        return isinstance(other, Bottom)

    def __hash__(self):
        return hash("Bottom")

    def __str__(self):
        return "⊥"

    def to_formula(self):
        return "⊥"


class Conj(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} ∧ {self.rhs})"

    def __str__(self):
        return self.to_formula()

    def __eq__(self, other):
        return isinstance(other, Conj) and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        return hash(('Conj', self.lhs, self.rhs))


class Disj(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} ∨ {self.rhs})"

    def __str__(self):
        return self.to_formula()

    def __eq__(self, other):
        return isinstance(other, Disj) and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        return hash(('Disj', self.lhs, self.rhs))


__all__ = [
    'ParseNode', 'T', 'F', 'Var', 'And', 'Or', 'Not', 'Next', 'Until', 'Release',
    'Globally', 'Eventually', 'Implies', 'Iff', 'Modality', 'DualModality',
    'Top', 'Bottom', 'Conj', 'Disj',
]
