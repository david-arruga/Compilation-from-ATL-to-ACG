from __future__ import annotations
from itertools import combinations
from preprocessing.ast_nodes import Var, Not

class UniversalAtom:
    def __init__(self, state, agents):
        self.state = state
        self.agents = frozenset(agents)

    def __eq__(self, other):
        return (
            isinstance(other, UniversalAtom) and
            self.state == other.state and
            self.agents == other.agents
        )

    def __hash__(self):
        return hash(("UniversalAtom", self.state, self.agents))

    def __str__(self):
        agents_str = ", ".join(sorted(self.agents)) if self.agents else "∅"
        return f"( {self.state}, □, {{{agents_str}}} )"
    
    def to_formula(self):
        agents_str = ", ".join(sorted(self.agents)) if self.agents else "∅"
        return f"( {self.state}, □, {{{agents_str}}} )"

class ExistentialAtom:
    def __init__(self, state, agents):
        self.state = state
        self.agents = frozenset(agents)

    def __eq__(self, other):
        return (
            isinstance(other, ExistentialAtom) and
            self.state == other.state and
            self.agents == other.agents
        )

    def __hash__(self):
        return hash(("ExistentialAtom", self.state, self.agents))

    def __str__(self):
        agents_str = ", ".join(sorted(self.agents)) if self.agents else "∅"
        return f"( {self.state}, ◇, {{{agents_str}}} )"

    def to_formula(self):
        agents_str = ", ".join(sorted(self.agents)) if self.agents else "∅"
        return f"( {self.state}, ◇, {{{agents_str}}} )"

class EpsilonAtom:
    def __init__(self, state):
        self.state = state

    def __eq__(self, other):
        return isinstance(other, EpsilonAtom) and self.state == other.state

    def __hash__(self):
        return hash(("EpsilonAtom", self.state))

    def __str__(self):
        return f"( {self.state}, ε )"
    
    def to_formula(self):
        return f"( {self.state}, ε )"

class ACG:
    WILDCARD = None

    def __init__(self):
        self.propositions  = set()
        self.states        = set()
        self.initial_state = None
        self.final_states  = set()
        self.alphabet      = set()
        self.transitions   = {}

    def generate_alphabet(self):
        self.alphabet = {
            frozenset(s)
            for r in range(len(self.propositions) + 1)
            for s in combinations(self.propositions, r)
        }

    def add_transition(self, state_from, input_symbol, transition_formula):
        if input_symbol is not self.WILDCARD and input_symbol not in self.alphabet:
            raise ValueError(
                f"Input symbol {input_symbol} is not in the alphabet 2^AP."
            )
        self.transitions[(state_from, input_symbol)] = transition_formula

    def get_transition(self, state, input_symbol):
        try:
            return self.transitions[(state, input_symbol)]
        except KeyError:
            try:
                return self.transitions[(state, self.WILDCARD)]
            except KeyError:
                raise KeyError(
                    f"δ undefined for state={state}, σ={input_symbol}"
                )

    def __str__(self):
        def is_atomic(state):
            return isinstance(state, Var) or (
                isinstance(state, Not) and isinstance(state.sub, Var)
            )

        alpha = sorted(
            ["{" + ", ".join(sorted(a)) + "}" if a else "{}"
             for a in self.alphabet]
        )

        lines = [
            "ACG(",
            f"  Alphabet: {alpha},",
            f"  States: {[str(s) for s in sorted(self.states, key=str)]},",
            f"  Initial State: {self.initial_state},",
            f"  Final States: {[str(s) for s in sorted(self.final_states, key=str)]},",
            "  Transitions:",
        ]

        shown = set()
        for (q, sigma), rhs in self.transitions.items():
            if is_atomic(q):
                sigma_str = "{" + ", ".join(sorted(sigma)) + "}" if sigma else "{}"
            else:
                if q in shown:
                    continue
                shown.add(q)
                sigma_str = "*"

            lines.append(f"    δ({q}, {sigma_str}) → {rhs}")

        lines.append(")")
        return "\n".join(lines)