class ParseNode:
    def __str__(self):
        return self.to_formula()

    def to_formula(self):
        return ""

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}{self.__class__.__name__}\n"
    
class T(ParseNode):    
    def to_formula(self):
        return "⊤"

    def __str__(self):
        return self.to_formula()
    
    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}T"

class Var(ParseNode):
    def __init__(self, name):
        self.name = name

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
        lhs_str = f"({self.lhs})" if isinstance(self.lhs, (And,Or)) else str(self.lhs)
        rhs_str = f"({self.rhs})" if isinstance(self.rhs, (And,Or)) else str(self.rhs)
        return f"{lhs_str} and {rhs_str}"
    
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
        lhs_str = f"({self.lhs})" if isinstance(self.lhs, (And,Or)) else str(self.lhs)
        rhs_str = f"({self.rhs})" if isinstance(self.rhs, (And,Or)) else str(self.rhs)
        return f"{lhs_str} or {rhs_str}"

    def __str__(self):
        return self.to_formula()
    
    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Or\n{self.lhs.to_tree(level+1)}\n{self.rhs.to_tree(level+1)}"

class Not(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"(not {self.sub})"

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
    def __init__(self, lhs, rhs,generated_from_eventually=False):
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

class ACG:
    def __init__(self):
        self.propositions = set()  
        self.states = set()  
        self.initial_state = None  
        self.transitions = {}  
        self.final_states = set()  
        self.alphabet = set()  

    def generate_alphabet(self):
        self.alphabet = set(frozenset(s) for s in chain.from_iterable(combinations(self.propositions, r) for r in range(len(self.propositions) + 1)))

    def add_proposition(self, proposition):
        self.propositions.add(proposition)
        self.generate_alphabet()  

    def add_state(self, state):
        self.states.add(state)

    def add_initial_state(self, state):
        self.add_state(state)
        self.initial_state = state

    def add_final_state(self, state):
        self.add_state(state)
        self.final_states.add(state)

    def add_transition(self, state_from, input_symbol, transition_formula):
        if state_from not in self.states:
            self.add_state(state_from)
        if input_symbol not in self.alphabet:
            raise ValueError(f"Input symbol {input_symbol} is not in the alphabet (2^AP).")

        self.transitions[(state_from, input_symbol)] = transition_formula

    def get_transition(self, state, input_symbol):
        return self.transitions.get((state, input_symbol), "∅")

    def __str__(self):
        state_formulas = sorted(str(state) for state in self.states)

        alphabet_str = sorted(["{" + ", ".join(sorted(a)) + "}" if a else "{}" for a in self.alphabet])

        formatted_transitions = []
        for (state, sigma) in self.transitions:
            sigma_str = "{" + ", ".join(sorted(sigma)) + "}" if sigma else "{}"
            transition_str = self.get_transition(state, sigma)
            formatted_transitions.append(f"    δ({state}, {sigma_str}) → {transition_str}")

        return (
            f"ACG(\n"
            f"  Alphabet: {alphabet_str},\n"
            f"  States: {state_formulas},\n"
            f"  Initial State: {self.initial_state},\n"
            f"  Transitions:\n" +
            "\n".join(formatted_transitions) +
            f"\n)"
    )

class Top:
    def __str__(self):
        return "⊤"
    
class Bottom:
    def __str__(self):
        return "⊥"

class Conj(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} AND {self.rhs})"

    def __str__(self):
        return self.to_formula()

class Disj(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} OR {self.rhs})"

    def __str__(self):
        return self.to_formula()

class UniversalAtom:
    def __init__(self, state, agents):
        self.state = state
        self.agents = frozenset(agents)

    def __str__(self):
        agents_str = ", ".join(sorted(self.agents)) if self.agents else "∅"
        return f"( {self.state}, □, {{{agents_str}}} )"

class ExistentialAtom:
    def __init__(self, state, agents):
        self.state = state
        self.agents = frozenset(agents)

    def __str__(self):
        agents_str = ", ".join(sorted(self.agents)) if self.agents else "∅"
        return f"( {self.state}, ◇, {{{agents_str}}} )"

class EpsilonAtom:
    def __init__(self, state):
        self.state = state

    def __str__(self):
        return f"( {self.state}, ε )"
 
