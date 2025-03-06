class ParseNode:
    def __str__(self):
        return self.to_formula()

    def to_formula(self):
        return ""

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}{self.__class__.__name__}\n"

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
        return f"({self.lhs} and {self.rhs})"

    def __str__(self):
        return self.to_formula()

class Or(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} or {self.rhs})"

    def __str__(self):
        return self.to_formula()

class Not(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"(not {self.sub})"

    def __str__(self):
        return self.to_formula()

class Next(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"(⃝ {self.sub})"

    def __str__(self):
        return self.to_formula()

class Until(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} U {self.rhs})"

    def __str__(self):
        return self.to_formula()

class Release(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} R {self.rhs})"

    def __str__(self):
        return self.to_formula()

class Globally(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"(□ {self.sub})"

    def __str__(self):
        return self.to_formula()

class Eventually(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"(◇ {self.sub})"

    def __str__(self):
        return self.to_formula()

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

class ACG:
    def __init__(self, propositions=None, states=None, initial_state=None, transitions=None, final_states=None):
        self.propositions = propositions if propositions else set()  
        self.states = states if states else set()  
        self.initial_state = initial_state  
        self.transitions = transitions if transitions else {}  
        self.final_states = final_states if final_states else set()  
        self.alphabet = self.generate_alphabet()  

    def generate_alphabet(self):
        return set(frozenset(s) for s in chain.from_iterable(combinations(self.propositions, r) for r in range(len(self.propositions) + 1)))

    def add_proposition(self, proposition):
        self.propositions.add(proposition)
        self.alphabet = self.generate_alphabet()

    def add_state(self, state_name):
        if state_name not in self.states:
            self.states.add(state_name)

    def add_initial_state(self, state_name):
        if state_name not in self.states:
            self.add_state(state_name)
        self.initial_state = state_name

    def add_final_state(self, state_name):
        if state_name not in self.states:
            self.add_state(state_name)
        self.final_states.add(state_name)

    def add_transition(self, state_from, input_symbol, state_to, atom_type, agents=None):
        if state_from not in self.states:
            self.add_state(state_from)
        if state_to not in self.states:
            self.add_state(state_to)
        if input_symbol not in self.alphabet:
            raise ValueError(f"Input symbol {input_symbol} is not in the alphabet (2^AP).")
        if (state_from, input_symbol) not in self.transitions:
            self.transitions[(state_from, input_symbol)] = {
                "universal": [], "existential": [], "epsilon": []
            }
        if atom_type in ["universal", "existential"]:
            if agents is None:
                raise ValueError(f"Transition type '{atom_type}' requires a set of agents.")
            transition_tuple = (state_to, atom_type, frozenset(agents))
        else:
            transition_tuple = (state_to, atom_type)
        self.transitions[(state_from, input_symbol)][atom_type].append(transition_tuple)

    def get_transitions(self, state, input_symbol):
        return self.transitions.get((state, input_symbol), {"universal": [], "existential": [], "epsilon": []})

    def __str__(self):
        state_formulas = [str(state) for state in self.states]
        alphabet_str = sorted(["{" + ", ".join(a) + "}" if a else "{}" for a in self.alphabet])
        return (
            f"ACG(\n"
            f"  Alphabet: {alphabet_str},\n"
            f"  States: {state_formulas},\n"
            f"  Initial State: {self.initial_state}\n"
            f")"
        )

