from itertools import chain, combinations, product
from copy import deepcopy



LPAREN, RPAREN = 1, 2
LBRACE, RBRACE = 3, 4
LTRI, RTRI = 5, 6
COMMA = 7
AND, OR, NOT = 8, 9, 10
IMPLIES, IFF = 11, 12
NEXT, UNTIL, RELEASE = 13, 14, 15
GLOBALLY, EVENTUALLY = 16, 17
PROPOSITION, AGENT_NAME = 18, 19
NAME, UNKNOWN, END_OF_INPUT = 20, 21, 22
LBRACKET, RBRACKET = 23,24
SYMBOL_MAP = {
    "◯": NEXT,
    "□": GLOBALLY,
    "◇": EVENTUALLY,
    "U": UNTIL,
    "R": RELEASE,
    "G": GLOBALLY,
    "F": EVENTUALLY,
    "X": NEXT
}

#===========
# ACG
#===========

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
        lhs_str = f"({self.lhs})" if isinstance(self.lhs, (And,Or)) else str(self.lhs)
        rhs_str = f"({self.rhs})" if isinstance(self.rhs, (And,Or)) else str(self.rhs)
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
        lhs_str = f"({self.lhs})" if isinstance(self.lhs, (And,Or)) else str(self.lhs)
        rhs_str = f"({self.rhs})" if isinstance(self.rhs, (And,Or)) else str(self.rhs)
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

class Bottom:
    def __eq__(self, other):
        return isinstance(other, Bottom)

    def __hash__(self):
        return hash("Bottom")

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
    
    def __eq__(self, other):
        return isinstance(other, Conj) and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        return hash(('Conj', self.lhs, self.rhs))

class Disj(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} OR {self.rhs})"

    def __str__(self):
        return self.to_formula()
    
    def __eq__(self, other):
        return isinstance(other, Disj) and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        return hash(('Disj', self.lhs, self.rhs))

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

class EpsilonAtom:
    def __init__(self, state):
        self.state = state

    def __eq__(self, other):
        return isinstance(other, EpsilonAtom) and self.state == other.state

    def __hash__(self):
        return hash(("EpsilonAtom", self.state))

    def __str__(self):
        return f"( {self.state}, ε )"
    
class ACG:
    def __init__(self):
        self.propositions = set()  
        self.states = set()  
        self.initial_state = None  
        self.transitions = {}  
        self.final_states = set()  
        self.alphabet = set()  

    def generate_alphabet(self):
        self.alphabet = set(
            frozenset(s)
            for s in chain.from_iterable(
                combinations(self.propositions, r) for r in range(len(self.propositions) + 1)
            )
        )

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
        final_formulas = sorted(str(state) for state in self.final_states)

        alphabet_str = sorted(
            ["{" + ", ".join(sorted(a)) + "}" if a else "{}" for a in self.alphabet]
        )

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
            f"  Final States: {final_formulas},\n"
            f"  Transitions:\n" +
            "\n".join(formatted_transitions) +
            f"\n)"
        )

def tokenize(source): 
    tokens = []
    cursor = 0
    length = len(source)

    keywords = {
        "and": AND,
        "or": OR,
        "not": NOT,
        "next": NEXT,
        "until": UNTIL,
        "release": RELEASE,
        "globally": GLOBALLY,
        "eventually": EVENTUALLY,
        "implies": IMPLIES
    }

    inside_agents = False  

    while cursor < length:
        symbol = source[cursor]

        if symbol in " \t\n":
            pass  

        elif symbol == "(":
            tokens.append((LPAREN, symbol))
        elif symbol == ")":
            tokens.append((RPAREN, symbol))
        elif symbol == "<":
            tokens.append((LTRI, symbol))
            inside_agents = True  
        elif symbol == ">":
            tokens.append((RTRI, symbol))
            inside_agents = False  
        elif symbol == "[":
            tokens.append((LBRACKET, symbol))
            inside_agents = True
        elif symbol == "]":
            tokens.append((RBRACKET, symbol))
            inside_agents = False
        elif symbol == ",":
            tokens.append((COMMA, symbol))
        elif symbol == "|":
            tokens.append((OR, symbol))
        elif symbol == "&":
            tokens.append((AND, symbol))
        elif symbol in SYMBOL_MAP and inside_agents == False:
            tokens.append((SYMBOL_MAP[symbol], symbol))
        
        elif symbol.isalpha():
            buffer = symbol
            cursor += 1
            while cursor < length and source[cursor].isalnum():
                buffer += source[cursor]
                cursor += 1

            if inside_agents:
                tokens.append((AGENT_NAME, buffer))
            else:
                token_type = keywords.get(buffer.lower(), PROPOSITION)  
                tokens.append((token_type, buffer))

            cursor -= 1  

        else:
            tokens.append((UNKNOWN, symbol))

        cursor += 1

    return tokens

def parse(tokens):
    cursor = 0
    limit = len(tokens)

    def advance():
        nonlocal cursor
        if cursor < limit:
            cursor += 1

    def current_token():
        return tokens[cursor] if cursor < limit else (END_OF_INPUT, "")

    def parse_atomic_formula():
        token = current_token()

        if token[0] == LPAREN:
            advance()
            result = parse_formula()
            if current_token()[0] != RPAREN:
                raise ValueError("Parse error: Expected ')' but not found.")
            advance()
            return result

        elif token[0] == PROPOSITION:
            advance()
            return Var(token[1])

        elif token[0] == NOT:
            advance()
            return Not(parse_atomic_formula())

        elif token[0] == LTRI:
            agents = []
            advance()
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()
            if current_token()[0] != RTRI:
                raise ValueError("Parse error: Expected '>' but not found.")
            advance()
            return Modality(agents, parse_temporal())

        elif token[0] == LBRACKET:
            agents = []
            advance()
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()
            if current_token()[0] != RBRACKET:
                raise ValueError("Parse error: Expected ']' but not found.")
            advance()
            return DualModality(agents, parse_temporal())

        else:
            raise ValueError(f"Parse error: unexpected token '{token[1]}'.")

    def parse_temporal():
        token = current_token()

        if token[0] == NEXT:
            advance()
            return Next(parse_temporal())

        elif token[0] == GLOBALLY:
            advance()
            return Globally(parse_temporal())

        elif token[0] == EVENTUALLY:
            advance()
            return Eventually(parse_temporal())

        elif token[0] == LTRI:
            agents = []
            advance()
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()
            if current_token()[0] != RTRI:
                raise ValueError("Parse error: Expected '>' but not found.")
            advance()
            return Modality(agents, parse_temporal())

        elif token[0] == LBRACKET:
            agents = []
            advance()
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()
            if current_token()[0] != RBRACKET:
                raise ValueError("Parse error: Expected ']' but not found.")
            advance()
            return DualModality(agents, parse_temporal())

        else:
            return parse_atomic_formula()

    def parse_until():
        result = parse_temporal()
        while current_token()[0] in {UNTIL, RELEASE}:
            token_type = current_token()[0]
            advance()
            rhs = parse_temporal()

            if token_type == UNTIL:
                if isinstance(result, Modality):
                    result = Modality(result.agents, Until(result.sub, rhs))
                elif isinstance(result, DualModality):
                    result = DualModality(result.agents, Until(result.sub, rhs))
                else:
                    result = Until(result, rhs)

            elif token_type == RELEASE:
                if isinstance(result, Modality):
                    result = Modality(result.agents, Release(result.sub, rhs))
                elif isinstance(result, DualModality):
                    result = DualModality(result.agents, Release(result.sub, rhs))
                else:
                    result = Release(result, rhs)

        return result


    def parse_conjunction():
        result = parse_until()
        while current_token()[0] == AND:
            advance()
            rhs = parse_until()
            result = And(result, rhs)
        return result

    def parse_disjunction():
        result = parse_conjunction()
        while current_token()[0] == OR:
            advance()
            rhs = parse_conjunction()
            result = Or(result, rhs)
        return result

    def parse_implies():
        result = parse_disjunction()
        while current_token()[0] in {IMPLIES, IFF}:
            token_type = current_token()[0]
            advance()
            rhs = parse_disjunction()
            if token_type == IMPLIES:
                result = Implies(result, rhs)
            elif token_type == IFF:
                result = Iff(result, rhs)
        return result

    def parse_formula():
        return parse_implies()

    ast_root = parse_formula()

    if current_token()[0] != END_OF_INPUT:
        raise ValueError(f"Parse error: Unexpected token '{current_token()[1]}' at end of input.")

    return ast_root

def eliminate_f_and_r(node: ParseNode) -> ParseNode:
    if isinstance(node, Eventually):
        return Until(T(), eliminate_f_and_r(node.sub), generated_from_eventually=True)

    elif isinstance(node, Release):
        lhs = eliminate_f_and_r(node.lhs)
        rhs = eliminate_f_and_r(node.rhs)
        return Not(Until(Not(lhs), Not(rhs)))

    elif isinstance(node, And):
        return And(eliminate_f_and_r(node.lhs), eliminate_f_and_r(node.rhs))

    elif isinstance(node, Or):
        return Or(eliminate_f_and_r(node.lhs), eliminate_f_and_r(node.rhs))

    elif isinstance(node, Not):
        return Not(eliminate_f_and_r(node.sub))

    elif isinstance(node, Next):
        return Next(eliminate_f_and_r(node.sub))

    elif isinstance(node, Until):
        return Until(eliminate_f_and_r(node.lhs), eliminate_f_and_r(node.rhs))

    elif isinstance(node, Globally):
        return Globally(eliminate_f_and_r(node.sub))

    elif isinstance(node, Eventually):
        return Until(T(), eliminate_f_and_r(node.sub), generated_from_eventually=True)

    elif isinstance(node, Implies):
        return Implies(eliminate_f_and_r(node.lhs), eliminate_f_and_r(node.rhs))

    elif isinstance(node, Iff):
        return Iff(eliminate_f_and_r(node.lhs), eliminate_f_and_r(node.rhs))

    elif isinstance(node, Modality):
        return Modality(node.agents, eliminate_f_and_r(node.sub))

    elif isinstance(node, DualModality):
        return DualModality(node.agents, eliminate_f_and_r(node.sub))

    return node 

def push_negations_to_nnf(node: ParseNode) -> ParseNode:
    if isinstance(node, Not):
        sub = node.sub

        if isinstance(sub, Not):
            return push_negations_to_nnf(sub.sub)

        if isinstance(sub, And):
            return Or(
                push_negations_to_nnf(Not(sub.lhs)),
                push_negations_to_nnf(Not(sub.rhs))
            )

        if isinstance(sub, Or):
            return And(
                push_negations_to_nnf(Not(sub.lhs)),
                push_negations_to_nnf(Not(sub.rhs))
            )

        if isinstance(sub, T):
            return F()

        if isinstance(sub, F):
            return T()

        return Not(push_negations_to_nnf(sub))

    elif isinstance(node, And):
        return And(push_negations_to_nnf(node.lhs), push_negations_to_nnf(node.rhs))

    elif isinstance(node, Or):
        return Or(push_negations_to_nnf(node.lhs), push_negations_to_nnf(node.rhs))

    elif isinstance(node, Next):
        return Next(push_negations_to_nnf(node.sub))

    elif isinstance(node, Until):
        return Until(push_negations_to_nnf(node.lhs), push_negations_to_nnf(node.rhs))

    elif isinstance(node, Globally):
        return Globally(push_negations_to_nnf(node.sub))

    elif isinstance(node, Eventually):
        return Eventually(push_negations_to_nnf(node.sub))

    elif isinstance(node, Modality):
        return Modality(node.agents, push_negations_to_nnf(node.sub))

    elif isinstance(node, DualModality):
        return DualModality(node.agents, push_negations_to_nnf(node.sub))

    return node  

def apply_modal_dualities(node: ParseNode) -> ParseNode:
    if isinstance(node, Not):
        inner = node.sub

        # ¬<A> ...
        if isinstance(inner, Modality):
            sub = inner.sub
            agents = inner.agents

            # ¬⟨A⟩ X φ ⇒ [A] X ¬φ
            if isinstance(sub, Next):
                return DualModality(agents, Next(Not(apply_modal_dualities(sub.sub))))

            # ¬⟨A⟩ G φ ⇒ [A] (⊤ U ¬φ)
            elif isinstance(sub, Globally):
                return DualModality(agents, Until(T(), Not(apply_modal_dualities(sub.sub))))

            # ¬⟨A⟩ (⊤ U φ) ⇒ [A] G ¬φ
            elif isinstance(sub, Until) and isinstance(sub.lhs, T):
                return DualModality(agents, Globally(Not(apply_modal_dualities(sub.rhs))))

            # ¬⟨A⟩ (φ U ψ) ⇒ [A] ¬(ψ U φ)
            elif isinstance(sub, Until):
                return DualModality(agents, Not(Until(apply_modal_dualities(sub.rhs), apply_modal_dualities(sub.lhs))))

            # ¬⟨A⟩ ¬(φ U ψ) ⇒ [A] (φ U ψ)
            elif isinstance(sub, Not) and isinstance(sub.sub, Until):
                return DualModality(agents, Until(apply_modal_dualities(sub.sub.lhs), apply_modal_dualities(sub.sub.rhs)))

        # ¬[A] ...
        elif isinstance(inner, DualModality):
            sub = inner.sub
            agents = inner.agents

            # ¬[A] X φ ⇒ ⟨A⟩ X ¬φ
            if isinstance(sub, Next):
                return Modality(agents, Next(Not(apply_modal_dualities(sub.sub))))

            # ¬[A] G φ ⇒ ⟨A⟩ (⊤ U ¬φ)
            elif isinstance(sub, Globally):
                return Modality(agents, Until(T(), Not(apply_modal_dualities(sub.sub))))

            # ¬[A] (⊤ U φ) ⇒ ⟨A⟩ G ¬φ
            elif isinstance(sub, Until) and isinstance(sub.lhs, T):
                return Modality(agents, Globally(Not(apply_modal_dualities(sub.rhs))))

            # ¬[A] (φ U ψ) ⇒ ⟨A⟩ ¬(ψ U φ)
            elif isinstance(sub, Until):
                return Modality(agents, Not(Until(apply_modal_dualities(sub.rhs), apply_modal_dualities(sub.lhs))))

            # ¬[A] ¬(φ U ψ) ⇒ ⟨A⟩ (φ U ψ)
            elif isinstance(sub, Not) and isinstance(sub.sub, Until):
                return Modality(agents, Until(apply_modal_dualities(sub.sub.lhs), apply_modal_dualities(sub.sub.rhs)))

    # Recursión en hijos
    elif isinstance(node, And):
        return And(apply_modal_dualities(node.lhs), apply_modal_dualities(node.rhs))
    elif isinstance(node, Or):
        return Or(apply_modal_dualities(node.lhs), apply_modal_dualities(node.rhs))
    elif isinstance(node, Not):
        return Not(apply_modal_dualities(node.sub))
    elif isinstance(node, Next):
        return Next(apply_modal_dualities(node.sub))
    elif isinstance(node, Until):
        return Until(apply_modal_dualities(node.lhs), apply_modal_dualities(node.rhs))
    elif isinstance(node, Globally):
        return Globally(apply_modal_dualities(node.sub))
    elif isinstance(node, Modality):
        return Modality(node.agents, apply_modal_dualities(node.sub))
    elif isinstance(node, DualModality):
        return DualModality(node.agents, apply_modal_dualities(node.sub))
    
    return node

def normalize_formula(ast: ParseNode) -> ParseNode:
    previous = None
    current = ast

    while previous != current:
        previous = current
        current = eliminate_f_and_r(current)
        current = push_negations_to_nnf(current)

    return current

def filter(ast, strict_ATL=True):
    has_modality = False
    agents_set = None

    def check_invalid(node):
        nonlocal has_modality

        if isinstance(node, (Modality, DualModality)):
            has_modality = True

            if not isinstance(node.sub, ParseNode) or not isinstance(node.agents, list):
                print("ERROR: Modality must have valid agents and subformula.")
                return "INVALID"

        if isinstance(node, Until):
            if not isinstance(node.lhs, ParseNode) or not isinstance(node.rhs, ParseNode):
                print("ERROR: Until must have well defined arguments.")
                return "INVALID"

        for child in getattr(node, "__dict__", {}).values():
            if isinstance(child, ParseNode):
                result = check_invalid(child)
                if result:
                    return result

        return None

    resultado = check_invalid(ast)
    if resultado:
        return resultado

    if not strict_ATL:
        return "ATL*"

    def check_strict_ATL(node, parent=None, grandparent=None):
        nonlocal has_modality, agents_set

        if isinstance(node, (Modality, DualModality)):
            has_modality = True

            if agents_set is None:
                agents_set = set(node.agents)
            else:
                if set(node.agents) != agents_set:
                    print(f"ERROR: Modalities with different agents: {agents_set} vs {set(node.agents)}.")
                    return "ATL* but not ATL"

            if not isinstance(node.sub, (Next, Globally, Until, Not)):
                print("ERROR: Modality can only be applied to Next, Globally or Until.")
                return "ATL* but not ATL"

        if isinstance(node, Until):
            if getattr(node, "generated_from_eventually", False):
                pass
            elif isinstance(parent, (Modality, DualModality)):
                pass
            elif isinstance(parent, Not) and isinstance(grandparent, (Modality, DualModality)):
                pass
            else:
                print(f"ERROR: Until is not immediately preceded by a modality.")
                return "ATL* but not ATL"

        if isinstance(node, (Next, Globally)):
            if not isinstance(parent, (Modality, DualModality)):
                print(f"ERROR: {node.__class__.__name__} is not immediately preceded by a modality.")
                return "ATL* but not ATL"

        for key, child in node.__dict__.items():
            if isinstance(child, ParseNode):
                result = check_strict_ATL(child, node, parent)
                if result:
                    return result
            elif isinstance(child, list):
                for sub in child:
                    if isinstance(sub, ParseNode):
                        result = check_strict_ATL(sub, node, parent)
                        if result:
                            return result

        return None

    result_ATL = check_strict_ATL(ast)
    if result_ATL:
        return result_ATL

    return "ATL"

def extract_propositions(node):
    propositions = set()

    def traverse(n):
        if isinstance(n, Var):
            propositions.add(n.name) 
        
        for child in getattr(n, "__dict__", {}).values():
            if isinstance(child, ParseNode):
                traverse(child)

    traverse(node)
    return propositions 

def generate_closure(ast):
    closure = set()

    def negate_and_push(node):
        neg = Not(deepcopy(node))
        return push_negations_to_nnf(neg)

    def add_with_negations(node):
        closure.add(node)
        closure.add(negate_and_push(node))

    def traverse(node, parent=None):
        if isinstance(node, Not) and isinstance(node.sub, (Modality, DualModality)):
            inner = node.sub.sub
            if isinstance(inner, (Next, Globally)):
                add_with_negations(node)
                traverse(inner.sub, inner)
            elif isinstance(inner, Until):
                add_with_negations(node)
                traverse(inner.lhs, inner)
                traverse(inner.rhs, inner)
            return

        elif isinstance(node, (Modality, DualModality)):
            add_with_negations(node)
            traverse(node.sub, node)
            return

        elif isinstance(node, (And, Or)):
            add_with_negations(node)
            traverse(node.lhs, node)
            traverse(node.rhs, node)
            return

        elif isinstance(node, Not):
            if isinstance(node.sub, Var):
                closure.add(node)
                closure.add(node.sub)
            else:
                traverse(node.sub, node)
            return

        elif isinstance(node, Var):
            if not isinstance(parent, Not):
                closure.add(node)
                closure.add(Not(deepcopy(node)))  
            return

        elif isinstance(node, Until):
            if isinstance(parent, (Modality, DualModality)):
                traverse(node.lhs, node)
                traverse(node.rhs, node)
            return

        elif isinstance(node, (Next, Globally)):
            if isinstance(parent, (Modality, DualModality)):
                traverse(node.sub, node)
            return

    traverse(ast)
    return closure

def generate_transitions(acg):
    for state in acg.states:
        for sigma in acg.alphabet:

            #  p
            if isinstance(state, Var):
                if state.name in sigma:
                    acg.add_transition(state, sigma, Top())
                else:
                    acg.add_transition(state, sigma, Bottom())

            # ¬p
            elif isinstance(state, Not) and isinstance(state.sub, Var):
                if state.sub.name in sigma:
                    acg.add_transition(state, sigma, Bottom())
                else:
                    acg.add_transition(state, sigma, Top())

            # AND
            elif isinstance(state, And):
                acg.add_transition(state, sigma, Conj(
                    EpsilonAtom(state.lhs), EpsilonAtom(state.rhs)))

            # OR
            elif isinstance(state, Or):
                acg.add_transition(state, sigma, Disj(
                    EpsilonAtom(state.lhs), EpsilonAtom(state.rhs)))

            # ⟨A⟩ X φ
            elif isinstance(state, Modality) and isinstance(state.sub, Next):
                next_state = state.sub.sub
                agents = frozenset(state.agents)
                acg.add_transition(state, sigma, ExistentialAtom(next_state, agents))

            # ¬⟨A⟩ X φ 
            elif isinstance(state, Not) and isinstance(state.sub, Modality) and isinstance(state.sub.sub, Next):
                inner = state.sub.sub.sub
                negated_inner = push_negations_to_nnf(Not(inner))
                agents = frozenset(state.sub.agents)
                acg.add_transition(state, sigma, UniversalAtom(negated_inner, agents))

            # ⟨A⟩ G φ
            elif isinstance(state, Modality) and isinstance(state.sub, Globally):
                phi = state.sub.sub
                agents = frozenset(state.agents)
                acg.add_transition(state, sigma, Conj(
                    EpsilonAtom(phi),
                    ExistentialAtom(state, agents)
                ))

            # ¬⟨A⟩ G φ 
            elif isinstance(state, Not) and isinstance(state.sub, Modality) and isinstance(state.sub.sub, Globally):
                phi = state.sub.sub.sub
                neg_phi = push_negations_to_nnf(Not(phi))
                agents = frozenset(state.sub.agents)
                acg.add_transition(state, sigma, Disj(
                    EpsilonAtom(neg_phi),
                    UniversalAtom(state, agents)
                ))

            # ⟨A⟩ (φ1 U φ2)
            elif isinstance(state, Modality) and isinstance(state.sub, Until):
                phi1 = state.sub.lhs
                phi2 = state.sub.rhs
                agents = frozenset(state.agents)
                acg.add_transition(state, sigma, Disj(
                    EpsilonAtom(phi2),
                    Conj(
                        EpsilonAtom(phi1),
                        ExistentialAtom(state, agents)
                    )
                ))

            # ¬⟨A⟩ (φ1 U φ2)
            elif isinstance(state, Not) and isinstance(state.sub, Modality) and isinstance(state.sub.sub, Until):
                phi1 = state.sub.sub.lhs
                phi2 = state.sub.sub.rhs
                neg_phi1 = push_negations_to_nnf(Not(phi1))
                neg_phi2 = push_negations_to_nnf(Not(phi2))
                agents = frozenset(state.sub.agents)
                acg.add_transition(state, sigma, Conj(
                    EpsilonAtom(neg_phi2),
                    Disj(
                        EpsilonAtom(neg_phi1),
                        UniversalAtom(state, agents)
                    )
                ))

def build_acg(transformed_ast):
    ap_set = extract_propositions(transformed_ast)
    alphabet = set(frozenset(s)for s in chain.from_iterable(combinations(ap_set, r) for r in range(len(ap_set) + 1)))
    acg = ACG()
    acg.propositions = ap_set
    acg.alphabet = alphabet

    closure = generate_closure(transformed_ast)
    acg.states = closure
    acg.initial_state = transformed_ast

    for node in closure:
        if isinstance(node, Modality) and isinstance(node.sub, Globally):
            acg.final_states.add(node)

        elif isinstance(node, Not) and isinstance(node.sub, Modality) and isinstance(node.sub.sub, Until):
            acg.final_states.add(node)

    generate_transitions(acg)

    return acg

#===========
# CGS
#===========

class CGS:
    def __init__(self):
        self.propositions = set()
        self.agents = set()
        self.states = set()
        self.initial_state = None
        self.labeling_function = {}
        self.decisions = {}
        self.transition_function = {}
        self.strategies = {}

    def add_proposition(self, proposition):
        self.propositions.add(proposition)

    def add_state(self, state):
        self.states.add(state)

    def set_initial_state(self, state):
        if state not in self.states:
            self.add_state(state)
        self.initial_state = state

    def label_state(self, state, propositions):
        if state not in self.states:
            self.add_state(state)
        self.labeling_function[state] = set(propositions)

    def add_agent(self, agent):
        self.agents.add(agent)

    def add_decisions(self, agent, decision_set):
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the CGS.")
        self.decisions[agent] = set(decision_set)

    def add_transition(self, state, joint_decision, next_state):
        if state not in self.states:
            self.add_state(state)
        if next_state not in self.states:
            self.add_state(next_state)

        ordered_joint_action = frozenset(sorted(joint_decision, key=lambda x: x[0]))
        self.transition_function[(state, ordered_joint_action)] = next_state

    
    def get_all_agent_choices(self, agent_subset):

        agent_subset = sorted(agent_subset)
        all_choices = [self.decisions[agent] for agent in agent_subset]
        combinations = product(*all_choices)

        return [
            dict(zip(agent_subset, combo)) for combo in combinations
        ]

    def get_joint_actions_for_agents(self, agent_subset):

        agent_subset = sorted(agent_subset)
        all_choices = [self.decisions[agent] for agent in agent_subset]
        combos = product(*all_choices)

        return [dict(zip(agent_subset, combo)) for combo in combos]

    def get_successor(self, state, joint_decision_dict):

        joint_action = frozenset(sorted(joint_decision_dict.items()))
        return self.transition_function.get((state, joint_action), None)


    def __str__(self):
        formatted_transitions = []
        for (state, joint_decision) in self.transition_function:
            decision_str = ", ".join([f"({agent}, {decision})" for agent, decision in joint_decision])
            next_state = self.transition_function[(state, joint_decision)]
            formatted_transitions.append(f"    τ({state}, {{{decision_str}}}) → {next_state}")

        return (
            f"CGS(\n"
            f"  Propositions: {self.propositions}\n"
            f"  Agents: {sorted(self.agents)}\n"
            f"  States: {sorted(self.states)}\n"
            f"  Initial State: {self.initial_state}\n"
            f"  Labeling Function: {self.labeling_function}\n"
            f"  Decisions: {self.decisions}\n"
            f"  Transitions:\n" +
            "\n".join(formatted_transitions) +
            f"\n)"
        )

class Strategy:
    def __init__(self, agents, decision_map=None):
        self.agents = set(agents)
        self.decision_map = decision_map if decision_map else {}

    def add_decision(self, history, decisions):
        if not isinstance(decisions, dict):
            raise ValueError("Decisions must be a dictionary {agent: decision}.")
        self.decision_map[tuple(history)] = decisions

    def get_decision(self, history):
        return self.decision_map.get(tuple(history), {})

    def __str__(self):
        formatted_decisions = [
            f"  History: ({' → '.join(history)}) → Decisions: {decisions}"
            for history, decisions in self.decision_map.items()
        ]
        return f"Strategy({self.agents}):\n" + "\n".join(formatted_decisions)

class CounterStrategy:
    def __init__(self, agents, decision_map=None):
        self.agents = set(agents)
        self.decision_map = decision_map if decision_map else {}

    def add_decision(self, history, decision_A_prime, decision_A_complement):
        key = (tuple(history), frozenset(decision_A_prime.items()))
        self.decision_map[key] = decision_A_complement

    def get_decision(self, history, decision_A_prime):
        key = (tuple(history), frozenset(decision_A_prime.items()))
        return self.decision_map.get(key, {})

    def __str__(self):
        formatted_decisions = [
            f"  History: ({' → '.join(history)}), Decision_A': {decision_A_prime} → Decision_A\A': {decisions_A_complement}"
            for (history, decision_A_prime), decisions_A_complement in self.decision_map.items()
        ]
        return f"CounterStrategy({self.agents}):\n" + "\n".join(formatted_decisions)

def play(game, initial_state, strategy, counterstrategy, max_steps=10):
    history = [initial_state]
    current_state = initial_state

    for _ in range(max_steps):
        print(f"\n🟢 Current State: {current_state}")

        decision_A = strategy.get_decision([current_state])
        if not decision_A:
            print(f" Strategy failed to return a decision at {current_state}. Stopping.")
            break  

        decision_A_complement = counterstrategy.get_decision([current_state], decision_A)
        if not decision_A_complement:
            print(f" CounterStrategy failed to return a decision at {current_state}. Stopping.")
            break  

        joint_action = frozenset(list(decision_A.items()) + list(decision_A_complement.items()))

        if (current_state, joint_action) in game.transition_function:
            current_state = game.transition_function[(current_state, joint_action)]
            history.append(current_state)
            print(f" Transition found! Moving to: {current_state}")
        else:
            print(" No matching transition found. Stopping.")
            break  

    return history


#===========
# GAME
#===========


def evaluate_boolean_formula(formula, atom_set):

    if isinstance(formula, str):
        return formula != "∅"  

    if isinstance(formula, Top):
        return True

    if isinstance(formula, Bottom):
        return False

    if isinstance(formula, Conj):
        return (
            evaluate_boolean_formula(formula.lhs, atom_set)
            and evaluate_boolean_formula(formula.rhs, atom_set)
        )

    if isinstance(formula, Disj):
        return (
            evaluate_boolean_formula(formula.lhs, atom_set)
            or evaluate_boolean_formula(formula.rhs, atom_set)
        )

    for atom in atom_set:
        if formula == atom:
            return True

    return False

def generate_possibilities(formula):

    if isinstance(formula, (EpsilonAtom, UniversalAtom, ExistentialAtom)):
        return [frozenset([formula])]

    elif isinstance(formula, Conj):
        left = generate_possibilities(formula.lhs)
        right = generate_possibilities(formula.rhs)
        return [a.union(b) for a in left for b in right]

    elif isinstance(formula, Disj):
        left = generate_possibilities(formula.lhs)
        right = generate_possibilities(formula.rhs)
        return left + right

    return []


class GameProduct:
    def __init__(self, acg: ACG, cgs: CGS):
        self.acg = acg
        self.cgs = cgs
        self.states = set()
        self.transitions = dict()
        self.initial_states = set()
        self.S1 = set()
        self.S2 = set()
        self.B = set()  

    def __str__(self):
        pretty_initial = [pretty_node(s) for s in sorted(self.initial_states, key=str)]
        pretty_s1 = [pretty_node(s) for s in sorted(self.S1, key=str)]
        pretty_s2 = [pretty_node(s) for s in sorted(self.S2, key=str)]
        pretty_B = [pretty_node(s) for s in sorted(self.B, key=str)]

        lines = [
            "GameProduct(",
            f"  Initial States: {pretty_initial}",
            f"  Total States: {len(self.states)}",
            f"  Player Accept States (S1): {pretty_s1}",
            f"  Player Reject States (S2): {pretty_s2}",
            f"  Büchi Final States (B): {pretty_B}",
            ")"
        ]
        return "\n".join(lines)


def generate_initial_game_states(product: GameProduct):
    q0 = product.acg.initial_state
    s0 = product.cgs.initial_state
    initial = ("state", q0, s0)

    product.initial_states.add(initial)
    product.states.add(initial)
    product.S1.add(initial)

    return initial

def expand_from_state_node(product: GameProduct, q, s):
    sigma = frozenset(product.cgs.labeling_function[s])
    delta_formula = product.acg.get_transition(q, sigma)

    for U in generate_possibilities(delta_formula):
        atom_selection_node = ("atom_selection", q, s, frozenset(U))

        product.states.add(atom_selection_node)
        product.transitions[(("state", q, s), atom_selection_node)] = None
        product.S2.add(atom_selection_node)

def expand_from_atom_selection_node(product: GameProduct, q, s, U):
    for alpha in U:
        if isinstance(alpha, (EpsilonAtom, UniversalAtom, ExistentialAtom)):
            q_prime = alpha.state
            new_node = ("atom_applied", q_prime, s, alpha)
            
            product.states.add(new_node)
            product.transitions[(("atom_selection", q, s, U), new_node)] = None
            product.S1.add(new_node)  

def expand_from_atom_applied_node(product: GameProduct, q, s, alpha):
    if isinstance(alpha, EpsilonAtom):
        q_prime = alpha.state
        s_prime = s
        new_state = ("state", q_prime, s_prime)
        product.states.add(new_state)
        product.transitions[(("atom_applied", q, s, alpha), new_state)] = None
        product.S1.add(new_state)

    elif isinstance(alpha, UniversalAtom):
        for d in product.cgs.get_all_agent_choices(alpha.agents):
            reject_univ_node = (
                "reject_univ",
                q,
                s,
                alpha.agents,
                alpha,
                frozenset(d.items())
            )
            product.states.add(reject_univ_node)
            product.transitions[(("atom_applied", q, s, alpha), reject_univ_node)] = None
            product.S2.add(reject_univ_node)

    elif isinstance(alpha, ExistentialAtom):
        A_prime = alpha.agents
        A = product.cgs.agents
        A_minus_A_prime = A - A_prime

        for v_reject in product.cgs.get_all_agent_choices(A_minus_A_prime):
            accept_exist_node = (
                "accept_exist",
                q,
                s,
                A_prime,
                alpha,
                frozenset(v_reject.items())
            )
            product.states.add(accept_exist_node)
            product.transitions[(("atom_applied", q, s, alpha), accept_exist_node)] = None
            product.S1.add(accept_exist_node)

def expand_from_reject_univ_node(product: GameProduct, q, s, A, alpha, dA_frozen):
    dA = dict(dA_frozen)
    remaining_agents = product.cgs.agents - A

    for d_reject in product.cgs.get_all_agent_choices(remaining_agents):
        full_decision = {**dA, **d_reject}
        successor = product.cgs.get_successor(s, full_decision)

        if successor is not None:
            q_prime = alpha.state
            new_node = ("state", q_prime, successor)

            product.states.add(new_node)
            product.transitions[(("reject_univ", q, s, A, alpha, dA_frozen), new_node)] = None
            product.S1.add(new_node)

def expand_from_accept_exist_node(product: GameProduct, q, s, agents_prime, alpha, v_reject):

    q_prime = alpha.state
    A_prime = agents_prime
    A = product.cgs.agents
    A_minus_A_prime = A - A_prime

    v_reject_dict = dict(v_reject)
    if set(v_reject_dict.keys()) != A_minus_A_prime:
        raise ValueError(f"Reject's decision should cover exactly A \\ A', got {v_reject_dict.keys()}")

    for v_accept in product.cgs.get_joint_actions_for_agents(A_prime):
        full_joint_action = {**v_reject_dict, **v_accept}

        s_prime = product.cgs.get_successor(s, full_joint_action)

        if s_prime is not None:
            new_state = ("state", q_prime, s_prime)
            product.states.add(new_state)
            product.transitions[(("accept_exist", q, s, A_prime, alpha, v_reject), new_state)] = None
            product.S1.add(new_state)

def compute_buchi_states(product: GameProduct) -> set:
    return {
        ("state", q, s)
        for q in product.acg.final_states
        for s in product.cgs.states
        
    }

def pretty_node(node):
    if node[0] == "state":
        _, q, s = node
        return f"('state', {q.to_formula()}, {s})"

    elif node[0] == "atom_selection":
        _, q, s, U = node
        atoms_str = ", ".join(str(a) for a in sorted(U, key=str))
        return f"('atom_selection', {q.to_formula()}, {s}, {{{atoms_str}}})"

    elif node[0] == "atom_applied":
        _, q, s, alpha = node
        return f"('atom_applied', {q.to_formula()}, {s}, {str(alpha)})"

    elif node[0] == "reject_univ":
        _, q, s, A, alpha, d = node
        d_str = ", ".join(f"{k}: {v}" for k, v in dict(d).items())
        return f"('reject_univ', {q.to_formula()}, {s}, {A}, {str(alpha)}, {{{d_str}}})"

    elif node[0] == "accept_exist":
        _, q, s, A_prime, alpha, v_reject = node
        A_str = "{" + ", ".join(A_prime) + "}"
        v_reject_str = ", ".join(f"{k}: {v}" for k, v in dict(v_reject).items())
        return f"('accept_exist', {q.to_formula()}, {s}, {A_str}, {str(alpha)}, {{{v_reject_str}}})"

    return str(node)

def expand_node(product, node):
    kind = node[0]
    q, s = None, None

    if kind == "state":
        _, q, s = node
        expand_from_state_node(product, q, s)
        return [dst for (src, dst) in product.transitions if src == node]

    elif kind == "atom_selection":
        _, q, s, U = node
        expand_from_atom_selection_node(product, q, s, U)
        return [dst for (src, dst) in product.transitions if src == node]

    elif kind == "atom_applied":
        _, q, s, alpha = node
        expand_from_atom_applied_node(product, q, s, alpha)
        return [dst for (src, dst) in product.transitions if src == node]

    elif kind == "reject_univ":
        _, q, s, A, alpha, dA = node
        expand_from_reject_univ_node(product, q, s, A, alpha, dA)
        return [dst for (src, dst) in product.transitions if src == node]

    elif kind == "accept_exist":
        _, q, s, A_prime, alpha, v_reject = node
        expand_from_accept_exist_node(product, q, s, A_prime, alpha, v_reject)
        return [dst for (src, dst) in product.transitions if src == node]

    return []

def build_game(acg, cgs):
    product = GameProduct(acg, cgs)

    initial = generate_initial_game_states(product)
    worklist = [initial]
    visited = set()

    while worklist:
        node = worklist.pop()
        if node in visited:
            continue
        visited.add(node)

        new_nodes = expand_node(product, node)
        worklist.extend(new_nodes)

    product.B = compute_buchi_states(product)

    return product.states, product.transitions, product.S1, product.S2, product.B, initial



cgs = CGS()

cgs.add_proposition("safe")
cgs.add_proposition("critical")
cgs.add_proposition("operational")
cgs.add_proposition("shutdown")

cgs.add_agent("Reactor")
cgs.add_agent("Pressure")

cgs.add_decisions("Reactor", {"heat", "cool"})
cgs.add_decisions("Pressure", {"hold", "vent"})

cgs.add_state("Start")
cgs.add_state("Stable")
cgs.add_state("Cold")
cgs.add_state("Critical")
cgs.add_state("Shut Down")

cgs.set_initial_state("Start")

cgs.label_state("Start", {"safe"})
cgs.label_state("Critical", {"critical"})
cgs.label_state("Stable", {"safe", "operational"})
cgs.label_state("Cold", {"safe"})
cgs.label_state("Shut Down", {"shutdown"})


cgs.add_transition("Start", {("Reactor", "heat"), ("Pressure", "hold")}, "Stable")
cgs.add_transition("Start", {("Reactor", "heat"), ("Pressure", "vent")}, "Stable")
cgs.add_transition("Start", {("Reactor", "cool"), ("Pressure", "hold")}, "Start")
cgs.add_transition("Start", {("Reactor", "cool"), ("Pressure", "vent")}, "Start")

cgs.add_transition("Stable", {("Reactor", "heat"), ("Pressure", "vent")}, "Stable")
cgs.add_transition("Stable", {("Reactor", "cool"), ("Pressure", "hold")}, "Cold")
cgs.add_transition("Stable", {("Reactor", "cool"), ("Pressure", "vent")}, "Cold")
cgs.add_transition("Stable", {("Reactor", "heat"), ("Pressure", "hold")}, "Critical")

cgs.add_transition("Cold", {("Reactor", "heat"), ("Pressure", "hold")}, "Stable")
cgs.add_transition("Cold", {("Reactor", "heat"), ("Pressure", "vent")}, "Stable")
cgs.add_transition("Cold", {("Reactor", "cool"), ("Pressure", "hold")}, "Start")
cgs.add_transition("Cold", {("Reactor", "cool"), ("Pressure", "vent")}, "Start")

cgs.add_transition("Critical", {("Reactor", "cool"), ("Pressure", "hold")}, "Critical")
cgs.add_transition("Critical", {("Reactor", "heat"), ("Pressure", "vent")}, "Stable")
cgs.add_transition("Critical", {("Reactor", "cool"), ("Pressure", "vent")}, "Stable")
cgs.add_transition("Critical", {("Reactor", "heat"), ("Pressure", "hold")}, "Shut Down")

cgs.add_transition("Shut Down", {("Reactor", "heat"), ("Pressure", "hold")}, "Start")
cgs.add_transition("Shut Down", {("Reactor", "heat"), ("Pressure", "vent")}, "Start")
cgs.add_transition("Shut Down", {("Reactor", "cool"), ("Pressure", "vent")}, "Start")
cgs.add_transition("Shut Down", {("Reactor", "cool"), ("Pressure", "hold")}, "Start")


# ================================================================
# MAIN
# ================================================================

test_formula = " <Reactor> safe until not cold"

print("=" * 80)
print(f"🔹 Formula Input : {test_formula}")

try:
    tokens = tokenize(test_formula)
    print(f"\n📎 Tokens: {tokens}")

    ast = parse(tokens)
    print("\n Initial AST:")
    print(ast.to_tree())

    normalized_ast = normalize_formula(ast)
    print("\n Normalized AST:")
    print(normalized_ast.to_tree())

    normalized_formula = normalized_ast.to_formula()
    print(f"\n Normalized Formula: {normalized_formula}")

    classification = filter(normalized_ast)
    print(f"\n Classification: {classification}")

    acg = build_acg(normalized_ast)

    print("\n ACG Summary:")
    print(acg)

    S, E, S1, S2, B ,s0= build_game(acg, cgs)

    print("\n Game Construction Summary:")
    print(f"🔹 Total States: {len(S)}")
    print(f"🔹 Transitions: {len(E)}")
    print(f"🔹 Player Accept States (S1): {len(S1)}")
    print(f"🔹 Player Reject States (S2): {len(S2)}")
    print(f"🔹 Büchi Final States (B): {len(B)}")


except ValueError as e:
    print(f"\n Parsing error: {e}")


print("\n Detailed GameProduct States:")
for state in sorted(S, key=str):
    owner = (
        "Accept (S1)" if state in S1 else
        "Reject (S2)" if state in S2 else
        "Unknown (?)"
    )
    print(f"  - {pretty_node(state)}  →  Owner: {owner}")

print("\n Transitions:")
for (src, dst), _ in E.items():
    print(f"  {pretty_node(src)}  →  {pretty_node(dst)}")

print("\n Initial state :\n")
print(f"{pretty_node(s0)}")

print("\n Büchi Final States (B):")
for b_state in sorted(B, key=str):
    print(f"  - {pretty_node(b_state)}")

print("\n Done.")
