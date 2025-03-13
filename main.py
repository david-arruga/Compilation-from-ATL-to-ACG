from itertools import chain, combinations

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

SYMBOL_MAP = {
    "◯": NEXT,
    "□": GLOBALLY,
    "◇": EVENTUALLY,
    "U": UNTIL,
    "R": RELEASE
}

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
        elif symbol == ",":
            tokens.append((COMMA, symbol))
        elif symbol == "|":
            tokens.append((OR, symbol))
        elif symbol == "&":
            tokens.append((AND, symbol))
        elif symbol in SYMBOL_MAP:
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
            subformula = parse_temporal()  

            return Eventually(subformula)  

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
            
            subformula = parse_temporal()
            return Modality(agents, subformula)

        else:
            return parse_atomic_formula()  


    def parse_until():
        result = parse_temporal()

        while current_token()[0] in {UNTIL, RELEASE}:
            token_type = current_token()[0]
            advance()
            rhs = parse_temporal()

            if isinstance(result, Modality):
                result = Modality(result.agents, Until(result.sub, rhs))  
            else:
                result = Until(result, rhs)

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

def transform_to_fundamental(node):

    if isinstance(node, Eventually):
        return Until(T(), transform_to_fundamental(node.sub), generated_from_eventually=True)  

    elif isinstance(node, Release):
        lhs_transformed = transform_to_fundamental(node.lhs)
        rhs_transformed = transform_to_fundamental(node.rhs)
        
        if isinstance(lhs_transformed, Modality):
            return Modality(lhs_transformed.agents, Not(Until(Not(lhs_transformed.sub), Not(rhs_transformed))))
        
        return Not(Until(Not(lhs_transformed), Not(rhs_transformed)))

    elif isinstance(node, And):
        return And(transform_to_fundamental(node.lhs), transform_to_fundamental(node.rhs))

    elif isinstance(node, Or):
        return Or(transform_to_fundamental(node.lhs), transform_to_fundamental(node.rhs))

    elif isinstance(node, Not):
        return Not(transform_to_fundamental(node.sub))

    elif isinstance(node, Next):
        return Next(transform_to_fundamental(node.sub))

    elif isinstance(node, Until):
        return Until(transform_to_fundamental(node.lhs), transform_to_fundamental(node.rhs))


    elif isinstance(node, Globally):
        return Globally(transform_to_fundamental(node.sub))

    elif isinstance(node, Modality):
        return Modality(node.agents, transform_to_fundamental(node.sub))

    return node

def filter(ast, strict_ATL=True):

    has_modality = False  
    agents_set = None  

    def check_invalid(node):
        nonlocal has_modality

        if isinstance(node, Modality):
            has_modality = True  

            if not isinstance(node.sub, ParseNode) or not isinstance(node.agents, list):
                print(" ERROR: Modality must have valid agents and subformula.")
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

    def check_strict_ATL(node, parent=None):
        nonlocal has_modality, agents_set

        if isinstance(node, Modality):
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
            # else:
            #    if not isinstance(node.rhs, (Var, Modality)):
            #        print(" ERROR: φ₂ in Until must be modal or atomic.")
            #        return "ATL* but not ATL"

        if isinstance(node, (Next, Globally, Until)):
            if not isinstance(parent, Modality):  
                print(f" ERROR: {node.__class__.__name__}  is not immediately preceded by a modality.")
                return "ATL* but not ATL"

        for key, child in node.__dict__.items():
            if isinstance(child, ParseNode):
                result = check_strict_ATL(child, node)  
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

    def is_subformula(node):
        
        if node == ast:
            return True

        if isinstance(node, Var):
            return True

        if isinstance(node, Modality):
            return True

        if isinstance(node, (Next, Globally)):
            return False

        if isinstance(node, Until) and not isinstance(node, Modality):
            return False

        return True  


    def traverse(node):
        
        if is_subformula(node):
            closure.add(node)

        for child in getattr(node, "__dict__", {}).values():
            if isinstance(child, ParseNode):
                traverse(child)

    traverse(ast)
    return closure

def generate_transitions(acg):
    
    for state in acg.states:
        for sigma in acg.alphabet:  
            
            if isinstance(state, Var):  
                if state.name in sigma:
                    acg.add_transition(state, sigma, Top())  
                else:
                    acg.add_transition(state, sigma, Bottom())  

            elif isinstance(state, Not) and isinstance(state.sub, Var):  
                negated_prop = state.sub.name
                if negated_prop in sigma:
                    acg.add_transition(state, sigma, Bottom())  
                else:
                    acg.add_transition(state, sigma, Top())  

            elif isinstance(state, And):  
                acg.add_transition(state, sigma, Conj(EpsilonAtom(state.lhs), EpsilonAtom(state.rhs)))  

            elif isinstance(state, Or):  
                acg.add_transition(state, sigma, Disj(EpsilonAtom(state.lhs), EpsilonAtom(state.rhs)))  

            elif isinstance(state, Modality) and isinstance(state.sub, Next):
                next_state = state.sub.sub  
                agents = frozenset(state.agents)  
                acg.add_transition(state, sigma, ExistentialAtom(next_state, agents))  

            elif isinstance(state, Modality) and isinstance(state.sub, Globally):
                phi = state.sub.sub  
                agents = frozenset(state.agents)  
                acg.add_transition(state, sigma, Conj(EpsilonAtom(phi), ExistentialAtom(state, agents)))

            elif isinstance(state, Modality) and isinstance(state.sub, Until):
                phi1 = state.sub.lhs  
                phi2 = state.sub.rhs  
                agents = frozenset(state.agents)  
                acg.add_transition(state, sigma, Disj(EpsilonAtom(phi2), Conj(EpsilonAtom(phi1), ExistentialAtom(state, agents))))

def build_acg(transformed_ast):
    
    ap_set = extract_propositions(transformed_ast)
    alphabet = set(frozenset(s) for s in chain.from_iterable(combinations(ap_set, r) for r in range(len(ap_set) + 1)))
    acg = ACG()
    acg.propositions = ap_set
    acg.alphabet = alphabet 
    closure = generate_closure(transformed_ast)
    acg.states = closure  
    acg.initial_state = transformed_ast  
    generate_transitions(acg)

    return acg  


test_formulas = [
        "<A> next (p and q)",
        "<A> eventually (p or q)",
        "<A> eventually (<A> p until q)",
        "<A> next (p or (q and <A> eventually r))"
    ]

for formula in test_formulas:
    print("=" * 70)
    print(f" Original Formula : {formula}")

    tokens = tokenize(formula)
    print(f" Tokens: {tokens}")

    try:
        ast = parse(tokens)
        print(" Initial AST :")
        print(ast.to_tree())

        transformed_ast = transform_to_fundamental(ast)
        print(" Transformed AST :")
        print(transformed_ast.to_tree())

        reconstructed_formula = transformed_ast.to_formula()
        print(f" Reconstructed Formula : {reconstructed_formula}")

        result = filter(transformed_ast, strict_ATL=True)
        print(f" Filter result :  {result}")

        if result == "ATL":
            acg = build_acg(transformed_ast)
            print(acg)

        else:
            print(" Formula is not ATL, unable to build ACG")

    except ValueError as e:
        print(f" Parsing error : {e}")

    print("=" * 70, "\n")
