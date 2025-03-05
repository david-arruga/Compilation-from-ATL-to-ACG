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

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Var('{self.name}')"

class And(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} and {self.rhs})"

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}And(\n{self.lhs.to_tree(level + 1)},\n{self.rhs.to_tree(level + 1)}\n{indent})"

class Or(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} or {self.rhs})"

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Or(\n{self.lhs.to_tree(level + 1)},\n{self.rhs.to_tree(level + 1)}\n{indent})"

class Not(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"(not {self.sub})"

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Not(\n{self.sub.to_tree(level + 1)}\n{indent})"

class Next(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"( ⃝  {self.sub})"

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Next(\n{self.sub.to_tree(level + 1)}\n{indent})"

class Until(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} U {self.rhs})"

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Until(\n{self.lhs.to_tree(level + 1)},\n{self.rhs.to_tree(level + 1)}\n{indent})"

class Release(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} R {self.rhs})"

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Release(\n{self.lhs.to_tree(level + 1)},\n{self.rhs.to_tree(level + 1)}\n{indent})"

class Globally(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"(□ {self.sub})"

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Globally(\n{self.sub.to_tree(level + 1)}\n{indent})"

class Eventually(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"(◇ {self.sub})"

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Eventually(\n{self.sub.to_tree(level + 1)}\n{indent})"

class Implies(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} -> {self.rhs})"

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Implies(\n{self.lhs.to_tree(level + 1)},\n{self.rhs.to_tree(level + 1)}\n{indent})"

class Iff(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} <-> {self.rhs})"

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Iff(\n{self.lhs.to_tree(level + 1)},\n{self.rhs.to_tree(level + 1)}\n{indent})"

class Modality(ParseNode):
    def __init__(self, agents, sub):
        self.agents = agents
        self.sub = sub

    def to_formula(self):
        agents_str = ", ".join(self.agents)
        return f"<{agents_str}> {self.sub}"

    def to_tree(self, level=0):
        indent = "    " * level
        agents_str = ", ".join(self.agents)
        return f"{indent}Modality([{agents_str}],\n{self.sub.to_tree(level + 1)}\n{indent})"
    
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
                "universal": [],
                "existential": [],
                "epsilon": []
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

        elif symbol.isalpha():
            buffer = symbol
            cursor += 1
            while cursor < length and source[cursor].isalnum():
                buffer += source[cursor]
                cursor += 1

            if inside_agents:
                tokens.append((AGENT_NAME, buffer))
            else:
                token_type = keywords.get(buffer, PROPOSITION)
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
            return Next(parse_until()) 

        elif token[0] == GLOBALLY:
            advance()
            return Globally(parse_temporal())

        elif token[0] == EVENTUALLY:
            advance()
            return Eventually(parse_temporal())

        else:
            return parse_atomic_formula()  

    def parse_until():
       
        result = parse_temporal()
        while current_token()[0] in {UNTIL, RELEASE}:
            token_type = current_token()[0]
            advance()
            rhs = parse_temporal() 
            if token_type == UNTIL:
                result = Until(result, rhs)
            elif token_type == RELEASE:
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

    return parse_formula()

def formula_validity(tokens, strict_ATL=True):

    cursor = 0
    limit = len(tokens)
    open_parentheses = 0
    open_modalities = 0  
    last_modality_agents = None  
    inside_modality = False  
    inside_boolean = False  

    def advance():
        nonlocal cursor
        if cursor < limit:
            cursor += 1

    def current_token():
        return tokens[cursor] if cursor < limit else (END_OF_INPUT, "")

    while cursor < limit:
        token = current_token()

        if token[0] == LPAREN:
            open_parentheses += 1
        elif token[0] == RPAREN:
            if open_parentheses > 0:
                open_parentheses -= 1
            else:
                print("Syntax error: Unmatched closing parenthesis.")
                return None  

        elif token[0] == LTRI:
            open_modalities += 1  
            inside_modality = True  
            advance()


            agents = []
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()

            if current_token()[0] != RTRI:
                print("Syntax error: Modality incorrectly closed.")
                return None  
            advance()  
            open_modalities -= 1  

            next_token = current_token()
            if strict_ATL and next_token[0] not in {NEXT, GLOBALLY, EVENTUALLY, UNTIL, RELEASE}:
                print(f"Error: Modality <A> applied to '{next_token[1]}', which is not allowed in ATL.")
                return None  

            if strict_ATL:
                agent_set = frozenset(agents)  
                if last_modality_agents is None:
                    last_modality_agents = agent_set  
                elif last_modality_agents != agent_set:
                    print("Error: Different agent sets detected in ATL.")
                    return None  

        elif token[0] in {AND, OR}:
            inside_boolean = True  
            next_token = tokens[cursor + 1] if cursor + 1 < limit else (END_OF_INPUT, "")

            if strict_ATL and next_token[0] == LTRI and not inside_modality:
                print("Error: Modality <A> found inside OR/AND without proper coverage.")
                return None  

        elif token[0] == RTRI:
            inside_modality = False  

        inside_boolean = False  
        advance()

    if open_parentheses != 0:
        print(f"Syntax error: Unmatched opening parenthesis ({open_parentheses} left open).")
        return None  
    if open_modalities != 0:
        print(f"Syntax error: Unmatched modality opening ({open_modalities} left open).")
        return None  

    return tokens  

def build_acg(formula):
  
    acg = ACG()  

    def collect_subformulas(node):
        if node not in acg.states:
            acg.add_state(node)  

            if isinstance(node, Var):  
                acg.add_proposition(node.name)  

            for child in getattr(node, "__dict__", {}).values():
                if isinstance(child, ParseNode):
                    collect_subformulas(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, ParseNode):
                            collect_subformulas(item)

    collect_subformulas(formula)  
    acg.add_initial_state(formula)  

    return acg  



source = "<A> globally (p or <A> eventually q)"
tokens = tokenize(source)

try:
    formula = formula_validity(tokens, strict_ATL=True)  
    ast = parse(tokens)
    acg = build_acg(ast)
    print("\nACG Representation:")
    print(acg)

except ValueError as e:
    print(e)


