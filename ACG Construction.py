
LPAREN, RPAREN = 1, 2
LBRACE, RBRACE = 3, 4
LTRI, RTRI = 5, 6
COMMA = 7
AND, OR, NOT = 8, 9, 10
IMPLIES, IFF = 11, 12
NEXT, UNTIL, RELEASE = 13, 14, 15
GLOBALLY, EVENTUALLY = 16, 17
MODALITY_START, MODALITY_END = 18, 19
NAME, UNKNOWN, END_OF_INPUT = 20, 21, 22
AGENT_NAME = NAME 


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

        if (state_from, input_symbol) not in self.transitions:
            self.transitions[(state_from, input_symbol)] = {
                "universal": [],
                "existential": [],
                "epsilon": []
            }

        if atom_type in ["universal", "existential"]:
            if agents is None:
                raise ValueError(f"Transition type '{atom_type}' requires a set of agents.")
            transition_tuple = (state_to, atom_type, frozenset(agents))  # Store as (q', type, A)
        else:  
            transition_tuple = (state_to, atom_type)

        self.transitions[(state_from, input_symbol)][atom_type].append(transition_tuple)


    def get_transitions(self, state, input_symbol):
       
        return self.transitions.get((state, input_symbol), {"universal": [], "existential": [], "epsilon": []})




# Tokenization function 
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
        "eventually": EVENTUALLY
    }
    while cursor < length:
        symbol = source[cursor]

        if symbol in " \t":
            pass 
        elif symbol == "(":
            tokens.append((LPAREN, symbol))
        elif symbol == ")":
            tokens.append((RPAREN, symbol))
        elif symbol == "[":
            tokens.append((LBRACE, symbol))
        elif symbol == "]":
            tokens.append((RBRACE, symbol))
        elif symbol == "<":
            tokens.append((LTRI, symbol))
        elif symbol == ">":
            tokens.append((RTRI, symbol))
        elif symbol == ",":
            tokens.append((COMMA, symbol))
        elif symbol == "&":
            tokens.append((AND, symbol))
        elif symbol == "|":
            tokens.append((OR, symbol))
        elif symbol == "-" and cursor + 1 < length and source[cursor + 1] == ">":
            tokens.append((IMPLIES, "->"))
            cursor += 1
        elif symbol == "<" and cursor + 1 < length and source[cursor + 1] == "-":
            tokens.append((IFF, "<->"))
            cursor += 1
        elif symbol.isalpha():
            buffer = symbol
            cursor += 1
            while cursor < length and source[cursor].isalpha():
                buffer += source[cursor]
                cursor += 1
            token_type = keywords.get(buffer, NAME)
            tokens.append((token_type, buffer))
            cursor -= 1  
        else:
            tokens.append((UNKNOWN, symbol))   

        cursor += 1
    return tokens


# Parsing function 
def parse(tokens):
    cursor = 0
    limit = len(tokens)

    def advance():
        nonlocal cursor
        cursor += 1

    def current_token():
        return tokens[cursor] if cursor < limit else (END_OF_INPUT, "")

    def parse_atomic_formula():
        token = current_token()
        if token[0] == LPAREN:
            advance()
            result = parse_formula()
            if current_token()[0] != RPAREN:
                raise ValueError("Parse error, expecting a closing parenthesis.")
            advance()
            return result
        elif token[0] == NAME:
            result = Var(token[1])
            advance()
            return result
        elif token[0] == NOT:
            advance()
            subformula = parse_atomic_formula()
            return Not(subformula)
        elif token[0] == NEXT:
            advance()
            subformula = parse_atomic_formula()
            return Next(subformula)
        elif token[0] == GLOBALLY:
            advance()
            subformula = parse_atomic_formula()
            return Globally(subformula)
        elif token[0] == EVENTUALLY:
            advance()
            subformula = parse_atomic_formula()
            return Eventually(subformula)
        elif token[0] == LTRI:
            agents = []
            advance()
            while current_token()[0] == NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()
            if current_token()[0] != RTRI:
                raise ValueError("Parse error, expecting a closing > for a modality.")
            advance()
            subformula = parse_atomic_formula()
            return Modality(agents, subformula)
        else:
            raise ValueError(f"Parse error, unexpected token: {token}")

    def parse_disjunction():
        result = parse_atomic_formula()
        while current_token()[0] == OR:
            advance()
            result = Or(result, parse_atomic_formula())
        return result

    def parse_conjunction():
        result = parse_disjunction()
        while current_token()[0] == AND:
            advance()
            result = And(result, parse_disjunction())
        return result

    def parse_until():
        result = parse_conjunction()
        while current_token()[0] == UNTIL or current_token()[0] == RELEASE:
            token_type = current_token()[0]
            advance()
            if token_type == UNTIL:
                result = Until(result, parse_conjunction())
            elif token_type == RELEASE:
                result = Release(result, parse_conjunction())
        return result

    def parse_formula():
        result = parse_until()
        if current_token()[0] == IMPLIES:
            advance()
            result = Implies(result, parse_formula())
        elif current_token()[0] == IFF:
            advance()
            result = Iff(result, parse_formula())
        return result

    return parse_formula()


source = "<Angel, Carlos> globally (<Juan, Carlos> eventually p)"
print(f"Input formula: \n{source}\n")
tokens = tokenize(source)
print("Generated tokens:\n", tokens)
parsed_formula = parse(tokens)
print("\nParsed formula")
print(parsed_formula)
print("\nParsed tree:\n")
print(parsed_formula.to_tree())
