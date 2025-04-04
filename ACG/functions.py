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
