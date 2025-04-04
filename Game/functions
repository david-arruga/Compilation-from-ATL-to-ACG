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

