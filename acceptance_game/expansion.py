from preprocessing.ast_nodes import Top, Bottom
from acg import EpsilonAtom, UniversalAtom, ExistentialAtom
from .utils import generate_possibilities

def generate_initial_game_states(product):
    q0 = product.acg.initial_state
    s0 = product.cgs.initial_state
    initial = ("state", q0, s0)
    product.initial_states.add(initial)
    product.states.add(initial)
    product.S1.add(initial)
    if q0 in product.acg.final_states:
        product.B.add(initial)
    return initial

def expand_from_state_node(product, q, s):
    relevant_props = product.acg.propositions
    projected_label = frozenset(p for p in product.cgs.labeling_function[s] if p in relevant_props)
    delta_formula = product.acg.get_transition(q, projected_label)
    if "true_sink" not in product.states:
        product.states.add("true_sink")
        product.transitions[("true_sink", "true_sink")] = None
        product.S1.add("true_sink")
        product.B.add("true_sink")
    if "false_sink" not in product.states:
        product.states.add("false_sink")
        product.transitions[("false_sink", "false_sink")] = None
        product.S2.add("false_sink")
    if isinstance(delta_formula, Top):
        product.states.add("true_sink")
        product.transitions[(("state", q, s), "true_sink")] = None
        return
    if isinstance(delta_formula, Bottom):
        product.states.add("false_sink")
        product.transitions[(("state", q, s), "false_sink")] = None
        return
    for U in generate_possibilities(delta_formula):
        atom_selection_node = ("atom_selection", q, s, frozenset(U))
        product.states.add(atom_selection_node)
        product.transitions[(("state", q, s), atom_selection_node)] = None
        product.S2.add(atom_selection_node)

def expand_from_atom_selection_node(product, q, s, U):
    for alpha in U:
        if isinstance(alpha, EpsilonAtom):
            q_prime = alpha.state
            new_state = ("state", q_prime, s)
            product.states.add(new_state)
            product.transitions[(("atom_selection", q, s, U), new_state)] = None
            product.S1.add(new_state)
            if q_prime in product.acg.final_states:
                product.B.add(new_state)
        elif isinstance(alpha, UniversalAtom):
            q_prime = alpha.state
            new_node = ("atom_applied", q_prime, s, alpha)
            product.states.add(new_node)
            product.transitions[(("atom_selection", q, s, U), new_node)] = None
            product.S1.add(new_node)
        elif isinstance(alpha,ExistentialAtom):
            q_prime = alpha.state
            new_node = ("atom_applied", q_prime, s, alpha)
            product.states.add(new_node)
            product.transitions[(("atom_selection", q, s, U), new_node)] = None
            product.S2.add(new_node)

def expand_from_atom_applied_node(product, q, s, alpha):
    if isinstance(alpha, UniversalAtom):
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

def expand_from_reject_univ_node(product, q, s, A, alpha, dA_frozen):
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
            if q_prime in product.acg.final_states:
                product.B.add(new_node)

def expand_from_accept_exist_node(product, q, s, agents_prime, alpha, v_reject):
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
            if q_prime in product.acg.final_states:
                product.B.add(new_state)

def expand_node(product, node):
    if isinstance(node, str):
        return []
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