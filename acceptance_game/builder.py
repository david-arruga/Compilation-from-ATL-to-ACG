from .model import GameProduct
from .expansion import generate_initial_game_states, expand_node

def build_game(acg, cgs, *, full_arena=False):
    """Build chapter-4 vertices; optionally seed every configuration Q x S.

    Default exploration follows all edges from the initial configuration.
    Both absorbing sinks are retained even when unreachable.
    """
    cgs.validate()
    if acg.initial_state not in acg.states:
        raise ValueError("ACG initial state is not declared.")
    if not acg.final_states <= acg.states:
        raise ValueError("ACG accepting states must be declared states.")
    product = GameProduct(acg, cgs)
    initial = generate_initial_game_states(product)
    worklist = [initial]
    if full_arena:
        for q in acg.states:
            for s in cgs.states:
                node = ("state", q, s)
                product.states.add(node)
                product.S1.add(node)
                if q in acg.final_states:
                    product.B.add(node)
                worklist.append(node)
    visited = set()
    while worklist:
        node = worklist.pop()
        if node in visited:
            continue
        visited.add(node)
        new_nodes = expand_node(product, node)
        worklist.extend(new_nodes)
    return product.states, product.transitions, product.S1, product.S2, product.B, initial