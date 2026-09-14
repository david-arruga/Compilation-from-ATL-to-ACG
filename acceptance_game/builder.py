from .model import GameProduct
from .expansion import generate_initial_game_states, expand_node

def build_game(acg, cgs):
    cgs.validate()
    if acg.initial_state not in acg.states:
        raise ValueError("ACG initial state is not declared.")
    if not acg.final_states <= acg.states:
        raise ValueError("ACG accepting states must be declared states.")
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
    return product.states, product.transitions, product.S1, product.S2, product.B, initial