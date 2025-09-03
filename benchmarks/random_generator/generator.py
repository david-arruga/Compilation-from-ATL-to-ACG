from __future__ import annotations
import random
from itertools import chain, combinations
from preprocessing.ast_nodes import Var, Not, And, Or, Modality, Next, Globally, Until
from preprocessing.transformer import normalize_formula
from preprocessing.validator import filter

def extract_props_and_agents_from_cgs(cgs):
    return list(cgs.get_propositions()), list(cgs.get_agents())

def powerset_nonempty(lst):
    return [list(subset) for subset in chain.from_iterable(combinations(lst, r) for r in range(1, len(lst)+1))]

def random_var(cgs):
    prop_pool, _ = extract_props_and_agents_from_cgs(cgs)
    return Var(random.choice(prop_pool))

def random_modality_temporal_subformula(cgs,depth):
    _, agent_pool = extract_props_and_agents_from_cgs(cgs)
    coalition = random.choice(powerset_nonempty(agent_pool))
    op = random.choice(['next', 'globally', 'until'])
    if op == 'next':
        temp = Next(random_node(cgs, depth - 2, inside_modality=True, modality_needed=False))
    elif op == 'globally':
        temp = Globally(random_node(cgs, depth - 2, inside_modality=True, modality_needed=False))
    elif op == 'until':
        temp = Until(
            random_node(cgs, depth - 2, inside_modality=True, modality_needed=False),
            random_node(cgs, depth - 2, inside_modality=True, modality_needed=False)
        )
    return Modality(coalition, temp)

def random_node(cgs, depth, inside_modality=False, modality_needed=True):
    if depth <= 1:
        return random_var(cgs)
    if modality_needed and depth >= 3 and random.random() < 1 / depth:
        return random_modality_temporal_subformula(cgs, depth)
    op = random.choice(['and', 'or', 'not', 'modality_temporal'])
    if op == 'and':
        left_needs  = modality_needed and random.choice([True, False])
        right_needs = modality_needed and not left_needs
        return And(
            random_node(cgs, depth-1, inside_modality, left_needs),
            random_node(cgs, depth-1, inside_modality, right_needs)
        )
    elif op == 'or':
        left_needs  = modality_needed and random.choice([True, False])
        right_needs = modality_needed and not left_needs
        return Or(
            random_node(cgs, depth-1, inside_modality, left_needs),
            random_node(cgs, depth-1, inside_modality, right_needs)
        )
    elif op == 'not':
        return Not(random_node(cgs, depth-1, inside_modality, modality_needed))
    elif op == 'modality_temporal' and not inside_modality:
        return random_modality_temporal_subformula(cgs, depth)
    else:
        return random_var(cgs)

def formula_depth(node):
    if isinstance(node, Var):
        return 1
    if isinstance(node, Not):
        return 1 + formula_depth(node.sub)
    if isinstance(node, (And, Or)):
        return 1 + max(formula_depth(node.lhs), formula_depth(node.rhs))
    if isinstance(node, Modality):
        return 1 + formula_depth(node.sub)
    if isinstance(node, (Next, Globally)):
        return 1 + formula_depth(node.sub)
    if isinstance(node, Until):
        return 1 + max(formula_depth(node.lhs), formula_depth(node.rhs))
    return 1

def generate_random_valid_atl_formula(cgs, depth, modality_needed=True, max_tries=10_000):
    tries = 0
    while tries < max_tries:
        raw = random_node(cgs, depth=depth, modality_needed=modality_needed)
        f = normalize_formula(raw)
        if filter(f) == "ATL" and formula_depth(f) == depth:
            return f
        tries += 1
    raise RuntimeError(f"No se pudo generar fórmula de profundidad {depth} tras {max_tries} intentos")

def generate_valid_formulas_by_depth(cgs, min_depth: int, max_depth: int, samples_per_depth: int):
    for depth in range(min_depth, max_depth + 1):
        print(f"\n Depth {depth}")
        for i in range(1, samples_per_depth + 1):
            f = generate_random_valid_atl_formula(cgs, depth, modality_needed=(depth >= 3))
            assert formula_depth(f) == depth
            print(f"  ✔️ {f.to_formula()}")
    print("\n")
    print("\n")