from preprocessing.ast_nodes import Top, Bottom, Conj, Disj
from acg import EpsilonAtom, UniversalAtom, ExistentialAtom

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

def pretty_node(node):
    if isinstance(node, str):
        if node == "true_sink":
            return "('true_sink')"
        elif node == "false_sink":
            return "('false_sink')"
        else:
            return str(node)
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