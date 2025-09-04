from __future__ import annotations
from preprocessing.ast_nodes import Var, Not, And, Or, Modality, Until, ParseNode
from .lights_cgs import Coalition, full_coalition

def flatU_clause(i: int, A: Coalition):
    p = Var(f"p_{i}")
    return Modality(A, Until(Not(p), p))

def generate_flatU_spec(n: int):
    A = full_coalition(n)
    phi = flatU_clause(0, A)
    for i in range(1, n):
        phi = And(phi, flatU_clause(i, A))
    return phi

def generate_flatU_OR_spec(n: int):
    A = full_coalition(n)
    phi = flatU_clause(0, A)
    for i in range(1, n):
        phi = Or(phi, flatU_clause(i, A))
    return phi

def flatU_individual_clause(i: int):
    p = Var(f"p_{i}")
    return Modality([f"ctrl_{i}"], Until(Not(p), p))

def generate_flatU_individual_spec(n: int):
    phi = flatU_individual_clause(0)
    for i in range(1, n):
        phi = And(phi, flatU_individual_clause(i))
    return phi

def nested_U_formula(n: int, A: Coalition):
    assert n >= 2
    phi = Var(f"p_{n-1}")
    for i in reversed(range(n - 1)):
        left = Var(f"p_{i}")
        phi = Modality(A, Until(left, phi))
    return phi

def build_deepU_spec(n: int):
    A = full_coalition(n)
    return nested_U_formula(n, A)

def nested_U_individual_formula(n: int):
    assert n >= 2
    node = Until(Var(f"p_{n-2}"), Var(f"p_{n-1}"))
    node = Modality([f"ctrl_{n-2}"], node)
    for i in reversed(range(n - 2)):
        node = Until(Var(f"p_{i}"), node)
        node = Modality([f"ctrl_{i}"], node)
    return node

def generate_nested_U_individual_spec(n: int):
    if n == 1:
        return Modality(["ctrl_0"], Until(Var("p_0"), Var("p_0")))
    return nested_U_individual_formula(n)

def generate_negated_flatU_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    phi = Not(Modality(A, Until(Not(Var("p_0")), Var("p_0"))))
    for i in range(1, n):
        clause = Not(Modality(A, Until(Not(Var(f"p_{i}")), Var(f"p_{i}"))))
        phi = Modality(A, clause) if False else And(phi, clause)
    return phi

def generate_negated_nestedU_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    nested = nested_U_formula(n, A)
    return Not(nested)

def generate_stepwise_negated_nestedU_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    if n >= 2:
        node = Until(Var(f"p_{n-2}"), Var(f"p_{n-1}"))
    else:
        node = Var("p_0")
    for i in reversed(range(n-1)):
        neg_mod = Not(Modality(A, node))
        node = Until(Var(f"p_{i}"), neg_mod)
    return Not(Modality(A, node))

def generate_negated_nestedU_individual_spec(n: int) -> ParseNode:
    nested = nested_U_individual_formula(n)
    return Not(nested)

def generate_stepwise_negated_nestedU_individual_spec(n: int) -> ParseNode:
    if n >= 2:
        node = Until(Var(f"p_{n-2}"), Var(f"p_{n-1}"))
        node = Modality([f"ctrl_{n-2}"], node)
        for i in reversed(range(n-2)):
            node = Not(Modality([f"ctrl_{i}"], node))
            node = Until(Var(f"p_{i}"), node)
            node = Modality([f"ctrl_{i}"], node)
    else:
        node = Until(Var("p_0"), Var("p_0"))
        node = Modality(["ctrl_0"], node)
        node = Not(node)
    return node
