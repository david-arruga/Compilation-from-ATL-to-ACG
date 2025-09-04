from __future__ import annotations
from preprocessing.ast_nodes import Var, Not, And, Or, Modality, Globally, ParseNode
from .lights_cgs import Coalition, full_coalition

def flatG_clause(i: int, A: Coalition):
    p = Var(f"p_{i}")
    body = Globally(p)
    return Modality(A, body)

def generate_flatG_spec(n: int):
    A = full_coalition(n)
    phi = flatG_clause(0, A)
    for i in range(1, n):
        phi = And(phi, flatG_clause(i, A))
    return phi

def generate_flatG_OR_spec(n: int):
    A = full_coalition(n)
    phi = flatG_clause(0, A)
    for i in range(1, n):
        phi = Or(phi, flatG_clause(i, A))
    return phi

def flatG_individual_clause(i: int):
    p = Var(f"p_{i}")
    return Modality([f"ctrl_{i}"], Globally(p))

def generate_flatG_individual_spec(n: int):
    phi = flatG_individual_clause(0)
    for i in range(1, n):
        phi = And(phi, flatG_individual_clause(i))
    return phi

def negated_flatG_clause(i: int, A: Coalition):
    p = Var(f"p_{i}")
    return Not(Modality(A, Globally(p)))

def generate_negated_flatG_spec(n: int):
    A = full_coalition(n)
    phi = negated_flatG_clause(0, A)
    for i in range(1, n):
        phi = And(phi, negated_flatG_clause(i, A))
    return phi

def nested_G_formula(n: int, A: Coalition):
    node = Var(f"p_{n-1}")
    for _ in range(n):
        node = Modality(A, Globally(node))
    return node

def generate_negated_nestedG_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    nested = nested_G_formula(n, A)
    return Not(nested)

def generate_stepwise_negated_nestedG_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    node = Var(f"p_{n-1}")
    for _ in range(n):
        node = Not(Modality(A, Globally(node)))
    return node

def build_deepG_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    return nested_G_formula(n, A)

def nested_G_individual_formula(n: int):
    node = Var(f"p_{n-1}")
    for i in reversed(range(n)):
        node = Modality([f"ctrl_{i}"], Globally(node))
    return node

def generate_nested_G_individual_spec(n: int):
    return nested_G_individual_formula(n)
