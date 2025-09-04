from __future__ import annotations
from preprocessing.ast_nodes import Var, Not, And, Or, Modality, Next
from .lights_cgs import Coalition, full_coalition

def flatX_clause(i: int, A: Coalition):
    p = Var(f"p_{i}")
    return Modality(A, Next(p))

def generate_flatX_spec(n: int):
    A = full_coalition(n)
    psi = flatX_clause(0, A)
    for i in range(1, n):
        psi = And(psi, flatX_clause(i, A))
    return psi

def generate_flatX_OR_spec(n: int):
    A = full_coalition(n)
    psi = flatX_clause(0, A)
    for i in range(1, n):
        psi = Or(psi, flatX_clause(i, A))
    return psi

def flatX_individual_clause(i: int):
    p = Var(f"p_{i}")
    return Modality([f"ctrl_{i}"], Next(p))

def generate_flatX_individual_spec(n: int):
    phi = flatX_individual_clause(0)
    for i in range(1, n):
        phi = And(phi, flatX_individual_clause(i))
    return phi

def negated_flatX_clause(i: int, A: Coalition):
    p = Var(f"p_{i}")
    return Not(Modality(A, Next(p)))

def generate_negated_flatX_spec(n: int):
    A = full_coalition(n)
    phi = negated_flatX_clause(0, A)
    for i in range(1, n):
        phi = And(phi, negated_flatX_clause(i, A))
    return phi

def nested_X_formula(n: int, A: Coalition):
    node = Var(f"p_{n-1}")
    for _ in range(n):
        node = Modality(A, Next(node))
    return node

def generate_negated_nestedX_spec(n: int):
    A = full_coalition(n)
    nested = nested_X_formula(n, A)
    return Not(nested)

def generate_stepwise_negated_nestedX_spec(n: int):
    A = full_coalition(n)
    node = Var(f"p_{n-1}")
    for _ in range(n):
        node = Not(Modality(A, Next(node)))
    return node

def generate_negated_nestedX_individual_spec(n: int):
    nested = nested_X_individual_formula(n)
    return Not(nested)

def generate_stepwise_negated_nestedX_individual_spec(n: int):
    node = Var(f"p_{n-1}")
    for i in reversed(range(n)):
        node = Not(Modality([f"ctrl_{i}"], Next(node)))
    return node

def build_deepX_spec(n: int):
    A = full_coalition(n)
    return nested_X_formula(n, A)

def nested_X_individual_formula(n: int):
    node = Var(f"p_{n-1}")
    for i in reversed(range(n)):
        node = Modality([f"ctrl_{i}"], Next(node))
    return node

def build_nestedX_individual_spec(n: int):
    return nested_X_individual_formula(n)