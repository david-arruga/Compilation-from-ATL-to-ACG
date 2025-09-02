# filepath: preprocessing/transform.py
"""Transformaciones: dualidades, eliminación de F/R, empuje de negaciones y normalización."""
from __future__ import annotations

from .ast_nodes import (
    ParseNode, T, F, Var, And, Or, Not, Next, Until, Release,
    Globally, Eventually, Implies, Iff, Modality, DualModality,
)


def apply_modal_dualities(node: ParseNode) -> ParseNode:
    if isinstance(node, DualModality):
        sub = apply_modal_dualities(node.sub)
        agents = node.agents
        if isinstance(sub, Next):
            return Not(Modality(agents, Next(Not(sub.sub))))
        elif isinstance(sub, Globally):
            return Not(Modality(agents, Eventually(Not(sub.sub))))
        elif isinstance(sub, Eventually):
            return Not(Modality(agents, Globally(Not(sub.sub))))
        else:
            return DualModality(agents, sub)
    elif isinstance(node, Not):
        return Not(apply_modal_dualities(node.sub))
    elif isinstance(node, And):
        return And(apply_modal_dualities(node.lhs), apply_modal_dualities(node.rhs))
    elif isinstance(node, Or):
        return Or(apply_modal_dualities(node.lhs), apply_modal_dualities(node.rhs))
    elif isinstance(node, Implies):
        return Implies(apply_modal_dualities(node.lhs), apply_modal_dualities(node.rhs))
    elif isinstance(node, Iff):
        return Iff(apply_modal_dualities(node.lhs), apply_modal_dualities(node.rhs))
    elif isinstance(node, Next):
        return Next(apply_modal_dualities(node.sub))
    elif isinstance(node, Until):
        return Until(apply_modal_dualities(node.lhs), apply_modal_dualities(node.rhs))
    elif isinstance(node, Globally):
        return Globally(apply_modal_dualities(node.sub))
    elif isinstance(node, Eventually):
        return Eventually(apply_modal_dualities(node.sub))
    elif isinstance(node, Modality):
        return Modality(node.agents, apply_modal_dualities(node.sub))
    return node


def eliminate_f_and_r(node: ParseNode) -> ParseNode:
    if isinstance(node, Release):
        lhs = eliminate_f_and_r(node.lhs)
        rhs = eliminate_f_and_r(node.rhs)
        return Not(Until(Not(lhs), Not(rhs)))
    elif isinstance(node, Implies):
        lhs = eliminate_f_and_r(node.lhs)
        rhs = eliminate_f_and_r(node.rhs)
        return Or(Not(lhs), rhs)
    elif isinstance(node, Iff):
        lhs = eliminate_f_and_r(node.lhs)
        rhs = eliminate_f_and_r(node.rhs)
        return And(Or(Not(lhs), rhs), Or(Not(rhs), lhs))
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
                push_negations_to_nnf(Not(sub.rhs)),
            )
        if isinstance(sub, Or):
            return And(
                push_negations_to_nnf(Not(sub.lhs)),
                push_negations_to_nnf(Not(sub.rhs)),
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
    # Si alguna vez quieres transformar DualModality en NNF, descomenta y define las reglas.
    return node


def normalize_formula(ast: ParseNode) -> ParseNode:
    previous = None
    current = ast
    while previous != current:
        previous = current
        current = eliminate_f_and_r(current)
        current = push_negations_to_nnf(current)
    return current


__all__ = [
    "apply_modal_dualities",
    "eliminate_f_and_r",
    "push_negations_to_nnf",
    "normalize_formula",
]
