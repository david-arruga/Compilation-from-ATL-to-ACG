"""
Preprocessing package: tokens, AST, parser, transformaciones y validador.
Permite:
    from preprocessing import tokenize, parse, normalize_formula, filter
"""
from .tokens import *  # LPAREN, RPAREN, ..., SYMBOL_MAP

from .ast_nodes import (
    ParseNode, T, F, Var, And, Or, Not, Next, Until, Release,
    Globally, Eventually, Implies, Iff, Modality, DualModality,
    Top, Bottom, Conj, Disj,
)

from .parser import tokenize, parse

# Si tu archivo se llama transformer.py, cambia a: from .transformer import ...
from .transformer import (
    apply_modal_dualities,
    eliminate_f_and_r,
    push_negations_to_nnf,
    normalize_formula,
)

from .validator import filter

__all__ = [
    # tokens
    "LPAREN", "RPAREN", "LBRACE", "RBRACE", "LTRI", "RTRI", "COMMA",
    "AND", "OR", "NOT", "IMPLIES", "IFF", "NEXT", "UNTIL", "RELEASE",
    "GLOBALLY", "EVENTUALLY", "PROPOSITION", "AGENT_NAME", "NAME",
    "UNKNOWN", "END_OF_INPUT", "LBRACKET", "RBRACKET", "SYMBOL_MAP",
    # AST
    "ParseNode", "T", "F", "Var", "And", "Or", "Not", "Next", "Until",
    "Release", "Globally", "Eventually", "Implies", "Iff",
    "Modality", "DualModality", "Top", "Bottom", "Conj", "Disj",
    # funciones
    "tokenize", "parse", "apply_modal_dualities", "eliminate_f_and_r",
    "push_negations_to_nnf", "normalize_formula", "filter",
]