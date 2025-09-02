"""
Preprocessing package: aquí exponemos tokens (y más adelante AST, parser, transforms, validator).
Permite escribir:  from preprocessing import LPAREN, RPAREN, SYMBOL_MAP
"""
from .tokens import *  # reexporta los tokens para imports sencillos

# Cuando añadas el resto, podrás reexportarlos aquí:
# from .ast_nodes import *
# from .parser import tokenize, parse
# from .transform import apply_modal_dualities, eliminate_f_and_r, push_negations_to_nnf, normalize_formula
# from .validator import filter