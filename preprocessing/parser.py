"""ATL surface syntax.

Precedence (tight to loose): unary, U/R, and, or, implies, iff.
U/R and implication associate right; iff associates left. Temporal state
operands containing Boolean operators must be parenthesized. <A>(p U q)
and the legacy <A>p U q are accepted; parentheses never move a modality.
The compiler's normalized-fragment validator remains a separate check.
"""
from .tokens import *
from .ast_nodes import (T, F, Var, And, Or, Not, Next, Until, Release,
                       Globally, Eventually, Implies, Iff, Modality, DualModality)

TRUE, FALSE = 25, 26


def tokenize(source: str):
    keywords = dict(and_=AND, or_=OR, not_=NOT)
    keywords = {k.rstrip('_'): v for k, v in keywords.items()}
    keywords.update(next=NEXT, until=UNTIL, release=RELEASE, globally=GLOBALLY,
                    eventually=EVENTUALLY, implies=IMPLIES, iff=IFF,
                    true=TRUE, false=FALSE)
    symbols = {**SYMBOL_MAP, '(': LPAREN, ')': RPAREN, '<': LTRI, '>': RTRI,
               '[': LBRACKET, ']': RBRACKET, ',': COMMA, '|': OR, '&': AND,
               '¬': NOT, '!': NOT, '∧': AND, '∨': OR, '→': IMPLIES,
               '↔': IFF, '⊤': TRUE, '⊥': FALSE}
    tokens, i, in_agents = [], 0, False
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if not in_agents and (source.startswith('<->', i) or source.startswith('->', i)):
            spelling = '<->' if source.startswith('<->', i) else '->'
            tokens.append((IFF if spelling == '<->' else IMPLIES, spelling))
            i += len(spelling)
            continue
        if ch.isalpha() or ch == '_':
            j = i + 1
            while j < len(source) and (source[j].isalnum() or source[j] == '_'):
                j += 1
            word = source[i:j]
            kind = AGENT_NAME if in_agents else keywords.get(
                word.lower(), SYMBOL_MAP.get(word, PROPOSITION))
            tokens.append((kind, word))
            i = j
            continue
        if ch not in symbols:
            raise ValueError(f"Unexpected character {ch!r} at offset {i}.")
        tokens.append((symbols[ch], ch))
        if ch in '<[': in_agents = True
        if ch in '>]': in_agents = False
        i += 1
    return tokens


def parse(tokens):
    cursor = 0

    def peek():
        return tokens[cursor][0] if cursor < len(tokens) else END_OF_INPUT

    def take(kind=None):
        nonlocal cursor
        if cursor >= len(tokens) or (kind is not None and peek() != kind):
            raise ValueError(f"Unexpected token at position {cursor}; expected {kind}.")
        token = tokens[cursor]
        cursor += 1
        return token[1]

    def unary():
        kind = peek()
        if kind == LPAREN:
            take()
            value = equivalence()
            take(RPAREN)
            return value
        if kind == PROPOSITION: return Var(take())
        if kind in (TRUE, FALSE):
            take()
            return T() if kind == TRUE else F()
        if kind in (NOT, NEXT, GLOBALLY, EVENTUALLY):
            take()
            return {NOT: Not, NEXT: Next, GLOBALLY: Globally,
                    EVENTUALLY: Eventually}[kind](unary())
        if kind in (LTRI, LBRACKET):
            take()
            close = RTRI if kind == LTRI else RBRACKET
            agents = []
            if peek() != close:
                agents.append(take(AGENT_NAME))
                while peek() == COMMA:
                    take()
                    agents.append(take(AGENT_NAME))
            take(close)
            parenthesized = peek() == LPAREN
            body = unary()
            if not parenthesized and peek() in (UNTIL, RELEASE):
                op = peek()
                take()
                body = (Until if op == UNTIL else Release)(body, temporal())
            return (Modality if kind == LTRI else DualModality)(agents, body)
        raise ValueError(f"Expected a formula at token {cursor}.")

    def temporal():
        left = unary()
        if peek() in (UNTIL, RELEASE):
            op = peek()
            take()
            return (Until if op == UNTIL else Release)(left, temporal())
        return left

    def conjunction():
        left = temporal()
        while peek() == AND:
            take()
            left = And(left, temporal())
        return left

    def disjunction():
        left = conjunction()
        while peek() == OR:
            take()
            left = Or(left, conjunction())
        return left

    def implication():
        left = disjunction()
        if peek() == IMPLIES:
            take()
            return Implies(left, implication())
        return left

    def equivalence():
        left = implication()
        while peek() == IFF:
            take()
            left = Iff(left, implication())
        return left

    result = equivalence()
    if cursor != len(tokens):
        raise ValueError(f"Unexpected trailing token {tokens[cursor][1]!r}.")
    return result


__all__ = ['tokenize', 'parse']
