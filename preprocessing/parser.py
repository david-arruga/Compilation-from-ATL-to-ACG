# filepath: preprocessing/parser.py
"""Lexer (tokenize) y parser (parse) a AST."""
from __future__ import annotations

from .tokens import (
    LPAREN, RPAREN, LBRACE, RBRACE, LTRI, RTRI, COMMA, AND, OR, NOT,
    IMPLIES, IFF, NEXT, UNTIL, RELEASE, GLOBALLY, EVENTUALLY,
    PROPOSITION, AGENT_NAME, NAME, UNKNOWN, END_OF_INPUT,
    LBRACKET, RBRACKET, SYMBOL_MAP,
)
from .ast_nodes import (
    ParseNode, T, F, Var, And, Or, Not, Next, Until, Release,
    Globally, Eventually, Implies, Iff, Modality, DualModality,
)


def tokenize(source: str):
    tokens = []
    cursor = 0
    length = len(source)

    keywords = {
        "and": AND,
        "or": OR,
        "not": NOT,
        "next": NEXT,
        "until": UNTIL,
        "release": RELEASE,
        "globally": GLOBALLY,
        "eventually": EVENTUALLY,
        "implies": IMPLIES,
        "iff": IFF,
    }

    inside_agents = False

    while cursor < length:
        symbol = source[cursor]

        if symbol in " \t\n":
            pass
        elif symbol == "(":
            tokens.append((LPAREN, symbol))
        elif symbol == ")":
            tokens.append((RPAREN, symbol))
        elif symbol == "<":
            tokens.append((LTRI, symbol))
            inside_agents = True
        elif symbol == ">":
            tokens.append((RTRI, symbol))
            inside_agents = False
        elif symbol == "[":
            tokens.append((LBRACKET, symbol))
            inside_agents = True
        elif symbol == "]":
            tokens.append((RBRACKET, symbol))
            inside_agents = False
        elif symbol == ",":
            tokens.append((COMMA, symbol))
        elif symbol == "|":
            tokens.append((OR, symbol))
        elif symbol == "&":
            tokens.append((AND, symbol))
        elif symbol in SYMBOL_MAP and inside_agents is False:
            tokens.append((SYMBOL_MAP[symbol], symbol))
        elif symbol.isalpha():
            buffer = symbol
            cursor += 1
            while cursor < length and source[cursor].isalnum():
                buffer += source[cursor]
                cursor += 1

            if inside_agents:
                tokens.append((AGENT_NAME, buffer))
            else:
                token_type = keywords.get(buffer.lower(), PROPOSITION)
                tokens.append((token_type, buffer))

            cursor -= 1
        else:
            tokens.append((UNKNOWN, symbol))

        cursor += 1

    return tokens


def parse(tokens):
    cursor = 0
    limit = len(tokens)

    def advance():
        nonlocal cursor
        if cursor < limit:
            cursor += 1

    def current_token():
        return tokens[cursor] if cursor < limit else (END_OF_INPUT, "")

    def parse_atomic_formula():
        token = current_token()

        if token[0] == LPAREN:
            advance()
            result = parse_formula()
            if current_token()[0] != RPAREN:
                raise ValueError("Parse error: Expected ')' but not found.")
            advance()
            return result

        elif token[0] == PROPOSITION:
            advance()
            return Var(token[1])

        elif token[0] == NOT:
            advance()
            return Not(parse_atomic_formula())

        elif token[0] == LTRI:
            agents = []
            advance()
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()
            if current_token()[0] != RTRI:
                raise ValueError("Parse error: Expected '>' but not found.")
            advance()
            return Modality(agents, parse_temporal())

        elif token[0] == LBRACKET:
            agents = []
            advance()
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()
            if current_token()[0] != RBRACKET:
                raise ValueError("Parse error: Expected ']' but not found.")
            advance()
            return DualModality(agents, parse_temporal())

        else:
            raise ValueError(f"Parse error: unexpected token '{token[1]}'.")

    def parse_temporal():
        token = current_token()

        if token[0] == NEXT:
            advance()
            return Next(parse_temporal())

        elif token[0] == GLOBALLY:
            advance()
            return Globally(parse_temporal())

        elif token[0] == EVENTUALLY:
            advance()
            return Eventually(parse_temporal())

        elif token[0] == LTRI:
            agents = []
            advance()
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()
            if current_token()[0] != RTRI:
                raise ValueError("Parse error: Expected '>' but not found.")
            advance()
            return Modality(agents, parse_temporal())

        elif token[0] == LBRACKET:
            agents = []
            advance()
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()
            if current_token()[0] != RBRACKET:
                raise ValueError("Parse error: Expected ']' but not found.")
            advance()
            return DualModality(agents, parse_temporal())

        else:
            return parse_atomic_formula()

    def parse_until():
        result = parse_temporal()
        while current_token()[0] in {UNTIL, RELEASE}:
            token_type = current_token()[0]
            advance()
            rhs = parse_temporal()

            if token_type == UNTIL:
                if isinstance(result, Modality):
                    result = Modality(result.agents, Until(result.sub, rhs))
                elif isinstance(result, DualModality):
                    result = DualModality(result.agents, Until(result.sub, rhs))
                else:
                    result = Until(result, rhs)

            elif token_type == RELEASE:
                if isinstance(result, Modality):
                    result = Modality(result.agents, Release(result.sub, rhs))
                elif isinstance(result, DualModality):
                    result = DualModality(result.agents, Release(result.sub, rhs))
                else:
                    result = Release(result, rhs)

        return result

    def parse_conjunction():
        result = parse_until()
        while current_token()[0] == AND:
            advance()
            rhs = parse_until()
            result = And(result, rhs)
        return result

    def parse_disjunction():
        result = parse_conjunction()
        while current_token()[0] == OR:
            advance()
            rhs = parse_conjunction()
            result = Or(result, rhs)
        return result

    def parse_implies():
        result = parse_disjunction()
        while current_token()[0] in {IMPLIES, IFF}:
            token_type = current_token()[0]
            advance()
            rhs = parse_disjunction()
            if token_type == IMPLIES:
                result = Implies(result, rhs)
            elif token_type == IFF:
                result = Iff(result, rhs)
        return result

    def parse_formula():
        return parse_implies()

    ast_root = parse_formula()

    if current_token()[0] != END_OF_INPUT:
        raise ValueError(f"Parse error: Unexpected token '{current_token()[1]}' at end of input.")

    return ast_root


__all__ = ["tokenize", "parse"]
