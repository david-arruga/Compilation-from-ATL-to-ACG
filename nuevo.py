from __future__ import annotations
from itertools import chain, combinations, product
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import defaultdict
import pandas as pd
from matplotlib.patches import Patch
from collections import deque
import networkx as nx
from collections import defaultdict
import time
import traceback
import csv, time, traceback
from pathlib import Path
import csv, time, traceback
from pathlib import Path
import pandas as pd
import concurrent.futures, functools, time, traceback, csv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as TPTimeout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import cycle
from scipy.signal import savgol_filter
from multiprocessing import get_context, TimeoutError as MpTimeout
from matplotlib.ticker import ScalarFormatter

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import multiprocessing as mp
import time


from itertools import product
from pathlib import Path






LPAREN, RPAREN = 1, 2
LBRACE, RBRACE = 3, 4
LTRI, RTRI = 5, 6
COMMA = 7
AND, OR, NOT = 8, 9, 10
IMPLIES, IFF = 11, 12
NEXT, UNTIL, RELEASE = 13, 14, 15
GLOBALLY, EVENTUALLY = 16, 17
PROPOSITION, AGENT_NAME = 18, 19
NAME, UNKNOWN, END_OF_INPUT = 20, 21, 22
LBRACKET, RBRACKET = 23,24
SYMBOL_MAP = {
    "◯": NEXT,
    "□": GLOBALLY,
    "◇": EVENTUALLY,
    "U": UNTIL,
    "R": RELEASE,
    "G": GLOBALLY,
    "F": EVENTUALLY,
    "X": NEXT
}

class ParseNode:
    def __str__(self):
        return self.to_formula()

    def to_formula(self):
        return ""

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}{self.__class__.__name__}\n"
    
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __hash__(self):
        def make_hashable(value):
            if isinstance(value, list):
                return tuple(value)
            elif isinstance(value, dict):
                return tuple(sorted(value.items()))
            elif isinstance(value, set):
                return frozenset(value)
            elif isinstance(value, ParseNode):
                return hash(value)
            return value

        items = tuple(sorted((k, make_hashable(v)) for k, v in self.__dict__.items()))
        return hash((self.__class__.__name__, items))
    
class T(ParseNode):    
    def __eq__(self, other):
        return isinstance(other, T)

    def __hash__(self):
        return hash("T")

    def to_formula(self):
        return "⊤"

    def __str__(self):
        return self.to_formula()
    
    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}T"

class F(ParseNode):
    def __eq__(self, other):
        return isinstance(other, F)

    def __hash__(self):
        return hash("F")

    def to_formula(self):
        return "⊥"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}F"

class Var(ParseNode):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(("Var", self.name))

    def to_formula(self):
        return self.name

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Var('{self.name}')"

class And(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        lhs_str = f"({self.lhs})" if isinstance(self.lhs, (And,Or)) else str(self.lhs)
        rhs_str = f"({self.rhs})" if isinstance(self.rhs, (And,Or)) else str(self.rhs)
        return f"{lhs_str} ∧ {rhs_str}"
    
    def __str__(self):
        return self.to_formula()
    
    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}And\n{self.lhs.to_tree(level+1)}\n{self.rhs.to_tree(level+1)}"

class Or(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        lhs_str = f"({self.lhs})" if isinstance(self.lhs, (And,Or)) else str(self.lhs)
        rhs_str = f"({self.rhs})" if isinstance(self.rhs, (And,Or)) else str(self.rhs)
        return f"{lhs_str} ∨ {rhs_str}"

    def __str__(self):
        return self.to_formula()
    
    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Or\n{self.lhs.to_tree(level+1)}\n{self.rhs.to_tree(level+1)}"

class Not(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"(¬ {self.sub})"

    def __str__(self):
        return self.to_formula()
    
    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Not\n{self.sub.to_tree(level+1)}"

class Next(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"◯ {self.sub}" if isinstance(self.sub, Var) else f"◯ ({self.sub})"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Next\n{self.sub.to_tree(level+1)}"

class Until(ParseNode):
    def __init__(self, lhs, rhs,generated_from_eventually=False):
        self.lhs = lhs
        self.rhs = rhs
        self.generated_from_eventually = generated_from_eventually

    def to_formula(self):
        return f"({self.lhs} U {self.rhs})"

    def __str__(self):
        return self.to_formula()
    
    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Until\n{self.lhs.to_tree(level+1)}\n{self.rhs.to_tree(level+1)}"

class Release(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} R {self.rhs})"

    def __str__(self):
        return self.to_formula()
    
    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Release\n{self.lhs.to_tree(level+1)}\n{self.rhs.to_tree(level+1)}"

class Globally(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"□ {self.sub}" if isinstance(self.sub, Var) else f"□ ({self.sub})"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Globally\n{self.sub.to_tree(level+1)}"

class Eventually(ParseNode):
    def __init__(self, sub):
        self.sub = sub

    def to_formula(self):
        return f"◇ {self.sub}" if isinstance(self.sub, Var) else f"◇ ({self.sub})"

    def __str__(self):
        return self.to_formula()
    
    def to_tree(self, level=0):
        indent = "    " * level
        return f"{indent}Eventually\n{self.sub.to_tree(level+1)}"

class Implies(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} -> {self.rhs})"

    def __str__(self):
        return self.to_formula()

class Iff(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} <-> {self.rhs})"

    def __str__(self):
        return self.to_formula()

class Modality(ParseNode):
    def __init__(self, agents, sub):
        self.agents = agents
        self.sub = sub

    def to_formula(self):
        agents_str = ", ".join(self.agents)
        return f"<{agents_str}> {self.sub}"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
            indent = "    " * level
            agents_str = ", ".join(self.agents)
            return f"{indent}Modality ({agents_str})\n{self.sub.to_tree(level+1)}"

class DualModality(ParseNode):
    def __init__(self, agents, sub):
        self.agents = agents
        self.sub = sub

    def to_formula(self):
        agents_str = ", ".join(self.agents)
        return f"[{agents_str}] {self.sub}"

    def __str__(self):
        return self.to_formula()

    def to_tree(self, level=0):
        indent = "    " * level
        agents_str = ", ".join(self.agents)
        return f"{indent}DualModality ({agents_str})\n{self.sub.to_tree(level+1)}"

    def __eq__(self, other):
        return isinstance(other, DualModality) and self.agents == other.agents and self.sub == other.sub

    def __hash__(self):
        return hash(("DualModality", frozenset(self.agents), self.sub))

class Top:
    def __eq__(self, other):
        return isinstance(other, Top)

    def __hash__(self):
        return hash("Top")

    def __str__(self):
        return "⊤"
    
    def to_formula(self):
        return "⊤"

class Bottom:
    def __eq__(self, other):
        return isinstance(other, Bottom)

    def __hash__(self):
        return hash("Bottom")

    def __str__(self):
        return "⊥"
    
    def to_formula(self):
        return "⊥"
    
class Conj(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} ∧ {self.rhs})"

    def __str__(self):
        return self.to_formula()
    
    def __eq__(self, other):
        return isinstance(other, Conj) and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        return hash(('Conj', self.lhs, self.rhs))

class Disj(ParseNode):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def to_formula(self):
        return f"({self.lhs} ∨ {self.rhs})"

    def __str__(self):
        return self.to_formula()
    
    def __eq__(self, other):
        return isinstance(other, Disj) and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        return hash(('Disj', self.lhs, self.rhs))

class UniversalAtom:
    def __init__(self, state, agents):
        self.state = state
        self.agents = frozenset(agents)

    def __eq__(self, other):
        return (
            isinstance(other, UniversalAtom) and
            self.state == other.state and
            self.agents == other.agents
        )

    def __hash__(self):
        return hash(("UniversalAtom", self.state, self.agents))

    def __str__(self):
        agents_str = ", ".join(sorted(self.agents)) if self.agents else "∅"
        return f"( {self.state}, □, {{{agents_str}}} )"
    
    def to_formula(self):
        agents_str = ", ".join(sorted(self.agents)) if self.agents else "∅"
        return f"( {self.state}, □, {{{agents_str}}} )"

class ExistentialAtom:
    def __init__(self, state, agents):
        self.state = state
        self.agents = frozenset(agents)

    def __eq__(self, other):
        return (
            isinstance(other, ExistentialAtom) and
            self.state == other.state and
            self.agents == other.agents
        )

    def __hash__(self):
        return hash(("ExistentialAtom", self.state, self.agents))

    def __str__(self):
        agents_str = ", ".join(sorted(self.agents)) if self.agents else "∅"
        return f"( {self.state}, ◇, {{{agents_str}}} )"

    def to_formula(self):
        agents_str = ", ".join(sorted(self.agents)) if self.agents else "∅"
        return f"( {self.state}, ◇, {{{agents_str}}} )"

class EpsilonAtom:
    def __init__(self, state):
        self.state = state

    def __eq__(self, other):
        return isinstance(other, EpsilonAtom) and self.state == other.state

    def __hash__(self):
        return hash(("EpsilonAtom", self.state))

    def __str__(self):
        return f"( {self.state}, ε )"
    
    def to_formula(self):
        return f"( {self.state}, ε )"
    
class ACG:

    WILDCARD = None            

    def __init__(self):
        self.propositions  = set()    
        self.states        = set()    
        self.initial_state = None
        self.final_states  = set()
        self.alphabet      = set()    
        self.transitions   = {}       


    def generate_alphabet(self):
        self.alphabet = {
            frozenset(s)
            for r in range(len(self.propositions) + 1)
            for s in combinations(self.propositions, r)
        }

    def add_transition(self, state_from, input_symbol, transition_formula):
        if state_from not in self.states:
            self.states.add(state_from)
        if input_symbol is not self.WILDCARD and input_symbol not in self.alphabet:
            raise ValueError(
                f"Input symbol {input_symbol} is not in the alphabet 2^AP."
            )

        self.transitions[(state_from, input_symbol)] = transition_formula

    def get_transition(self, state, input_symbol):
        try:
            return self.transitions[(state, input_symbol)]
        except KeyError:
            try:
                return self.transitions[(state, self.WILDCARD)]
            except KeyError:
                raise KeyError(
                    f"δ undefined for state={state}, σ={input_symbol}"
                )

    def __str__(self):
        def is_atomic(state):
            return isinstance(state, Var) or (
                isinstance(state, Not) and isinstance(state.sub, Var)
            )

        alpha = sorted(
            ["{" + ", ".join(sorted(a)) + "}" if a else "{}"
             for a in self.alphabet]
        )

        lines = [
            "ACG(",
            f"  Alphabet: {alpha},",
            f"  States: {[str(s) for s in sorted(self.states, key=str)]},",
            f"  Initial State: {self.initial_state},",
            f"  Final States: {[str(s) for s in sorted(self.final_states, key=str)]},",
            "  Transitions:",
        ]

        shown = set()
        for (q, σ), rhs in self.transitions.items():
            if is_atomic(q):
                σ_str = "{" + ", ".join(sorted(σ)) + "}" if σ else "{}"
            else:
                if q in shown:
                    continue
                shown.add(q)
                σ_str = "*"

            lines.append(f"    δ({q}, {σ_str}) → {rhs}")

        lines.append(")")
        return "\n".join(lines)


#=========================================
# PREPROCESSING
#=========================================

def tokenize(source): 
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
        "iff": IFF
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
        elif symbol in SYMBOL_MAP and inside_agents == False:
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
        return And(
            Or(Not(lhs), rhs),  
            Or(Not(rhs), lhs) )

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
                push_negations_to_nnf(Not(sub.rhs))
            )

        if isinstance(sub, Or):
            return And(
                push_negations_to_nnf(Not(sub.lhs)),
                push_negations_to_nnf(Not(sub.rhs))
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

    #elif isinstance(node, DualModality):
        #return DualModality(node.agents, push_negations_to_nnf(node.sub))

    return node  

def normalize_formula(ast: ParseNode) -> ParseNode:
    previous = None
    current = ast

    while previous != current:
        previous = current
        current = eliminate_f_and_r(current)
        current = push_negations_to_nnf(current)

    return current

def filter(ast: ParseNode, strict_ATL: bool = True) -> str:
    def validate_structure(node: ParseNode) -> str | None:
        if isinstance(node, (Modality, DualModality)):
            if not isinstance(node.agents, list) or not all(isinstance(a, str) for a in node.agents):
                print("ERROR: Modality must have a list of agent names.")
                return "INVALID"
            if not isinstance(node.sub, ParseNode):
                print("ERROR: Modality must have a valid subformula.")
                return "INVALID"

        if isinstance(node, Until):
            if not isinstance(node.lhs, ParseNode) or not isinstance(node.rhs, ParseNode):
                print("ERROR: Until must have both lhs and rhs as valid subformulas.")
                return "INVALID"

        for value in getattr(node, "__dict__", {}).values():
            if isinstance(value, ParseNode):
                result = validate_structure(value)
                if result:
                    return result
        return None

    def validate_atl_semantics(node: ParseNode, parent: ParseNode | None = None, grandparent: ParseNode | None = None) -> str | None:
        if isinstance(node, (Modality, DualModality)):
            sub = node.sub
            if isinstance(sub, Not):
                sub = sub.sub  # allow Not(◯φ), Not(□φ), etc.
            if not isinstance(sub, (Next, Globally, Until)):
                print("ERROR: Modality must be applied to Next, Globally, or Until (or their negation).")
                return "ATL* but not ATL"

        if isinstance(node, Until):
            if getattr(node, "generated_from_eventually", False):
                pass
            elif isinstance(parent, (Modality, DualModality)):
                pass
            elif isinstance(parent, Not) and isinstance(grandparent, (Modality, DualModality)):
                pass
            else:
                print("ERROR: Until must be directly under a modality (or its negation).")
                return "ATL* but not ATL"

        if isinstance(node, (Next, Globally)):
            if not isinstance(parent, (Modality, DualModality)):
                print(f"ERROR: {node.__class__.__name__} must be directly under a modality.")
                return "ATL* but not ATL"

        for value in getattr(node, "__dict__", {}).values():
            if isinstance(value, ParseNode):
                result = validate_atl_semantics(value, node, parent)
                if result:
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ParseNode):
                        result = validate_atl_semantics(item, node, parent)
                        if result:
                            return result
        return None

    # Fase 1: validación de estructura general (tipos, campos)
    structure_result = validate_structure(ast)
    if structure_result:
        return structure_result

    if not strict_ATL:
        return "ATL*"

    # Fase 2: validación semántica ATL estricta
    semantic_result = validate_atl_semantics(ast)
    if semantic_result:
        return semantic_result

    return "ATL"


#========================================
# ACG
#========================================

def _is_atomic_state(state):
    return isinstance(state, Var) or (isinstance(state, Not) and isinstance(state.sub, Var))

def _delta_atomic(state, sigma):
    """
    sigma es el rotulado actual, como frozenset de proposiciones verdaderas.
    Devuelve Top()/Bottom() según p∈sigma.
    """
    if isinstance(state, Var):
        return Top() if state.name in sigma else Bottom()
    if isinstance(state, Not) and isinstance(state.sub, Var):
        return Bottom() if state.sub.name in sigma else Top()
    raise TypeError("delta_atomic solo para estados atómicos p o ¬p")

def extract_propositions(node):
    propositions = set()

    def traverse(n):
        if isinstance(n, Var):
            propositions.add(n.name) 
        
        for child in getattr(n, "__dict__", {}).values():
            if isinstance(child, ParseNode):
                traverse(child)

    traverse(node)
    return propositions 

def generate_closure(ast):
    closure = set()

    def negate_and_push(node):
        neg = Not(deepcopy(node))
        return push_negations_to_nnf(neg)

    def add_with_negations(node):
        closure.add(node)
        closure.add(negate_and_push(node))

    def traverse(node, parent=None):
        if isinstance(node, Not) and isinstance(node.sub, (Modality, DualModality)):
            inner = node.sub.sub
            if isinstance(inner, (Next, Globally)):
                add_with_negations(node)
                traverse(inner.sub, inner)
            elif isinstance(inner, Until):
                add_with_negations(node)
                traverse(inner.lhs, inner)
                traverse(inner.rhs, inner)
            return

        elif isinstance(node, (Modality, DualModality)):
            add_with_negations(node)
            traverse(node.sub, node)
            return

        elif isinstance(node, (And, Or)):
            add_with_negations(node)
            traverse(node.lhs, node)
            traverse(node.rhs, node)
            return

        elif isinstance(node, Not):
            if isinstance(node.sub, Var):
                closure.add(node)
                closure.add(node.sub)
            else:
                traverse(node.sub, node)
            return

        elif isinstance(node, Var):
            if not isinstance(parent, Not):
                closure.add(node)
                closure.add(Not(deepcopy(node)))  
            return

        elif isinstance(node, Until):
            if isinstance(parent, (Modality, DualModality)):
                traverse(node.lhs, node)
                traverse(node.rhs, node)
            return

        elif isinstance(node, (Next, Globally)):
            if isinstance(parent, (Modality, DualModality)):
                traverse(node.sub, node)
            return
        
        elif isinstance(node, (T, F)):  
            add_with_negations(node)
            return


    traverse(ast)
    return closure

def generate_transitions_final(acg, cgs):
    WILDCARD = acg.WILDCARD  # = None

    for state in acg.states:

        # ATÓMICAS: no almacenamos transiciones; δ se resuelve perezosamente
        if _is_atomic_state(state):
            continue

        sigma = WILDCARD

        if isinstance(state, T):
            acg.add_transition(state, sigma, Top())

        elif isinstance(state, F):
            acg.add_transition(state, sigma, Bottom())

        elif isinstance(state, And):
            acg.add_transition(
                state, sigma,
                Conj(EpsilonAtom(state.lhs), EpsilonAtom(state.rhs))
            )

        elif isinstance(state, Or):
            acg.add_transition(
                state, sigma,
                Disj(EpsilonAtom(state.lhs), EpsilonAtom(state.rhs))
            )

        # ⟨A⟩ X φ
        elif isinstance(state, Modality) and isinstance(state.sub, Next):
            next_state = state.sub.sub
            agents = frozenset(state.agents)
            acg.add_transition(state, sigma, UniversalAtom(next_state, agents))

        # ¬⟨A⟩ X φ
        elif isinstance(state, Not) and isinstance(state.sub, Modality) \
             and isinstance(state.sub.sub, Next):
            inner = state.sub.sub.sub
            neg_inner = push_negations_to_nnf(Not(inner))
            agents = frozenset(state.sub.agents)
            Omega = cgs.agents
            Agentsbuenos = Omega - agents
            acg.add_transition(state, sigma, ExistentialAtom(neg_inner, Agentsbuenos))

        # ⟨A⟩ G φ
        elif isinstance(state, Modality) and isinstance(state.sub, Globally):
            phi = state.sub.sub
            agents = frozenset(state.agents)
            acg.add_transition(state, sigma, Conj(EpsilonAtom(phi), UniversalAtom(state, agents)))

        # ¬⟨A⟩ G φ
        elif isinstance(state, Not) and isinstance(state.sub, Modality) \
             and isinstance(state.sub.sub, Globally):
            phi = state.sub.sub.sub
            neg_phi = push_negations_to_nnf(Not(phi))
            agents = frozenset(state.sub.agents)
            Omega = cgs.agents
            Agentsbuenos = Omega - agents
            acg.add_transition(state, sigma, Disj(EpsilonAtom(neg_phi), ExistentialAtom(state, Agentsbuenos)))

        # ⟨A⟩ (φ1 U φ2)
        elif isinstance(state, Modality) and isinstance(state.sub, Until):
            phi1, phi2 = state.sub.lhs, state.sub.rhs
            agents = frozenset(state.agents)
            acg.add_transition(
                state, sigma,
                Disj(EpsilonAtom(phi2), Conj(EpsilonAtom(phi1), UniversalAtom(state, agents)))
            )

        # ¬⟨A⟩ (φ1 U φ2)
        elif isinstance(state, Not) and isinstance(state.sub, Modality) \
             and isinstance(state.sub.sub, Until):
            phi1, phi2 = state.sub.sub.lhs, state.sub.sub.rhs
            neg_phi1 = push_negations_to_nnf(Not(phi1))
            neg_phi2 = push_negations_to_nnf(Not(phi2))
            agents = frozenset(state.sub.agents)
            Omega = cgs.agents
            Agentsbuenos = Omega - agents
            acg.add_transition(
                state, sigma,
                Conj(EpsilonAtom(neg_phi2), Disj(EpsilonAtom(neg_phi1), ExistentialAtom(state, Agentsbuenos)))
            )

def build_acg_final(transformed_ast, cgs, materialize_alphabet: bool = False):
    acg = ACG()

    # 1) AP
    ap_set = extract_propositions(transformed_ast)
    acg.propositions = ap_set

    # 2) Alfabeto (opcional). Si no lo necesitas para imprimir, no lo generes.
    if materialize_alphabet:
        acg.generate_alphabet()
    else:
        acg.alphabet = set()

    # 3) Estados / inicial / finales
    closure = generate_closure(transformed_ast)
    acg.states = closure
    acg.initial_state = transformed_ast

    for node in closure:
        if isinstance(node, Modality) and isinstance(node.sub, Globally):
            acg.final_states.add(node)
        elif isinstance(node, Not) and isinstance(node.sub, Modality) and isinstance(node.sub.sub, Until):
            acg.final_states.add(node)

    # 4) Transiciones compactas (sin σ por atómicas)
    generate_transitions_final(acg, cgs)

    # 5) Parchear get_transition SOLO en este objeto
    _orig_get = acg.get_transition  # referencia al método original

    def _get_transition_monkey(self, state, sigma):
        if _is_atomic_state(state):
            return _delta_atomic(state, sigma)
        # para el resto, usa wildcard/entrada explícita
        return _orig_get(state, sigma)

    # ligar el método al objeto 'acg'
    acg.get_transition = _get_transition_monkey.__get__(acg, ACG)

    return acg

def build_acg_with_timer_final(ast,cgs):
    start = time.perf_counter()
    acg = build_acg_final(ast,cgs)
    elapsed = time.perf_counter() - start
    size = compute_acg_size(acg)
    return acg, size, elapsed

def atom_counter(formula):
    atoms = set()

    def recurse(node):
        if isinstance(node, EpsilonAtom):
            atoms.add(("ε", node.state))
        elif isinstance(node, ExistentialAtom):
            atoms.add(("◇", node.state, frozenset(node.agents)))
        elif isinstance(node, UniversalAtom):
            atoms.add(("□", node.state, frozenset(node.agents)))
        elif isinstance(node, Conj) or isinstance(node, Disj):
            recurse(node.lhs)
            recurse(node.rhs)
        elif isinstance(node, Not):
            recurse(node.sub)
        elif isinstance(node, (Modality, DualModality, Next, Globally)):
            recurse(node.sub)
        elif isinstance(node, Until):
            recurse(node.lhs)
            recurse(node.rhs)

    recurse(formula)
    return atoms

def compute_acg_size(acg):
    state_count = len(acg.states)

    atom_set = set()
    for (state, sigma), formula in acg.transitions.items():
        atoms = atom_counter(formula)
        atom_set.update(atoms)

    return state_count + len(atom_set)



#========================================
# CGS
#========================================

class CGS:
    def __init__(self):
        self.propositions = set()
        self.agents = set()
        self.states = set()
        self.initial_state = None
        self.labeling_function = {}
        self.decisions = {}
        self.transition_function = {}
        self.strategies = {}

    def add_proposition(self, proposition):
        self.propositions.add(proposition)

    def add_state(self, state):
        self.states.add(state)

    def set_initial_state(self, state):
        if state not in self.states:
            self.add_state(state)
        self.initial_state = state

    def label_state(self, state, propositions):
        if state not in self.states:
            self.add_state(state)
        self.labeling_function[state] = set(propositions)

    def add_agent(self, agent):
        self.agents.add(agent)

    def add_decisions(self, agent, decision_set):
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the CGS.")
        self.decisions[agent] = set(decision_set)

    def add_transition(self, state, joint_decision, next_state):
        if state not in self.states:
            self.add_state(state)
        if next_state not in self.states:
            self.add_state(next_state)

        ordered_joint_action = frozenset(sorted(joint_decision, key=lambda x: x[0]))
        self.transition_function[(state, ordered_joint_action)] = next_state

    
    def get_all_agent_choices(self, agent_subset):

        agent_subset = sorted(agent_subset)
        all_choices = [self.decisions[agent] for agent in agent_subset]
        combinations = product(*all_choices)

        return [
            dict(zip(agent_subset, combo)) for combo in combinations
        ]

    def get_joint_actions_for_agents(self, agent_subset):

        agent_subset = sorted(agent_subset)
        all_choices = [self.decisions[agent] for agent in agent_subset]
        combos = product(*all_choices)

        return [dict(zip(agent_subset, combo)) for combo in combos]

    def get_successor(self, state, joint_decision_dict):

        joint_action = frozenset(sorted(joint_decision_dict.items()))
        return self.transition_function.get((state, joint_action), None)
    
    def get_propositions(self):
        return sorted(self.propositions)

    def get_agents(self):
        return sorted(self.agents)

    def __str__(self):
        formatted_transitions = []
        for (state, joint_decision) in self.transition_function:
            decision_str = ", ".join([f"({agent}, {decision})" for agent, decision in joint_decision])
            next_state = self.transition_function[(state, joint_decision)]
            formatted_transitions.append(f"    τ({state}, {{{decision_str}}}) → {next_state}")

        return (
            f"CGS(\n"
            f"  Propositions: {self.propositions}\n"
            f"  Agents: {sorted(self.agents)}\n"
            f"  States: {sorted(self.states)}\n"
            f"  Initial State: {self.initial_state}\n"
            f"  Labeling Function: {self.labeling_function}\n"
            f"  Decisions: {self.decisions}\n"
            f"  Transitions:\n" +
            "\n".join(formatted_transitions) +
            f"\n)"
        )
    
    def validate(self, *, check_reachability=False, verbose=False):
        errors = []

        if self.initial_state is None or self.initial_state not in self.states:
            errors.append("Initial state missing or not in self.states.")

        missing_dec_sets = [a for a in self.agents
                            if a not in self.decisions or not self.decisions[a]]
        if missing_dec_sets:
            errors.append(f"Agents without decision sets: {missing_dec_sets}")

        joint_actions = list(product(*[[(a,d) for d in sorted(self.decisions[a])]
                                    for a in sorted(self.agents)]))
        for s in self.states:
            for ja in joint_actions:
                ja_key = frozenset(ja)
                if (s, ja_key) not in self.transition_function:
                    errors.append(f"Missing transition from {s} with {dict(ja)}.")
        
        for (src, _), dst in self.transition_function.items():
            if dst not in self.states:
                errors.append(f"Transition points to undefined state {dst}.")

        for st in self.states:
            if st not in self.labeling_function:
                errors.append(f"State {st} has no label.")
            else:
                unknown_props = self.labeling_function[st] - self.propositions
                if unknown_props:
                    errors.append(f"Unknown propositions in label of {st}: {unknown_props}")

        if check_reachability and not errors:
            seen = {self.initial_state}
            frontier = [self.initial_state]
            while frontier:
                cur = frontier.pop()
                for ja in joint_actions:
                    nxt = self.transition_function[(cur, frozenset(ja))]
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)
            unreachable = self.states - seen
            if unreachable:
                errors.append(f"Unreachable states: {unreachable}")

        if errors:
            msg = "CGS validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(msg)
        if verbose:
            print("CGS validation successful: all checks passed.")

class Strategy:
    def __init__(self, agents, decision_map=None):
        self.agents = set(agents)
        self.decision_map = decision_map if decision_map else {}

    def add_decision(self, history, decisions):
        if not isinstance(decisions, dict):
            raise ValueError("Decisions must be a dictionary {agent: decision}.")
        self.decision_map[tuple(history)] = decisions

    def get_decision(self, history):
        return self.decision_map.get(tuple(history), {})

    def __str__(self):
        formatted_decisions = [
            f"  History: ({' → '.join(history)}) → Decisions: {decisions}"
            for history, decisions in self.decision_map.items()
        ]
        return f"Strategy({self.agents}):\n" + "\n".join(formatted_decisions)

class CounterStrategy:
    def __init__(self, agents, decision_map=None):
        self.agents = set(agents)
        self.decision_map = decision_map if decision_map else {}

    def add_decision(self, history, decision_A_prime, decision_A_complement):
        key = (tuple(history), frozenset(decision_A_prime.items()))
        self.decision_map[key] = decision_A_complement

    def get_decision(self, history, decision_A_prime):
        key = (tuple(history), frozenset(decision_A_prime.items()))
        return self.decision_map.get(key, {})

    def __str__(self):
        formatted_decisions = [
            f"  History: ({' → '.join(history)}), Decision_A': {decision_A_prime} → Decision_A\A': {decisions_A_complement}"
            for (history, decision_A_prime), decisions_A_complement in self.decision_map.items()
        ]
        return f"CounterStrategy({self.agents}):\n" + "\n".join(formatted_decisions)

def play(game, initial_state, strategy, counterstrategy, max_steps=10):
    history = [initial_state]
    current_state = initial_state

    for _ in range(max_steps):
        print(f"\n🟢 Current State: {current_state}")

        decision_A = strategy.get_decision([current_state])
        if not decision_A:
            print(f" Strategy failed to return a decision at {current_state}. Stopping.")
            break  

        decision_A_complement = counterstrategy.get_decision([current_state], decision_A)
        if not decision_A_complement:
            print(f" CounterStrategy failed to return a decision at {current_state}. Stopping.")
            break  

        joint_action = frozenset(list(decision_A.items()) + list(decision_A_complement.items()))

        if (current_state, joint_action) in game.transition_function:
            current_state = game.transition_function[(current_state, joint_action)]
            history.append(current_state)
            print(f" Transition found! Moving to: {current_state}")
        else:
            print(" No matching transition found. Stopping.")
            break  

    return history


#==========================================
# GAME
#==========================================


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


class GameProduct:
    def __init__(self, acg: ACG, cgs: CGS):
        self.acg = acg
        self.cgs = cgs
        self.states = set()
        self.transitions = dict()
        self.initial_states = set()
        self.S1 = set()
        self.S2 = set()
        self.B = set()  

    def __str__(self):
        pretty_initial = [pretty_node(s) for s in sorted(self.initial_states, key=str)]
        pretty_s1 = [pretty_node(s) for s in sorted(self.S1, key=str)]
        pretty_s2 = [pretty_node(s) for s in sorted(self.S2, key=str)]
        pretty_B = [pretty_node(s) for s in sorted(self.B, key=str)]

        lines = [
            "GameProduct(",
            f"  Initial States: {pretty_initial}",
            f"  Total States: {len(self.states)}",
            f"  Player Accept States (S1): {pretty_s1}",
            f"  Player Reject States (S2): {pretty_s2}",
            f"  Büchi Final States (B): {pretty_B}",
            f"  Transitions (|E| = {len(self.transitions)}):"
        ]

        for (src, dst), val in sorted(self.transitions.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            lines.append(f"    ({pretty_node(src)}) → ({pretty_node(dst)})")

        lines.append(")")
        return "\n".join(lines)

def generate_initial_game_states(product: GameProduct):
    q0 = product.acg.initial_state
    s0 = product.cgs.initial_state
    initial = ("state", q0, s0)

    product.initial_states.add(initial)
    product.states.add(initial)
    product.S1.add(initial)
    if q0 in product.acg.final_states:
        product.B.add(initial)

    return initial

def expand_from_state_node(product: GameProduct, q, s):
    relevant_props = product.acg.propositions  
    projected_label = frozenset(p for p in product.cgs.labeling_function[s] if p in relevant_props)
    delta_formula = product.acg.get_transition(q, projected_label)

    if "true_sink" not in product.states:
        product.states.add("true_sink")
        product.transitions[("true_sink", "true_sink")] = None
        product.S1.add("true_sink")
        product.B.add("true_sink")

    if "false_sink" not in product.states:
        product.states.add("false_sink")
        product.transitions[("false_sink", "false_sink")] = None
        product.S2.add("false_sink")

    if isinstance(delta_formula, Top):
        product.states.add("true_sink")
        product.transitions[(("state", q, s), "true_sink")] = None
        return

    if isinstance(delta_formula, Bottom):
        product.states.add("false_sink")
        product.transitions[(("state", q, s), "false_sink")] = None
        return

    for U in generate_possibilities(delta_formula):
        atom_selection_node = ("atom_selection", q, s, frozenset(U))
        product.states.add(atom_selection_node)
        product.transitions[(("state", q, s), atom_selection_node)] = None
        product.S2.add(atom_selection_node)

def expand_from_atom_selection_node(product: GameProduct, q, s, U):
    for alpha in U:
        if isinstance(alpha, EpsilonAtom):
            q_prime = alpha.state
            new_state = ("state", q_prime, s)
            product.states.add(new_state)
            product.transitions[(("atom_selection", q, s, U), new_state)] = None
            product.S1.add(new_state)

            if q_prime in product.acg.final_states:
                product.B.add(new_state)


        elif isinstance(alpha, UniversalAtom):
            q_prime = alpha.state
            new_node = ("atom_applied", q_prime, s, alpha)
            product.states.add(new_node)
            product.transitions[(("atom_selection", q, s, U), new_node)] = None
            product.S1.add(new_node)  

        elif isinstance(alpha,ExistentialAtom):
            q_prime = alpha.state
            new_node = ("atom_applied", q_prime, s, alpha)
            product.states.add(new_node)
            product.transitions[(("atom_selection", q, s, U), new_node)] = None
            product.S2.add(new_node)  

def expand_from_atom_applied_node(product: GameProduct, q, s, alpha):

    if isinstance(alpha, UniversalAtom):
        for d in product.cgs.get_all_agent_choices(alpha.agents):
            reject_univ_node = (
                "reject_univ",
                q,
                s,
                alpha.agents,
                alpha,
                frozenset(d.items())
            )
            product.states.add(reject_univ_node)
            product.transitions[(("atom_applied", q, s, alpha), reject_univ_node)] = None
            product.S2.add(reject_univ_node)

    elif isinstance(alpha, ExistentialAtom):
        A_prime = alpha.agents
        A = product.cgs.agents
        A_minus_A_prime = A - A_prime

        for v_reject in product.cgs.get_all_agent_choices(A_minus_A_prime):
            accept_exist_node = (
                "accept_exist",
                q,
                s,
                A_prime,
                alpha,
                frozenset(v_reject.items())
            )
            product.states.add(accept_exist_node)
            product.transitions[(("atom_applied", q, s, alpha), accept_exist_node)] = None
            product.S1.add(accept_exist_node)

def expand_from_reject_univ_node(product: GameProduct, q, s, A, alpha, dA_frozen):
    dA = dict(dA_frozen)
    remaining_agents = product.cgs.agents - A

    for d_reject in product.cgs.get_all_agent_choices(remaining_agents):
        full_decision = {**dA, **d_reject}
        successor = product.cgs.get_successor(s, full_decision)

        if successor is not None:
            q_prime = alpha.state
            new_node = ("state", q_prime, successor)
            product.states.add(new_node)
            product.transitions[(("reject_univ", q, s, A, alpha, dA_frozen), new_node)] = None
            product.S1.add(new_node)
            if q_prime in product.acg.final_states:
                product.B.add(new_node)

def expand_from_accept_exist_node(product: GameProduct, q, s, agents_prime, alpha, v_reject):

    q_prime = alpha.state
    A_prime = agents_prime
    A = product.cgs.agents
    A_minus_A_prime = A - A_prime

    v_reject_dict = dict(v_reject)
    if set(v_reject_dict.keys()) != A_minus_A_prime:
        raise ValueError(f"Reject's decision should cover exactly A \\ A', got {v_reject_dict.keys()}")

    for v_accept in product.cgs.get_joint_actions_for_agents(A_prime):
        full_joint_action = {**v_reject_dict, **v_accept}
        s_prime = product.cgs.get_successor(s, full_joint_action)

        if s_prime is not None:
            new_state = ("state", q_prime, s_prime)
            product.states.add(new_state)
            product.transitions[(("accept_exist", q, s, A_prime, alpha, v_reject), new_state)] = None
            product.S1.add(new_state)
            if q_prime in product.acg.final_states:
                product.B.add(new_state)


def pretty_node(node):

    if isinstance(node, str):
        # Manejo explícito de los estados especiales
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

def expand_node(product, node):

    if isinstance(node, str):
        return []

    kind = node[0]
    q, s = None, None

    if kind == "state":
        _, q, s = node
        expand_from_state_node(product, q, s)
        return [dst for (src, dst) in product.transitions if src == node]

    elif kind == "atom_selection":
        _, q, s, U = node
        expand_from_atom_selection_node(product, q, s, U)
        return [dst for (src, dst) in product.transitions if src == node]

    elif kind == "atom_applied":
        _, q, s, alpha = node
        expand_from_atom_applied_node(product, q, s, alpha)
        return [dst for (src, dst) in product.transitions if src == node]

    elif kind == "reject_univ":
        _, q, s, A, alpha, dA = node
        expand_from_reject_univ_node(product, q, s, A, alpha, dA)
        return [dst for (src, dst) in product.transitions if src == node]

    elif kind == "accept_exist":
        _, q, s, A_prime, alpha, v_reject = node
        expand_from_accept_exist_node(product, q, s, A_prime, alpha, v_reject)
        return [dst for (src, dst) in product.transitions if src == node]

    return []

def build_game(acg, cgs):
    product = GameProduct(acg, cgs)

    initial = generate_initial_game_states(product)
    worklist = [initial]
    visited = set()

    while worklist:
        node = worklist.pop()
        if node in visited:
            continue
        visited.add(node)

        new_nodes = expand_node(product, node)
        worklist.extend(new_nodes)


    return product.states, product.transitions, product.S1, product.S2, product.B, initial


#==========================================
# BÜCHI SOLVER
#==========================================


def predecessor_1(E, S1, S2, X):
    predecessor = set()

    for (src, dst) in E.keys():
        if dst in X:
            if src in S1:
                predecessor.add(src)
            elif src in S2:
                successors = {d for (s, d) in E if s == src}
                if successors <= X:
                    predecessor.add(src)
                    
    print(f"\PREDECESsOR GRANDE : {predecessor.__str__}")
    return predecessor

def predecessor_2(E, S1, S2, X):
    predecessor = set()

    for (src, dst) in E.keys():
        if dst in X:
            if src in S2:
                predecessor.add(src)
            elif src in S1:
                successors = {d for (s, d) in E if s == src}
                if successors <= X:
                    predecessor.add(src)
    return predecessor

def attractor_1(E, S1, S2, target):
    attractor = set(target)
    changed = True

    while changed:
        changed = False
        pred = predecessor_1(E, S1, S2, attractor)
        new = pred - attractor
        if new:
            attractor.update(new)
            changed = True
    return attractor

def attractor_2(E, S1, S2, target):
    attractor = set(target)
    changed = True

    while changed:
        changed = False
        pred = predecessor_2(E, S1, S2, attractor)
        new = pred - attractor
        if new:
            attractor.update(new)
            changed = True
    return attractor

def avoid_set_classical(Sj, Bj, S1, S2, E):
    A1j = attractor_1(E, S1, S2, Bj)
    notA1j = Sj - A1j
    Wj1 = attractor_2(E, S1, S2, notA1j)
    return Wj1

def solve_buchi_game(S, E, S1, S2, B):
    Sj = set(S)
    W_total = set()
    j = 0
    contador=0
    numerador=0

    while True:
        contador=contador+1
        if contador > 100:
            numerador=numerador+1
            print(f"\n Iteracion número {numerador*contador}")
            contador=0
        Bj = B & Sj
        Wj1 = avoid_set_classical(Sj, Bj, S1, S2, E)
        if not Wj1:
            break
        Sj = Sj - Wj1
        W_total |= Wj1
        j += 1

    return Sj, W_total




cgs5 = CGS()

cgs5.add_proposition("safe")
cgs5.add_proposition("start")
cgs5.add_proposition("operational")
cgs5.add_proposition("efficient")
cgs5.add_proposition("underpowered")
cgs5.add_proposition("danger")
cgs5.add_proposition("emergency")

cgs5.add_agent("Reactor")
cgs5.add_agent("Valve")

cgs5.add_decisions("Reactor", {"heat", "cool"})
cgs5.add_decisions("Valve", {"open", "lock"})

cgs5.add_state("start")
cgs5.add_state("efficient")
cgs5.add_state("underpowered")
cgs5.add_state("danger")
cgs5.add_state("shutdown")

cgs5.set_initial_state("start")

cgs5.label_state("start", {"safe","start"})
cgs5.label_state("efficient", {"safe","operational","efficient"})
cgs5.label_state("underpowered", {"safe","operational","underpowered"})
cgs5.label_state("danger", {"operational","danger"})
cgs5.label_state("shutdown", {"emergency"})


cgs5.add_transition("start", {("Reactor", "heat"), ("Valve", "open")}, "start")
cgs5.add_transition("start", {("Reactor", "heat"), ("Valve", "lock")}, "efficient")
cgs5.add_transition("start", {("Reactor", "cool"), ("Valve", "open")}, "start")
cgs5.add_transition("start", {("Reactor", "cool"), ("Valve", "lock")}, "start")

cgs5.add_transition("efficient", {("Reactor", "heat"), ("Valve", "open")}, "efficient")
cgs5.add_transition("efficient", {("Reactor", "heat"), ("Valve", "lock")}, "danger")
cgs5.add_transition("efficient", {("Reactor", "cool"), ("Valve", "open")}, "underpowered")
cgs5.add_transition("efficient", {("Reactor", "cool"), ("Valve", "lock")}, "underpowered")

cgs5.add_transition("underpowered", {("Reactor", "heat"), ("Valve", "open")}, "underpowered")
cgs5.add_transition("underpowered", {("Reactor", "heat"), ("Valve", "lock")}, "efficient")
cgs5.add_transition("underpowered", {("Reactor", "cool"), ("Valve", "open")}, "start")
cgs5.add_transition("underpowered", {("Reactor", "cool"), ("Valve", "lock")}, "start")

cgs5.add_transition("danger", {("Reactor", "heat"), ("Valve", "open")}, "danger")
cgs5.add_transition("danger", {("Reactor", "heat"), ("Valve", "lock")}, "shutdown")
cgs5.add_transition("danger", {("Reactor", "cool"), ("Valve", "open")}, "underpowered")
cgs5.add_transition("danger", {("Reactor", "cool"), ("Valve", "lock")}, "efficient")

cgs5.add_transition("shutdown", {("Reactor", "heat"), ("Valve", "open")}, "start")
cgs5.add_transition("shutdown", {("Reactor", "heat"), ("Valve", "lock")}, "start")
cgs5.add_transition("shutdown", {("Reactor", "cool"), ("Valve", "open")}, "start")
cgs5.add_transition("shutdown", {("Reactor", "cool"), ("Valve", "lock")}, "start")




cgs1 = CGS()

cgs1.add_proposition("safe")
cgs1.add_proposition("start")
cgs1.add_proposition("operational")
cgs1.add_proposition("efficient")
cgs1.add_proposition("underpowered")
cgs1.add_proposition("danger")
cgs1.add_proposition("emergency")

cgs1.add_agent("Reactor")
cgs1.add_agent("Valve")

cgs1.add_decisions("Reactor", {"heat", "cool"})
cgs1.add_decisions("Valve", {"open", "lock"})

cgs1.add_state("s0")
cgs1.add_state("s1")
cgs1.add_state("s2")
cgs1.add_state("s3")
cgs1.add_state("s4")

cgs1.set_initial_state("s0")

cgs1.label_state("s0", {"safe","start"})
cgs1.label_state("s1", {"safe","operational","efficient"})
cgs1.label_state("s2", {"safe","operational","underpowered"})
cgs1.label_state("s3", {"operational","danger"})
cgs1.label_state("s4", {"emergency"})


cgs1.add_transition("s0", {("Reactor", "heat"), ("Valve", "open")}, "s0")
cgs1.add_transition("s0", {("Reactor", "heat"), ("Valve", "lock")}, "s1")
cgs1.add_transition("s0", {("Reactor", "cool"), ("Valve", "open")}, "s0")
cgs1.add_transition("s0", {("Reactor", "cool"), ("Valve", "lock")}, "s0")

cgs1.add_transition("s1", {("Reactor", "heat"), ("Valve", "open")}, "s1")
cgs1.add_transition("s1", {("Reactor", "heat"), ("Valve", "lock")}, "s3")
cgs1.add_transition("s1", {("Reactor", "cool"), ("Valve", "open")}, "s2")
cgs1.add_transition("s1", {("Reactor", "cool"), ("Valve", "lock")}, "s2")

cgs1.add_transition("s2", {("Reactor", "heat"), ("Valve", "open")}, "s2")
cgs1.add_transition("s2", {("Reactor", "heat"), ("Valve", "lock")}, "s1")
cgs1.add_transition("s2", {("Reactor", "cool"), ("Valve", "open")}, "s0")
cgs1.add_transition("s2", {("Reactor", "cool"), ("Valve", "lock")}, "s0")

cgs1.add_transition("s3", {("Reactor", "heat"), ("Valve", "open")}, "s3")
cgs1.add_transition("s3", {("Reactor", "heat"), ("Valve", "lock")}, "s4")
cgs1.add_transition("s3", {("Reactor", "cool"), ("Valve", "open")}, "s2")
cgs1.add_transition("s3", {("Reactor", "cool"), ("Valve", "lock")}, "s1")

cgs1.add_transition("s4", {("Reactor", "heat"), ("Valve", "open")}, "s0")
cgs1.add_transition("s4", {("Reactor", "heat"), ("Valve", "lock")}, "s0")
cgs1.add_transition("s4", {("Reactor", "cool"), ("Valve", "open")}, "s0")
cgs1.add_transition("s4", {("Reactor", "cool"), ("Valve", "lock")}, "s0")



# ------------------------------------------------------------
#  CGS2 : 
# ------------------------------------------------------------
cgs2 = CGS()

for p in (
    "cars_go", "cars_wait", "cross", "dont_cross",
    "clear", "busy", "yellow_phase", "violation",
    "crash", "night_mode", "emergency", "sensor_fault"
):
    cgs2.add_proposition(p)

cgs2.add_agent("CarLight")
cgs2.add_agent("PedLight")

cgs2.add_decisions("CarLight", {"green", "yellow", "red"})
cgs2.add_decisions("PedLight", {"wait", "walk"})

for st in ("s0", "s1", "s2", "s3", "s4", "s5"):
    cgs2.add_state(st)

cgs2.set_initial_state("s1")   

cgs2.label_state("s0", {"cars_wait", "peds_cross", "busy"})
cgs2.label_state("s1", {"cars_wait", "peds_wait", "clear", "night_mode"})
cgs2.label_state("s2", {"cars_wait", "peds_wait", "yellow_phase", "busy"})
cgs2.label_state("s3", {"cars_go", "peds_wait", "busy"})
cgs2.label_state("s4", {"cars_go", "peds_cross", "violation", "busy"})
cgs2.label_state("s5", {"crash", "emergency_peds", "sensor_fault"})

cgs2.add_transition("s0", {("CarLight", "red"), ("PedLight", "walk")}, "s0")
cgs2.add_transition("s0", {("CarLight", "red"), ("PedLight", "dontWalk")}, "s1")
cgs2.add_transition("s0", {("CarLight", "yellow"), ("PedLight", "walk")}, "s4")
cgs2.add_transition("s0", {("CarLight", "yellow"), ("PedLight", "dontWalk")}, "s2")
cgs2.add_transition("s0", {("CarLight", "green"), ("PedLight", "walk")}, "s4")
cgs2.add_transition("s0", {("CarLight", "green"), ("PedLight", "dontWalk")}, "s3")

cgs2.add_transition("s1", {("CarLight", "red"),   ("PedLight", "walk")}, "s0")
cgs2.add_transition("s1", {("CarLight", "red"),   ("PedLight", "dontWalk")},  "s1")
cgs2.add_transition("s1", {("CarLight", "yellow"),("PedLight", "walk")},      "s4")
cgs2.add_transition("s1", {("CarLight", "yellow"),("PedLight", "dontWalk")},  "s2")
cgs2.add_transition("s1", {("CarLight", "green"), ("PedLight", "walk")},      "s4")
cgs2.add_transition("s1", {("CarLight", "green"), ("PedLight", "dontWalk")},  "s3")

cgs2.add_transition("s2", {("CarLight", "red"),   ("PedLight", "walk")},      "s0")
cgs2.add_transition("s2", {("CarLight", "red"),   ("PedLight", "dontWalk")},  "s1")
cgs2.add_transition("s2", {("CarLight", "yellow"),("PedLight", "walk")},      "s4")
cgs2.add_transition("s2", {("CarLight", "yellow"),("PedLight", "dontWalk")},  "s2")
cgs2.add_transition("s2", {("CarLight", "green"), ("PedLight", "walk")},      "s4")
cgs2.add_transition("s2", {("CarLight", "green"), ("PedLight", "dontWalk")},  "s3")

cgs2.add_transition("s3", {("CarLight", "red"),   ("PedLight", "walk")},      "s0")
cgs2.add_transition("s3", {("CarLight", "red"),   ("PedLight", "dontWalk")},  "s1")
cgs2.add_transition("s3", {("CarLight", "yellow"),("PedLight", "walk")},      "s4")
cgs2.add_transition("s3", {("CarLight", "yellow"),("PedLight", "dontWalk")},  "s2")
cgs2.add_transition("s3", {("CarLight", "green"), ("PedLight", "walk")},      "s4")
cgs2.add_transition("s3", {("CarLight", "green"), ("PedLight", "dontWalk")},  "s3")

cgs2.add_transition("s4", {("CarLight", "red"),   ("PedLight", "walk")},      "s0")
cgs2.add_transition("s4", {("CarLight", "red"),   ("PedLight", "dontWalk")},  "s1")
cgs2.add_transition("s4", {("CarLight", "yellow"),("PedLight", "walk")},      "s5")
cgs2.add_transition("s4", {("CarLight", "yellow"),("PedLight", "dontWalk")},  "s2")
cgs2.add_transition("s4", {("CarLight", "green"), ("PedLight", "walk")},      "s5")
cgs2.add_transition("s4", {("CarLight", "green"), ("PedLight", "dontWalk")},  "s4")

for car in ("red", "yellow", "green"):
    for ped in ("walk", "dontWalk"):
        cgs2.add_transition("s5", {("CarLight", car), ("PedLight", ped)}, "s5")


# ------------------------------------------------------------
#  CGS3 : Drone Delivery (20,9)
# ------------------------------------------------------------
cgs3 = CGS()

for p in (
    "idle", "airborne", "en_route", "deliver", "returning", "landed",
    "battery_low", "battery_ok", "gps_ok", "gps_lost", "no_fly_zone",
    "safe_zone", "package_onboard", "package_delivered", "package_lost",
    "emergency", "obstacle_detected", "clear_path", "mission_complete",
    "charging"
):
    cgs3.add_proposition(p)

cgs3.add_agent("Drone")
cgs3.add_agent("Package")

cgs3.add_decisions("Drone", {"fly", "hover", "land"})
cgs3.add_decisions("Package", {"attached", "released"})

for st in ("s0","s1","s2","s3","s4","s5","s6","s7","s8"):
    cgs3.add_state(st)

cgs3.set_initial_state("s0")

cgs3.label_state("s0", {"idle","landed","safe_zone","battery_ok","gps_ok",
                        "package_onboard","clear_path"})
cgs3.label_state("s1", {"airborne","en_route","battery_ok","gps_ok",
                        "package_onboard","clear_path"})
cgs3.label_state("s2", {"airborne","deliver","battery_ok","gps_ok",
                        "package_onboard","safe_zone"})
cgs3.label_state("s3", {"airborne","returning","battery_ok","gps_ok",
                        "package_delivered","clear_path"})
cgs3.label_state("s4", {"airborne","returning","battery_ok","gps_ok",
                        "package_delivered","clear_path"})
cgs3.label_state("s5", {"landed","mission_complete","charging","safe_zone",
                        "battery_ok","package_delivered"})
cgs3.label_state("s6", {"airborne","no_fly_zone","emergency","obstacle_detected",
                        "battery_ok","package_lost"})
cgs3.label_state("s7", {"airborne","gps_lost","emergency","battery_ok",
                        "package_onboard"})
cgs3.label_state("s8", {"airborne","battery_low","emergency","gps_ok",
                        "package_onboard"})

cgs3.add_transition("s0", {("Drone", "fly"),   ("Package", "attached")},  "s1")
cgs3.add_transition("s0", {("Drone", "fly"),   ("Package", "released")},  "s0")
cgs3.add_transition("s0", {("Drone", "hover"), ("Package", "attached")},  "s0")
cgs3.add_transition("s0", {("Drone", "hover"), ("Package", "released")},  "s0")
cgs3.add_transition("s0", {("Drone", "land"),  ("Package", "attached")},  "s0")
cgs3.add_transition("s0", {("Drone", "land"),  ("Package", "released")},  "s0")

cgs3.add_transition("s1", {("Drone", "fly"),   ("Package", "attached")},  "s1")
cgs3.add_transition("s1", {("Drone", "fly"),   ("Package", "released")},  "s6")
cgs3.add_transition("s1", {("Drone", "hover"), ("Package", "attached")},  "s2")
cgs3.add_transition("s1", {("Drone", "hover"), ("Package", "released")},  "s3")
cgs3.add_transition("s1", {("Drone", "land"),  ("Package", "attached")},  "s8")
cgs3.add_transition("s1", {("Drone", "land"),  ("Package", "released")},  "s3")

cgs3.add_transition("s2", {("Drone", "fly"),   ("Package", "attached")},  "s1")
cgs3.add_transition("s2", {("Drone", "fly"),   ("Package", "released")},  "s3")
cgs3.add_transition("s2", {("Drone", "hover"), ("Package", "attached")},  "s2")
cgs3.add_transition("s2", {("Drone", "hover"), ("Package", "released")},  "s3")
cgs3.add_transition("s2", {("Drone", "land"),  ("Package", "attached")},  "s6")
cgs3.add_transition("s2", {("Drone", "land"),  ("Package", "released")},  "s3")

cgs3.add_transition("s3", {("Drone", "fly"),   ("Package", "attached")},  "s6")
cgs3.add_transition("s3", {("Drone", "fly"),   ("Package", "released")},  "s4")
cgs3.add_transition("s3", {("Drone", "hover"), ("Package", "attached")},  "s6")
cgs3.add_transition("s3", {("Drone", "hover"), ("Package", "released")},  "s3")
cgs3.add_transition("s3", {("Drone", "land"),  ("Package", "attached")},  "s6")
cgs3.add_transition("s3", {("Drone", "land"),  ("Package", "released")},  "s4")

cgs3.add_transition("s4", {("Drone", "fly"),   ("Package", "attached")},  "s6")
cgs3.add_transition("s4", {("Drone", "fly"),   ("Package", "released")},  "s4")
cgs3.add_transition("s4", {("Drone", "hover"), ("Package", "attached")},  "s7")
cgs3.add_transition("s4", {("Drone", "hover"), ("Package", "released")},  "s4")
cgs3.add_transition("s4", {("Drone", "land"),  ("Package", "attached")},  "s6")
cgs3.add_transition("s4", {("Drone", "land"),  ("Package", "released")},  "s5")

cgs3.add_transition("s5", {("Drone", "fly"),   ("Package", "attached")},  "s1")
cgs3.add_transition("s5", {("Drone", "fly"),   ("Package", "released")},  "s1")
cgs3.add_transition("s5", {("Drone", "hover"), ("Package", "attached")},  "s5")
cgs3.add_transition("s5", {("Drone", "hover"), ("Package", "released")},  "s5")
cgs3.add_transition("s5", {("Drone", "land"),  ("Package", "attached")},  "s5")
cgs3.add_transition("s5", {("Drone", "land"),  ("Package", "released")},  "s5")

for d in ("fly","hover","land"):
    for p in ("attached","released"):
        cgs3.add_transition("s6", {("Drone", d), ("Package", p)}, "s6")

for d,p in (("land","released"),):
    cgs3.add_transition("s7", {("Drone", d), ("Package", p)}, "s8")
for d in ("fly","hover","land"):
    for p in ("attached","released"):
        if not (d=="land" and p=="released"):
            cgs3.add_transition("s7", {("Drone", d), ("Package", p)}, "s7")

cgs3.add_transition("s8", {("Drone", "land"), ("Package", "released")}, "s5")
cgs3.add_transition("s8", {("Drone", "land"), ("Package", "attached")}, "s6")
for d in ("fly","hover"):
    for p in ("attached","released"):
        cgs3.add_transition("s8", {("Drone", d), ("Package", p)}, "s8")


cgs4 = CGS()

for p in (
    "robot_zone1", "robot_zone2", "robot_zone3",
    "robot_idle", "robot_holding", "robot_empty",
    "conveyor_forward", "conveyor_reverse", "conveyor_stop",
    "item_on_conv", "item_at_zone1", "item_at_zone2", "item_at_zone3",
    "item_sorted", "package_lost",
    "sensor_clear", "sensor_blocked", "obstacle_detected",
    "jammed", "collision", "conveyor_overload",
    "battery_low", "battery_ok", "arm_overheat",
    "maintenance_mode", "manual_override",
    "emergency_stop", "shutdown",
    "goal_reached", "charging"
):
    cgs4.add_proposition(p)

cgs4.add_agent("RobotArm")
cgs4.add_agent("Conveyor")

cgs4.add_decisions("RobotArm",
                   {"move_left", "move_right", "stay", "pick", "drop"})
cgs4.add_decisions("Conveyor",
                   {"forward", "reverse", "stop"})

for st in ("s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11"):
    cgs4.add_state(st)

cgs4.set_initial_state("s0")

cgs4.label_state("s0", {"robot_zone1","robot_empty","robot_idle",
                        "conveyor_stop","item_at_zone1",
                        "sensor_clear","battery_ok"})
cgs4.label_state("s1", {"robot_zone1","robot_holding",
                        "conveyor_stop","sensor_clear"})
cgs4.label_state("s2", {"robot_zone2","robot_holding",
                        "conveyor_stop","sensor_clear"})
cgs4.label_state("s3", {"robot_zone2","robot_holding",
                        "conveyor_forward","item_on_conv","sensor_clear"})
cgs4.label_state("s4", {"robot_zone3","robot_holding",
                        "conveyor_stop","sensor_clear"})
cgs4.label_state("s5", {"robot_zone3","robot_empty","item_sorted",
                        "goal_reached","conveyor_stop","sensor_clear"})
cgs4.label_state("s6", {"jammed","conveyor_stop","sensor_blocked"})
cgs4.label_state("s7", {"collision","emergency_stop"})
cgs4.label_state("s8", {"battery_low","conveyor_stop","charging"})
cgs4.label_state("s9", {"maintenance_mode","conveyor_stop","manual_override"})
cgs4.label_state("s10",{"conveyor_reverse","item_on_conv","sensor_clear"})
cgs4.label_state("s11",{"shutdown","emergency_stop"})

RA = ("move_left", "move_right", "stay", "pick", "drop")
CV = ("forward", "reverse", "stop")

def add_all(state, spec, default_dest):
    """Añade las 15 combinaciones (RA × CV) usando spec para los casos especiales."""
    for ra in RA:
        for cv in CV:
            dest = spec.get((ra, cv), default_dest)
            cgs4.add_transition(
                state,
                {("RobotArm", ra), ("Conveyor", cv)},
                dest
            )

add_all("s0",
    {("pick", "stop"): "s1"},
    "s0"
)

add_all("s1",
    {("move_right", "stop"): "s2",
     ("drop",       "stop"): "s0"},
    "s1"
)

add_all("s2",
    {("stay",       "forward"): "s3",
     ("move_left",  "stop"):    "s1",
     ("drop",       "forward"): "s7",
     ("stay",       "reverse"): "s10"},
    "s2"
)

add_all("s3",
    {("move_right", "stop"):    "s4",
     ("stay",       "forward"): "s6"},
    "s3"
)

add_all("s4",
    {("drop",      "stop"): "s5",
     ("move_left", "stop"): "s3"},
    "s4"
)

add_all("s5",
    {
     ("move_left", "reverse"): "s0",
     ("stay",      "stop"):    "s8"},
    "s5"
)

add_all("s6",
    {
     ("stay",      "stop"): "s9"},
    "s6"
)

add_all("s7",
    {("stay", "stop"): "s11"},   
    "s7"                         
)


add_all("s8",
    {("stay", "stop"): "s0"},   
    "s8"
)

add_all("s9",
    {("stay", "stop"): "s0"},  
    "s9"
)

add_all("s10",
    {("stay", "stop"):    "s2", 
     ("stay", "forward"): "s3"},
    "s10"
)

add_all("s11", {}, "s11")



#=========================================#
#          FORMULA GENERATOR              #
#=========================================#


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

def random_node(cgs, depth,
                inside_modality=False,
                modality_needed=True):
    # → Hoja atómica cuando depth ≤ 1
    if depth <= 1:
        return random_var(cgs)          # o T / F si lo prefieres

    # → Inserta una modalidad obligatoria con pequeña probabilidad
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
    else:                               # salvaguarda
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
    return 1   # T, F, etc.

def generate_random_valid_atl_formula(cgs, depth,
                                      modality_needed=True,
                                      max_tries=10_000):
    tries = 0
    while tries < max_tries:
        raw = random_node(cgs, depth=depth, modality_needed=modality_needed)
        f   = normalize_formula(raw)
        if filter(f) == "ATL" and formula_depth(f) == depth:
            return f
        tries += 1
    raise RuntimeError(f"No se pudo generar fórmula de profundidad {depth} tras {max_tries} intentos")

def generate_valid_formulas_by_depth(cgs,
                                     min_depth: int,
                                     max_depth: int,
                                     samples_per_depth: int):
    for depth in range(min_depth, max_depth + 1):
        print(f"\n Depth {depth}")
        for i in range(1, samples_per_depth + 1):
            f = generate_random_valid_atl_formula(
                    cgs, depth,
                    modality_needed=(depth >= 3)
                )
            assert formula_depth(f) == depth
            print(f"  ✔️ {f.to_formula()}")
    print("\n")
    print("\n")

# ===========================
#  ACG BUILD
# ===========================


def classify_game_node(node):
    if isinstance(node, ExistentialAtom):
        return 'S1'
    if isinstance(node, UniversalAtom):
        return 'S2'
    if isinstance(node, Conj) or isinstance(node, Disj):
        atoms = {node.lhs, node.rhs}
        if any(isinstance(a, ExistentialAtom) for a in atoms):
            return 'S2'
        if any(isinstance(a, UniversalAtom) for a in atoms):
            return 'S1'
        return 'S2'  # Default: nondeterminism adversarial
    if isinstance(node, Top):
        return 'S1'
    if isinstance(node, Bottom):
        return 'S2'
    return 'S1'

def build_game_from_acg_robust(acg):
    S = set()
    E = {}

    for (src, _), transition in acg.transitions.items():
        def expand(t):
            if isinstance(t, (Conj, Disj)):
                S.add(t)
                E[(src, t)] = True
                expand(t.lhs)
                expand(t.rhs)
            else:
                S.add(t)
                E[(src, t)] = True
                if hasattr(t, 'state'):
                    S.add(t.state)
                    E[(t, t.state)] = True

        S.add(src)
        expand(transition)

    for const in [Top(), Bottom()]:
        S.add(const)
        E[(const, const)] = True

    S1, S2 = set(), set()
    for node in S:
        (S1 if classify_game_node(node) == 'S1' else S2).add(node)

    B = set(acg.final_states)

    return S, E, S1, S2, B

def generate_and_display_game_from_formula(cgs,depth=3):

    formula = generate_random_valid_atl_formula(cgs, depth, modality_needed=(depth >= 3))

    print("\n" + "="*80)
    print("🧠 Formula (AST):")
    print(formula.to_formula())
    print(formula.to_tree())

    acg, _, _ = build_acg_with_timer(formula)

    print("\n" + "="*80)
    print("🏗️  ACG:")
    print(acg)

    S, E, S1, S2, B = build_game_from_acg_robust(acg)

    print("\n" + "="*80)
    print("🎮 Game:")
    print(f"States (|S| = {len(S)}):")
    for s in S:
        print("  ", s)
    print(f"\nPlayer 1 States (S1 = {len(S1)}):")
    for s in S1:
        print("  ", s)
    print(f"\nPlayer 2 States (S2 = {len(S2)}):")
    for s in S2:
        print("  ", s)
    print(f"\nTransitions (|E| = {len(E)}):")
    for (src, dst) in E:
        print(f"  ({src}) → ({dst})")
    print(f"\nFinal/Winning States (|B| = {len(B)}):")
    for s in B:
        print("  ", s)


    G = nx.DiGraph()
    for (src, dst) in E:
        G.add_edge(str(src), str(dst))

    color_map = []
    for node in G.nodes:
        node_obj = next((s for s in S1 if str(s) == node), None)
        if node_obj:
            color_map.append("skyblue")  
        elif any(str(s) == node for s in S2):
            color_map.append("lightgreen")  
        else:
            color_map.append("gray")  

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=1800, font_size=7, arrows=True, node_color=color_map)
    plt.title("🎮 Büchi Game Graph from ACG (Color = Owner)")
    plt.tight_layout()
    plt.show()


# ===========================
#  METRICS
# ===========================


def acg_vs_cgs_compare(
    cgs,                    # instancia CGS
    cgs_id: str,            # etiqueta para CSV / log
    min_depth: int,
    max_depth: int,
    samples_per_depth: int,
    csv_path: Path,
    overwrite_csv: bool = False   # ← TRUE = reescribe archivo
):
    """
    Para cada fórmula aleatoria genera:
      • ACG denso   → juego producto → resuelve
      • ACG compacto → juego producto → resuelve
    Registra dos filas (dense/compact) por fórmula.
    """

    # ------------ constructores de ACG (ya definidos por ti)
    VARIANTS = [
        ("dense",   build_acg_with_timer),   # función que devuelve (acg, acg_size, build_t)
        ("compact", build_acg_with_timer2),
    ]

    # ------------ abrir CSV
    mode = "w" if overwrite_csv else "a"
    need_header = overwrite_csv or not csv_path.exists()
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if need_header:
            writer.writerow([
                "cgs_id", "batch_size",
                "depth", "sample_idx",
                "acg_variant",
                "formula", "formula_len", "gen_time",
                "acg_states", "acg_edges", "acg_build_time", "acg_size",
                "game_states", "game_edges", "game_build_time",
                "solve_time", "total_time", "satisfiable"
            ])

        log(f"▶ START  CGS={cgs_id}  batch={samples_per_depth}  variants=2")

        # -------------------------------------------------- #
        # bucle profundidades
        # -------------------------------------------------- #
        for depth in range(min_depth, max_depth + 1):
            sample = 0
            while sample < samples_per_depth:

                # 1) generar fórmula (una sola vez)
                log(f"  depth={depth}  sample={sample+1}/{samples_per_depth} – generating formula")
                t_form0 = time.perf_counter()
                try:
                    formula = generate_random_valid_atl_formula(
                        cgs, depth=depth,
                        modality_needed=(depth >= 3)
                    )
                except Exception as e:
                    log(f"    ❌ formula generation failed: {e}")
                    traceback.print_exc()
                    continue        # reintenta esta misma muestra

                gen_time = time.perf_counter() - t_form0
                formula_str = formula.to_formula()
                log(f"    formula generated ({len(formula_str)} chars)")

                # 2) ejecutar ambas variantes
                for variant_name, build_fn in VARIANTS:
                    try:
                        t_var0 = time.perf_counter()         # ← cronómetro variante

                        # -------- ACG ----------
                        acg, acg_size, acg_build_time = build_fn(formula)
                        acg_states = len(acg.states)
                        acg_edges  = len(acg.transitions)

                        # -------- Producto -----
                        t_p0 = time.perf_counter()
                        S,E,S1,S2,B,s0 = build_game(acg, cgs)
                        game_build_time = time.perf_counter() - t_p0
                        game_states, game_edges = len(S), len(E)

                        # -------- Solver -------
                        t_s0 = time.perf_counter()
                        S_win, _ = solve_buchi_game(S,E,S1,S2,B)
                        solve_time = time.perf_counter() - t_s0

                        total_time = time.perf_counter() - t_var0   # ← sólo esta variante
                        satisfiable = "Yes" if s0 in S_win else "No"

                        # -------- CSV ----------
                        writer.writerow([
                            cgs_id, samples_per_depth,
                            depth, sample+1,
                            variant_name,
                            formula_str, len(formula_str), f"{gen_time:.6f}",
                            acg_states, acg_edges, f"{acg_build_time:.6f}", acg_size,
                            game_states, game_edges, f"{game_build_time:.6f}",
                            f"{solve_time:.6f}", f"{total_time:.6f}", satisfiable
                        ])
                        f.flush()
                        log(f"    [{variant_name}] done  total={total_time:.3f}s  sat={satisfiable}")

                    except Exception as e:
                        log(f"    ❌ Exception in {variant_name}: {e}")
                        traceback.print_exc()

                sample += 1        # ambas variantes terminadas

        log(f"■ FINISHED CGS={cgs_id} batch={samples_per_depth}\n")


def heavy_pipeline(phi, cgs):
    """
    Ejecuta todo lo caro y devuelve las métricas.
    Se define en el nivel superior para que sea pickle-able.
    """
    # ACG
    acg, acg_size, acg_build = build_acg_with_timer2(phi,cgs)
    acg_states, acg_edges = len(acg.states), len(acg.transitions)

    # Juego
    t_g0 = time.perf_counter()
    S,E,S1,S2,B,s0 = build_game(acg, cgs)
    game_build = time.perf_counter() - t_g0
    game_states, game_edges = len(S), len(E)

    # Solver
    t_s0 = time.perf_counter()
    S_win,_ = solve_buchi_game(S,E,S1,S2,B)
    solve_t = time.perf_counter() - t_s0

    total_t = acg_build + game_build + solve_t
    sat     = "Yes" if s0 in S_win else "No"

    return (acg_states, acg_edges, acg_build, acg_size,
            game_states, game_edges, game_build,
            solve_t, total_t, sat)

def _timeout_worker(q, fn, args):
    try:
        q.put(("OK", fn(*args)))
    except Exception as e:
        q.put(("ERR", e))

def run_with_timeout(fn, timeout, *args):
    q = mp.Queue()
    p = mp.Process(target=_timeout_worker, args=(q, fn, args))
    t0 = time.perf_counter()
    p.start(); p.join(timeout)
    elapsed = time.perf_counter() - t0

    if p.is_alive():          # timeout
        p.terminate(); p.join()
        return False, "TIMEOUT", elapsed

    status, res = q.get()
    if status == "OK":
        return True, res, elapsed
    else:
        return False, res, elapsed


def compact_acg_study(
        cgs, cgs_id,
        min_depth, max_depth,
        samples_per_depth,
        csv_path: Path,
        overwrite_csv: bool = False,
        max_seconds: int = 120          # límite por intento
):
    mode        = "w" if overwrite_csv else "a"
    need_header = overwrite_csv or not csv_path.exists()

    header = [
        "cgs_id", "batch_size",
        "depth", "sample_idx",
        "status",                      # NEW
        "formula", "formula_len", "gen_time",
        "acg_states", "acg_edges", "acg_build_time", "acg_size",
        "game_states", "game_edges", "game_build_time",
        "solve_time", "total_time", "satisfiable"
    ]

    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if need_header:
            wr.writerow(header)

        log(f"▶ START {cgs_id}  depths {min_depth}–{max_depth} "
            f"samples={samples_per_depth}  timeout={max_seconds}s")

        for depth in range(min_depth, max_depth + 1):
            ok_count  = 0
            attempt   = 0
            while ok_count < samples_per_depth:
                attempt += 1

                # 1) generación de fórmula (rápida)
                try:
                    t_gen0   = time.perf_counter()
                    phi      = generate_random_valid_atl_formula(
                                  cgs, depth=depth, modality_needed=(depth >= 3))
                    gen_time = time.perf_counter() - t_gen0
                    phi_str  = phi.to_formula()
                except Exception as e:
                    wr.writerow([cgs_id, samples_per_depth,
                                 depth, attempt, "ERROR",
                                 "", "", "", "", "", "", "",
                                 "", "", "", "", "", "ERROR"])
                    log(f"    ❌ formula gen error: {e}")
                    continue

                # 2) pipeline pesado con límite
                def _run():
                    # ACG
                    acg, acg_size, acg_build = build_acg_with_timer2(phi,cgs)
                    acg_states, acg_edges = len(acg.states), len(acg.transitions)

                    # Juego
                    t_g0 = time.perf_counter()
                    S,E,S1,S2,B,s0 = build_game(acg, cgs)
                    game_build = time.perf_counter() - t_g0
                    game_states, game_edges = len(S), len(E)

                    # Solver
                    t_s0 = time.perf_counter()
                    S_win,_ = solve_buchi_game(S,E,S1,S2,B)
                    solve_t = time.perf_counter() - t_s0

                    total_t = acg_build + game_build + solve_t
                    sat     = "Yes" if s0 in S_win else "No"

                    return (acg_states, acg_edges, acg_build, acg_size,
                            game_states, game_edges, game_build,
                            solve_t, total_t, sat)

                ok, res, runtime = run_with_timeout(heavy_pipeline, max_seconds, phi, cgs)

                if not ok:                 # TIMEOUT o ERROR en pipeline
                    status = "TIMEOUT" if res == "TIMEOUT" else "ERROR"
                    wr.writerow([cgs_id, samples_per_depth,
                                 depth, attempt, status,
                                 phi_str[:60]+"…" if len(phi_str)>60 else phi_str,
                                 len(phi_str), f"{gen_time:.6f}",
                                 "", "", "", "",
                                 "", "", "",
                                 "", f"{runtime:.6f}", status])
                    log(f"    {status} after {runtime:.1f}s")
                    continue                # NO cuenta para ok_count

                # 3) éxito: escribir todas las métricas
                (acg_states, acg_edges, acg_build, acg_size,
                 game_states, game_edges, game_build,
                 solve_t, total_t, sat) = res

                wr.writerow([cgs_id, samples_per_depth,
                             depth, ok_count+1, "OK",
                             phi_str[:60]+"…" if len(phi_str)>60 else phi_str,
                             len(phi_str), f"{gen_time:.6f}",
                             acg_states, acg_edges, f"{acg_build:.6f}", acg_size,
                             game_states, game_edges, f"{game_build:.6f}",
                             f"{solve_t:.6f}", f"{total_t:.6f}", sat])
                log(f"    OK [{ok_count+1}/{samples_per_depth}] "
                    f"{total_t:.2f}s sat={sat}")
                ok_count += 1
                f.flush()

        log(f"■ FINISHED {cgs_id}\n")


# ===========================
#  PLOTTING
# ===========================


def plot_mean_acg_build_time(csv_path: str,
                             save_as: str | None = None,
                             figsize: tuple = (6, 4)):
    """
    Dibuja el tiempo medio de construcción del ACG (columna 'acg_build_time')
    para cada CGS y cada nivel de profundidad.
    """
    # 1) Cargar datos
    df = pd.read_csv(csv_path)

    # 2) Agrupar por CGS y profundidad, tomar la media
    mean_build = (
        df.groupby(["cgs_id", "depth"])["acg_build_time"]
          .mean()
          .reset_index()
    )

    # 3) Dibujar
    plt.figure(figsize=figsize)

    for cgs_id, sub in mean_build.groupby("cgs_id"):
        plt.plot(
            sub["depth"],               # eje X
            sub["acg_build_time"],      # eje Y
            marker="o",
            label=cgs_id
        )

    plt.title("Mean ACG build time vs. depth")
    plt.xlabel("Depth of formula (d)")
    plt.ylabel("Mean build time (s)")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()

def plot_satisfiable_ratio(csv_path: str,
                           save_as: str | None = None,
                           figsize: tuple = (6, 4)):
    """
    Muestra P(satisfiable) = (# fórmulas SAT / total) por profundidad y CGS.

    • Eje X: depth de la fórmula.
    • Eje Y: probabilidad (0-1).
    • Una línea por cada CGS.

    Parameters
    ----------
    csv_path : str
        Ruta al CSV que contiene las columnas 'cgs_id', 'depth'
        y 'satisfiable' ("Yes"/"No").
    save_as : str | None, optional
        Si se indica, guarda la figura en esa ruta.
    figsize : tuple, optional
        Dimensiones (ancho, alto) de la figura en pulgadas.
    """
    df = pd.read_csv(csv_path)

    # 1) Convertir 'satisfiable' a 0/1
    df["sat"] = (df["satisfiable"] == "Yes").astype(int)

    # 2) Agrupar y promediar  → esto da la “probabilidad empírica”
    sat_ratio = (
        df.groupby(["cgs_id", "depth"])["sat"]
          .mean()           # mean de 0/1 = proporción de Yes
          .reset_index()
    )

    # 3) Dibujar
    plt.figure(figsize=figsize)

    for cgs_id, sub in sat_ratio.groupby("cgs_id"):
        plt.plot(
            sub["depth"],
            sub["sat"],
            marker="o",
            label=cgs_id
        )

    plt.title("Satisfiable ratio vs. depth")
    plt.xlabel("Depth of formula (d)")
    plt.ylabel("P(satisfiable)")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()

def plot_global_build_times(csv_path: str, save_as: str | None = None):
    df = pd.read_csv(csv_path)

    mean_global = (                      # media sobre TODAS las filas
        df.groupby(["acg_variant", "depth"])["acg_build_time"]
          .mean()
          .reset_index()
    )

    plt.figure(figsize=(6,4))
    for variant, sub in mean_global.groupby("acg_variant"):
        plt.plot(sub["depth"], sub["acg_build_time"],
                 marker="o", label=variant)

    plt.title("Mean ACG build time (4 CGS aggregated)")
    plt.xlabel("Depth of formula")
    plt.ylabel("Mean build time (s)")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150); plt.close()
    else:
        plt.show()

def plot_build_times_per_cgs(csv_path: str, save_dir: str | None = None):
    df = pd.read_csv(csv_path)

    per_cgs = (
        df.groupby(["cgs_id", "acg_variant", "depth"])["acg_build_time"]
          .mean()
          .reset_index()
    )

    for cid, subc in per_cgs.groupby("cgs_id"):
        plt.figure(figsize=(6,4))
        for variant, sub in subc.groupby("acg_variant"):
            plt.plot(sub["depth"], sub["acg_build_time"],
                     marker="o", label=variant)
        plt.title(f"{cid} – Mean ACG build time")
        plt.xlabel("Depth")
        plt.ylabel("Mean build time (s)")
        plt.grid(True, alpha=.3)
        plt.legend()
        plt.tight_layout()

        if save_dir:
            path = Path(save_dir) / f"{cid}_build_time.png"
            plt.savefig(path, dpi=150); plt.close()
            print(f"  ▸ saved {path}")
        else:
            plt.show()

def plot_mean_acg_size(csv_path: str,
                       save_as: str | None = None,
                       figsize: tuple = (6, 4),
                       agg: str = "mean"):
    
    df = pd.read_csv(csv_path)

    size_stat = (
        df.groupby(["cgs_id", "depth"])["acg_size"]
          .agg(agg)
          .reset_index()
    )

    plt.figure(figsize=figsize)
    for cid, sub in size_stat.groupby("cgs_id"):
        plt.plot(sub["depth"], sub["acg_size"],
                 marker="o", label=cid)

    plt.title(f"{agg.capitalize()} ACG size vs depth (compact)")
    plt.xlabel("Depth of formula")
    plt.ylabel(f"{agg.capitalize()} ACG size  (#states + #atoms)")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()

def plot_mean_game_build_time(csv_path: str,
                              save_as: str | None = None,
                              figsize: tuple = (6, 4),
                              agg: str = "mean"):
  
    df = pd.read_csv(csv_path)

    stat = (
        df.groupby(["cgs_id", "depth"])["game_build_time"]
          .agg(agg)
          .reset_index()
    )

    plt.figure(figsize=figsize)
    for cid, sub in stat.groupby("cgs_id"):
        plt.plot(sub["depth"], sub["game_build_time"],
                 marker="o", label=cid)

    plt.title(f"{agg.capitalize()} game build time vs depth")
    plt.xlabel("Depth")
    plt.ylabel(f"{agg.capitalize()} build time (s)")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150); plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()

def plot_mean_game_size(csv_path: str,
                        metric: str = "game_states",   # o "game_edges"
                        save_as: str | None = None,
                        figsize: tuple = (6, 4),
                        agg: str = "mean"):
    
    if metric not in {"game_states", "game_edges"}:
        raise ValueError("metric debe ser 'game_states' o 'game_edges'")

    df = pd.read_csv(csv_path)

    stat = (
        df.groupby(["cgs_id", "depth"])[metric]
          .agg(agg)
          .reset_index()
    )

    plt.figure(figsize=figsize)
    for cid, sub in stat.groupby("cgs_id"):
        plt.plot(sub["depth"], sub[metric],
                 marker="o", label=cid)

    pretty = "states" if metric == "game_states" else "edges"
    plt.title(f"{agg.capitalize()} game {pretty} vs depth")
    plt.xlabel("Depth")
    plt.ylabel(f"{agg.capitalize()} #{pretty}")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150); plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()

def plot_time_breakdown_by_depth(csv_path,
                                 agg="mean",
                                 cgs_filter=None,
                                 absolute=False,        # ← True = escala absoluta
                                 figsize=(7,4),
                                 save_as=None):

    df = pd.read_csv(csv_path)
    if cgs_filter:
        df = df[df["cgs_id"].isin(cgs_filter)]

    phases = ["gen_time", "acg_build_time",
              "game_build_time", "solve_time"]
    colors = {
        "gen_time":        "#FFC857",
        "acg_build_time":  "#F55D3E",
        "game_build_time": "#3E7CB1",
        "solve_time":      "#2AB7CA"
    }

    grouped = (
        df.groupby("depth")[phases]
          .agg(agg)
          .reset_index()
          .sort_values("depth")
    )

    if not absolute:              # normaliza a proporciones
        grouped[phases] = grouped[phases].div(
            grouped[phases].sum(axis=1), axis=0)

    # ---------- plot ----------
    plt.figure(figsize=figsize)
    bottom = None
    for p in phases:
        plt.bar(grouped["depth"],
                grouped[p],
                bottom=bottom,
                color=colors[p],
                edgecolor="k",
                label=p.replace("_", " "))
        bottom = grouped[p] if bottom is None else bottom + grouped[p]

    scale = "Time (s)" if absolute else "Proportion of total time"
    plt.ylabel(scale)
    plt.xlabel("Depth of formula")
    plt.title(f"{agg.capitalize()} time breakdown per depth")
    plt.ylim(0, bottom.max()*1.05)
    plt.grid(axis="y", alpha=.3)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()

LOG = Path("run.log").open("a", encoding="utf-8")

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)



# ────────────────────────────────────────────────────────────────
# FAMILY BENCHMARKING
# ────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────
# Globally

def flatG_clause(i: int, A: Coalition):
    p = Var(f"online_{i}")             # o cualquier p_i
    body = Globally(Not(p))            # G ¬online_i
    return Modality(A, body)           # ⟨A⟩ G …

def generate_flatG_spec(n: int):
    A = full_coalition(n)
    # Arrancamos directamente con la cláusula de i=0:
    phi = flatG_clause(0, A)
    for i in range(1, n):
        phi = And(phi, flatG_clause(i, A))
    return phi

def generate_flatG_OR_spec(n: int):
    A = full_coalition(n)
    # Arrancamos directamente con la cláusula de i=0:
    phi = flatG_clause(0, A)
    for i in range(1, n):
        phi = Or(phi, flatG_clause(i, A))
    return phi

GOAL_PROP = "goal"

def nested_G_formula(n: int, A: Coalition):
    """
    ⟨A⟩ G ⟨A⟩ G … ⟨A⟩ G goal   (n copies of ⟨A⟩ G)
    """
    node = Var(GOAL_PROP)
    for _ in range(n):
        node = Modality(A, Globally(node))
    return node 

def build_deepG_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    return nested_G_formula(n, A)

def generate_reactor_cgs_deepG(n: int) -> CGS:
    g = generate_reactor_cgs(n)   # your existing helper
    g.add_proposition(GOAL_PROP)        # goal is always false
    return g


# ────────────────────────────────────────────────────────────────
# Flat Until

def flatU_clause(i: int, A: Coalition):
    """
    ⟨A⟩ (online_i  U  overheated_i)

    • Hijo directo de la modalidad  =  Until
    • Sin modalidades anidadas dentro del U
    """
    on  = Var(f"online_{i}")
    ov  = Var(f"overheated_{i}")
    core = Until(on, ov)               # online_i  U  overheated_i
    return Modality(A, core)           # ⟨A⟩ ( … )

def generate_flatU_spec(n: int):
    """
    ψₙ =  ⋀_{i=0}^{n-1}  ⟨Aₙ⟩ (online_i U overheated_i)
    """
    A   = full_coalition(n)
    psi = flatU_clause(0, A)
    for i in range(1,n):
        psi = And(psi, flatU_clause(i, A))
    return psi

def generate_flatU_OR_spec(n: int):
    """
    ψₙ =  ⋀_{i=0}^{n-1}  ⟨Aₙ⟩ (online_i U overheated_i)
    """
    A   = full_coalition(n)
    psi = flatU_clause(0, A)
    for i in range(1,n):
        psi = Or(psi, flatU_clause(i, A))
    return psi


def nested_U_formula(n: int, A: Coalition):
    """
    φₙ = ⟨A⟩( p₀ U ⟨A⟩( p₁ U ⋯ ⟨A⟩( pₙ₋₁ U goal )⋯ ) )
    • Cada nivel anida un Until dentro de la modalidad.
    """
    # empezamos por el fondo: goal
    node = Var(GOAL_PROP)
    # luego, de i = n-1 ↓ 0, envolvemos en ⟨A⟩( prop_i U node )
    for i in reversed(range(n)):
        pi = Var(f"prop_{i}")
        node = Modality(A, Until(pi, node))
    return node

def generate_deepU_spec(n: int):
    A = full_coalition(n)
    return nested_U_formula(n, A)

def generate_reactor_cgs_deepU(n: int) -> CGS:
    # reutilizamos tu CGS base para reactor
    g = generate_reactor_cgs(n)

    # 1) añadimos las prop_i de la cláusula
    for i in range(n):
        g.add_proposition(f"prop_{i}")
    # 2) añadimos goal
    g.add_proposition(GOAL_PROP)

    # 3) etiquetamos el estado inicial con todas las prop_i
    #    de modo que cada p_i sea alcanzable en el primer paso
    init = g.initial_state
    lab = set(g.labeling_function[init])
    for i in range(n):
        lab.add(f"prop_{i}")
    g.label_state(init, lab)

    # (no etiquetamos nunca “goal” para que la fórmula sea insatisfacible
    #  hasta el nivel más profundo, forzando el peor caso)

    return g



# ────────────────────────────────────────────────────────────────
# Flat Next



def generate_flatX_OR_spec(n: int):
    """
    ψₙ = ⋀_{i=0}^{n-1}  ⟨Aₙ⟩ X prop_i
    """
    A   = full_coalition(n)      # {ctrl_0 , … , ctrl_{n-1}}
    psi = flatX_clause(0, A)
    for i in range(1,n):
        psi = Or(psi, flatX_clause(i, A))
    return psi

def generate_reactor_cgs_flatX(n: int) -> CGS:
    g = CGS()

    # 2n proposiciones (de la familia original) …
    for i in range(n):
        g.add_proposition(f"overheated_{i}")
        g.add_proposition(f"online_{i}")

    # … y n proposiciones nuevas prop_i usadas en ψₙ
    for i in range(n):
        g.add_proposition(f"prop_{i}")

    # agentes y decisiones idénticos
    for i in range(n):
        a = f"ctrl_{i}"
        g.add_agent(a)
        g.add_decisions(a, {f"startup_{i}", f"detonate_{i}"})

    # ------------------- estados coherentes ---------------------
    states = []
    for ov in product([0, 1], repeat=n):
        for on in product([0, 1], repeat=n):
            if any(ov[j] and on[j] for j in range(n)):   # “overheated ⇒ ¬online”
                continue
            states.append((tuple(ov), tuple(on)))

    for s in states:
        g.add_state(s)
        lab = {f"overheated_{i}" for i, b in enumerate(s[0]) if b} | \
              {f"online_{i}"     for i, b in enumerate(s[1]) if b}

        # --- OPCIÓN: hacer satisfacible la cláusula ----------------
        # activamos prop_i únicamente en el estado inicial,
        # así <A> X prop_i es alcanzable en un paso.
        if s == states[0]:                      # todo apagado
            for i in range(n):
                lab.add(f"prop_{i}")
        # ----------------------------------------------------------------
        g.label_state(s, lab)

    g.set_initial_state(states[0])

    # ------------------- transición determinista ------------------
    for s in states:
        for joint in product(*[[(a, d) for d in g.decisions[a]]
                               for a in sorted(g.agents)]):
            ov, on = list(s[0]), list(s[1])
            for (agent, act) in joint:
                idx = int(agent.split('_')[1])
                if act.startswith("startup"):
                    on[idx] = 1
                elif act.startswith("detonate"):
                    ov[idx], on[idx] = 1, 0
            # coherencia: overheated ⇒ ¬online
            for j in range(n):
                if ov[j]:
                    on[j] = 0
            g.add_transition(s, joint, (tuple(ov), tuple(on)))

    return g


def nested_X_formula(n: int, A: Coalition):
    """
    Builds ⟨A⟩X ⟨A⟩X … ⟨A⟩X goal   (depth = n)
    """
    node = Var(GOAL_PROP)
    for _ in range(n):
        node = Modality(A, Next(node))
    return node       # this is ϕₙ

def build_deepX_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    return nested_X_formula(n, A)

def generate_reactor_cgs_deepX(n: int) -> CGS:
    g = generate_reactor_cgs_flatX(n)   # reuse your helper
    g.add_proposition(GOAL_PROP)
    # goal is never labelled ⇒ unsatisfiable
    return g


# <A>X<A>X<A>X---goal


GOAL_PROP = "goal"

def nested_X_formula(n: int, A: Coalition):
    """
    Builds ⟨A⟩X ⟨A⟩X … ⟨A⟩X goal   (depth = n)
    """
    node = Var(GOAL_PROP)
    for _ in range(n):
        node = Modality(A, Next(node))
    return node       # this is ϕₙ

def build_deepX_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    return nested_X_formula(n, A)

def generate_reactor_cgs_deepX(n: int) -> CGS:
    g = generate_reactor_cgs_flatX(n)   # reuse your helper
    g.add_proposition(GOAL_PROP)
    # goal is never labelled ⇒ unsatisfiable
    return g

def nested_G_formula(n: int, A: Coalition):
    """
    ⟨A⟩ G ⟨A⟩ G … ⟨A⟩ G goal   (n copies of ⟨A⟩ G)
    """
    node = Var(GOAL_PROP)
    for _ in range(n):
        node = Modality(A, Globally(node))
    return node 

def build_deepG_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    return nested_G_formula(n, A)

def generate_reactor_cgs_deepG(n: int) -> CGS:
    g = generate_reactor_cgs(n)   # your existing helper
    g.add_proposition(GOAL_PROP)        # goal is always false
    return g

def nested_U_formula(n: int, A: Coalition):
    """
    φₙ = ⟨A⟩( p₀ U ⟨A⟩( p₁ U ⋯ ⟨A⟩( pₙ₋₁ U goal )⋯ ) )
    • Cada nivel anida un Until dentro de la modalidad.
    """
    # empezamos por el fondo: goal
    node = Var(GOAL_PROP)
    # luego, de i = n-1 ↓ 0, envolvemos en ⟨A⟩( prop_i U node )
    for i in reversed(range(n)):
        pi = Var(f"prop_{i}")
        node = Modality(A, Until(pi, node))
    return node

def generate_deepU_spec(n: int):
    A = full_coalition(n)
    return nested_U_formula(n, A)

def generate_reactor_cgs_deepU(n: int) -> CGS:
    # reutilizamos tu CGS base para reactor
    g = generate_reactor_cgs(n)

    # 1) añadimos las prop_i de la cláusula
    for i in range(n):
        g.add_proposition(f"prop_{i}")
    # 2) añadimos goal
    g.add_proposition(GOAL_PROP)

    # 3) etiquetamos el estado inicial con todas las prop_i
    #    de modo que cada p_i sea alcanzable en el primer paso
    init = g.initial_state
    lab = set(g.labeling_function[init])
    for i in range(n):
        lab.add(f"prop_{i}")
    g.label_state(init, lab)

    # (no etiquetamos nunca “goal” para que la fórmula sea insatisfacible
    #  hasta el nivel más profundo, forzando el peor caso)

    return g



def generate_reactor_cgs(n: int) -> CGS:
    g = CGS()

    for i in range(n):
        g.add_proposition(f"overheated_{i}")
        g.add_proposition(f"online_{i}")

    for i in range(n):
        a = f"ctrl_{i}"
        g.add_agent(a)
        g.add_decisions(a, {f"startup_{i}", f"detonate_{i}"})


    states = []
    for ov in product([0,1], repeat=n):
        for on in product([0,1], repeat=n):
            if any(ov[j] and on[j] for j in range(n)):   
                continue
            states.append((tuple(ov), tuple(on)))

    for s in states:
        g.add_state(s)
        lab = {f"overheated_{i}" for i,b in enumerate(s[0]) if b} | \
              {f"online_{i}"     for i,b in enumerate(s[1]) if b}
        g.label_state(s, lab)

    g.set_initial_state(states[0])          # todos apagados

    # 4) transición determinista coherente
    for s in states:
        for joint in product(*[[(a,d) for d in g.decisions[a]]
                               for a in sorted(g.agents)]):
            ov, on = list(s[0]), list(s[1])
            for (agent, act) in joint:
                idx = int(agent.split('_')[1])
                if act.startswith("startup"):
                    on[idx] = 1
                elif act.startswith("detonate"):
                    ov[idx] = 1; on[idx] = 0
            # coherencia: overheated ⇒ ¬online
            for j in range(n):
                if ov[j] == 1:
                    on[j] = 0
            g.add_transition(s, joint, (tuple(ov), tuple(on)))

    return g


# ────────────────────────────────────────────────────────────────
# SAMPLEO MANUAL
# ────────────────


def sample_satisfiable_formulas(
        cgs,
        min_depth: int,
        max_depth: int,
        sat_per_depth: int,
        max_attempts: int = 5000,
        max_seconds: float | None = None   # None = sin límite
):

    print("=" * 80)
    print(f"CGS: {cgs}\n")

    for depth in range(min_depth, max_depth + 1):
        print(f"\n🟦 DEPTH {depth}")
        found, tries = 0, 0

        while found < sat_per_depth and tries < max_attempts:
            tries += 1
            # ---------- 1) generar fórmula ATL válida ----------
            phi = generate_random_valid_atl_formula(
                      cgs, depth, modality_needed=(depth >= 3))
            phi_str = phi.to_formula()
            print(f"Formula :{phi_str}")
         

            # ---------- 2) cronometrar pipeline opcional -------
            t0 = time.perf_counter()

            # ACG
            acg, acg_size, acg_build = build_acg_with_timer2(phi)
            print(acg)

            # Juego
            S,E,S1,S2,B,s0 = build_game(acg, cgs)
            print("\n🎮 Game Stats:")
            print(f"  |S| = {len(S)}  |E| = {len(E)}  |S1| = {len(S1)}  |S2| = {len(S2)}  |B| = {len(B)}")
            print("📌 Transitions (|E| = {}):".format(len(E)))
            for (src, dst) in E:
                print(f"  ({pretty_node(src)}) → ({pretty_node(dst)})")


            # Solver
            S_win,_ = solve_buchi_game(S,E,S1,S2,B)
            sat = (s0 in S_win)

            elapsed = time.perf_counter() - t0

            #  Si no es satisfacible, seguimos buscando
            if not sat:
                continue

            found += 1
            print("─" * 60)
            print(f"⧗  Fórmula #{found}  (intento {tries})")
            print(phi_str)
            print(f"  • ACG:  |Q|={len(acg.states)}  |δ|={len(acg.transitions)}")
            print(f"  • Game: |S|={len(S)}  |E|={len(E)}  |S1|={len(S1)}  "
                  f"|S2|={len(S2)}  |B|={len(B)}")
            print(f"  • Tiempo total: {elapsed:.3f} s")
            if max_seconds and elapsed > max_seconds:
                print("    ⚠️  excede límite solicitado")

            if max_seconds and elapsed > max_seconds:
                # si quieres descartar las que exceden cierto tiempo,
                # elimina el 'found += 1' anterior y coloca continue aquí
                pass

        if found < sat_per_depth:
            print(f"⚠️  Solo se encontraron {found} fórmulas "
                  f"satisfacibles tras {tries} intentos.")

def TESTEONORMAL(input_formula_str: str, cgs):
    print("=" * 80)
    print("📥 Input Formula String:")
    print(input_formula_str)

    try:
        # 1. Tokenize + Parse
        formula0=tokenize(input_formula_str)
        formula = parse(formula0)
        print("\n🌳 Parsed AST:")
        print(formula.to_tree())

        # 2. Normalize
        formula = normalize_formula(formula)
        print("\n🧹 Normalized Formula:")
        print(formula.to_formula())

        # 3. Filter ATL check
        if filter(formula) != "ATL":
            print("\n❌ Formula is not ATL after normalization.")
            return

        # 4. Build ACG
        acg, size, elapsed = build_acg_with_timer2(formula,cgs)
        print("\n🏗️  ACG:")
        print(acg)
        print(f"\n📐 ACG Size: {size} | ⏱️ Time: {elapsed:.4f} seconds")

        # 5. Build Büchi Game
        S, E, S1, S2, B, s0 = build_game(acg, cgs)

        print("\n🎮 Game Stats:")
        print(f"  |S| = {len(S)}  |E| = {len(E)}  |S1| = {len(S1)}  |S2| = {len(S2)}  |B| = {len(B)}")
        print("📌 Transitions (|E| = {}):".format(len(E)))
        for (src, dst) in E:
            print(f"  ({pretty_node(src)}) → ({pretty_node(dst)})")

        # 6. Solve Büchi Game
        S_win, _ = solve_buchi_game(S, E, S1, S2, B)

        satisfiable = s0 in S_win
        print(f"\n✅ SATISFIABLE? {satisfiable}")

    except Exception as e:
        print("❌ An error occurred:")
        import traceback
        traceback.print_exc()

def TESTEONORMAL_FINAL(input_formula_str: str, cgs):
    print("=" * 80)
    print("📥 Input Formula String:")
    print(input_formula_str)

    try:
        # 1. Tokenize + Parse
        formula0=tokenize(input_formula_str)
        formula = parse(formula0)
        print("\n🌳 Parsed AST:")
        print(formula.to_tree())

        # 2. Normalize
        formula = normalize_formula(formula)
        print("\n🧹 Normalized Formula:")
        print(formula.to_formula())

        # 3. Filter ATL check
        if filter(formula) != "ATL":
            print("\n❌ Formula is not ATL after normalization.")
            return

        # 4. Build ACG
        acg, size, elapsed = build_acg_with_timer_final(formula,cgs)
        print("\n🏗️  ACG:")
        print(acg)
        print(f"\n📐 ACG Size: {size} | ⏱️ Time: {elapsed:.4f} seconds")

        # 5. Build Büchi Game
        S, E, S1, S2, B, s0 = build_game(acg, cgs)

        print("\n🎮 Game Stats:")
        print(f"  |S| = {len(S)}  |E| = {len(E)}  |S1| = {len(S1)}  |S2| = {len(S2)}  |B| = {len(B)}")
        print("📌 Transitions (|E| = {}):".format(len(E)))
        for (src, dst) in E:
            print(f"  ({pretty_node(src)}) → ({pretty_node(dst)})")

        # 6. Solve Büchi Game
        S_win, _ = solve_buchi_game(S, E, S1, S2, B)

        satisfiable = s0 in S_win
        print(f"\n✅ SATISFIABLE? {satisfiable}")

    except Exception as e:
        print("❌ An error occurred:")
        import traceback
        traceback.print_exc()



# ────────────────────────────────────────────────────────────────
# PARAMETRICS
# ────────────────

class Coalition(list):
    def __sub__(self, other):   # A − A′
        return set(self) - set(other)

def full_coalition(n: int) -> Coalition:
    return Coalition(f"ctrl_{i}" for i in range(n))


def generate_lights_cgs(n: int) -> CGS:
    """
    CGS where n agents ctrl_0,…,ctrl_{n-1} each
    control a binary switch. Propositions p_i = "switch i is on".
    Actions: toggle_i (invertir bit) o wait (mantenerlo).
    """
    g = CGS()

    # 1) Proposiciones p_i
    for i in range(n):
        g.add_proposition(f"p_{i}")

    # 2) Agentes y decisiones
    for i in range(n):
        a = f"ctrl_{i}"
        g.add_agent(a)
        # cada agente puede “toggle_i” o “wait”
        g.add_decisions(a, {f"toggle_{i}", "wait"})

    # 3) Estados: todos los vectores de {0,1}^n
    states = [tuple(bs) for bs in product([0,1], repeat=n)]
    for s in states:
        g.add_state(s)
        # Etiquetado: p_i está si y solo si s[i]==1
        lab = {f"p_{i}" for i,bit in enumerate(s) if bit}
        g.label_state(s, lab)

    # 4) Estado inicial: todos apagados (0,…,0)
    init = tuple(0 for _ in range(n))
    g.set_initial_state(init)

    # 5) Transición determinista
    for s in states:
        for joint in product(*[[(a,d) for d in g.decisions[a]]
                               for a in sorted(g.agents)]):
            # arranco del bit-vector actual
            new_bits = list(s)
            # aplico cada acción
            for (agent, act) in joint:
                idx = int(agent.split("_")[1])
                if act == f"toggle_{idx}":
                    new_bits[idx] = 1 - new_bits[idx]
                # if act == "wait": no hace falta cambiar
            # siguiente estado
            s_next = tuple(new_bits)
            g.add_transition(s, joint, s_next)

    return g

# ────────────────────────────────────────────────────────────────
#  Next

def flatX_clause(i: int, A: Coalition):
    """
    ⟨A⟩ X p_i
    """
    p = Var(f"p_{i}")              
    return Modality(A, Next(p))

def generate_flatX_spec(n: int):
    """
    ψₙ = ⋀_{i=0}^{n-1} ⟨Aₙ⟩ X p_i
    """
    A = full_coalition(n)
    psi = flatX_clause(0, A)
    for i in range(1, n):
        psi = And(psi, flatX_clause(i, A))
    return psi

def generate_flatX_OR_spec(n: int):
    """
    ψₙ = ⋀_{i=0}^{n-1} ⟨Aₙ⟩ X p_i
    """
    A = full_coalition(n)
    # empiezo directamente con el primer término (sin T)
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
    """
    ¬⟨A⟩ X p_i
    """
    p = Var(f"p_{i}")
    return Not(Modality(A, Next(p)))

def generate_negated_flatX_spec(n: int):
    """
    ψₙ = ⋀_{i=0}^{n-1} ¬⟨Aₙ⟩ X p_i
    """
    A = full_coalition(n)
    phi = negated_flatX_clause(0, A)
    for i in range(1, n):
        phi = And(phi, negated_flatX_clause(i, A))
    return phi

def nested_X_formula(n: int, A: Coalition):
    """
    Builds ⟨A⟩X ⟨A⟩X … ⟨A⟩X goal   (depth = n)
    """
    node = Var(f"p_{n-1}")
    for _ in range(n):
        node = Modality(A, Next(node))
    return node  

def generate_negated_nestedX_spec(n: int) -> ParseNode:
    """
    ψₙ = ¬(⟨Aₙ⟩X ⟨Aₙ⟩X … ⟨Aₙ⟩X p_{n-1})
    Negate entire nested X over full coalition.
    """
    A = full_coalition(n)
    nested = nested_X_formula(n, A)
    return Not(nested)

def generate_stepwise_negated_nestedX_spec(n: int) -> ParseNode:
    """
    ψₙ = ¬⟨Aₙ⟩X ¬⟨Aₙ⟩X … ¬⟨Aₙ⟩X p_{n-1}
    Negate each modality stepwise over full coalition.
    """
    A = full_coalition(n)
    node = Var(f"p_{n-1}")
    for _ in range(n):
        node = Not(Modality(A, Next(node)))
    return node

def generate_negated_nestedX_individual_spec(n: int) -> ParseNode:
    """
    φₙ = ¬(⟨{ctrl_0}⟩X ⟨{ctrl_1}⟩X … ⟨{ctrl_{n-1}}⟩X p_{n-1})
    Negate entire nested X over individual controllers.
    """
    nested = nested_X_individual_formula(n)
    return Not(nested)

def generate_stepwise_negated_nestedX_individual_spec(n: int) -> ParseNode:
    """
    φₙ = ¬⟨{ctrl_0}⟩X ¬⟨{ctrl_1}⟩X … ¬⟨{ctrl_{n-1}}⟩X p_{n-1}
    Negate each modality stepwise over individual controllers.
    """
    node = Var(f"p_{n-1}")
    for i in reversed(range(n)):
        node = Not(Modality([f"ctrl_{i}"], Next(node)))
    return node


def build_deepX_spec(n: int) -> ParseNode:
    A = full_coalition(n)
    return nested_X_formula(n, A)

def nested_X_individual_formula(n: int):
    node = Var(f"p_{n-1}")
    for i in reversed(range(n)):
        node = Modality([f"ctrl_{i}"], Next(node))
    return node

def build_nestedX_individual_spec(n: int):
    return nested_X_individual_formula(n)

# ────────────────────────────────────────────────────────────────
# Globally

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
    """
    ¬⟨A⟩ G p_i
    """
    p = Var(f"p_{i}")
    return Not(Modality(A, Globally(p)))

def generate_negated_flatG_spec(n: int):
    """
    ψₙ = ⋀_{i=0}^{n-1} ¬⟨Aₙ⟩ G p_i
    """
    A = full_coalition(n)
    phi = negated_flatG_clause(0, A)
    for i in range(1, n):
        phi = And(phi, negated_flatG_clause(i, A))
    return phi


def nested_G_formula(n: int, A: Coalition) :
    """
    Builds ⟨A⟩ G ⟨A⟩ G … ⟨A⟩ G p_{n-1}   (depth = n)
    """
    node = Var(f"p_{n-1}")
    for _ in range(n):
        node = Modality(A, Globally(node))
    return node

def generate_negated_nestedG_spec(n: int) -> ParseNode:
    """
    ψₙ = ¬(⟨Aₙ⟩ G ⟨Aₙ⟩ G … ⟨Aₙ⟩ G p_{n-1})
    Negate entire nested G over full coalition.
    """
    A = full_coalition(n)
    nested = nested_G_formula(n, A)
    return Not(nested)

def generate_stepwise_negated_nestedG_spec(n: int) -> ParseNode:
    """
    ψₙ = ¬⟨Aₙ⟩ G ¬⟨Aₙ⟩ G … ¬⟨Aₙ⟩ G p_{n-1}
    Negate each modality stepwise over full coalition.
    """
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


# ────────────────────────────────────────────────────────────────
# Until

def flatU_clause(i: int, A: Coalition):
    """
    ⟨A⟩ (¬p_i U p_i)
    """
    p = Var(f"p_{i}")
    return Modality(A, Until(Not(p), p))

def generate_flatU_spec(n: int) :
    A=full_coalition(n)
    phi = flatU_clause(0,A)
    for i in range(1, n):
        phi = And(phi, flatU_clause(i,A))
    return phi

def generate_flatU_OR_spec(n: int) :
    A=full_coalition(n)
    phi = flatU_clause(0,A)
    for i in range(1, n):
        phi = Or(phi, flatU_clause(i,A))
    return phi

def flatU_individual_clause(i: int):
    p = Var(f"p_{i}")
    return Modality([f"ctrl_{i}"], Until(Not(p), p))

def generate_flatU_individual_spec(n: int):
    phi = flatU_individual_clause(0)
    for i in range(1, n):
        phi = And(phi, flatU_individual_clause(i))
    return phi

def nested_U_formula(n: int, A: Coalition) :
    """
    Builds: ⟨A⟩ (p_0 U ⟨A⟩ (p_1 U ⟨A⟩ ( ... (p_{n-2} U p_{n-1}) ... )))
    """
    assert n >= 2, "Nested U requires at least two levels (n ≥ 2)"
    phi = Var(f"p_{n-1}")
    for i in reversed(range(n - 1)):
        left = Var(f"p_{i}")
        phi = Modality(A, Until(left, phi))
    return phi

def build_deepU_spec(n: int) :
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
    """
    ψₙ = ⋀_{i=0}^{n-1} ¬⟨Aₙ⟩ (¬p_i U p_i)
    Negate each flat U clause over full coalition.
    """
    A = full_coalition(n)
    phi = Not(Modality(A, Until(Not(Var("p_0")), Var("p_0"))))
    for i in range(1, n):
        clause = Not(Modality(A, Until(Not(Var(f"p_{i}")), Var(f"p_{i}"))))
        phi = Modality(A, clause) if False else And(phi, clause)  # fix And import if needed
    return phi

def generate_negated_nestedU_spec(n: int) -> ParseNode:
    """
    ψₙ = ¬(⟨Aₙ⟩ (p_0 U (⟨Aₙ⟩ (p_1 U … p_{n-1}))) )
    Negate entire nested U over full coalition.
    """
    A = full_coalition(n)
    nested = nested_U_formula(n, A)
    return Not(nested)

def generate_stepwise_negated_nestedU_spec(n: int) -> ParseNode:
    """
    ψₙ = ¬⟨Aₙ⟩(p_0 U ¬⟨Aₙ⟩(p_1 U … ¬⟨Aₙ⟩(p_{n-2} U p_{n-1})))
    """
    A = full_coalition(n)

    # Base: p_{n-2} U p_{n-1}  (o p_0 si n<2)
    if n >= 2:
        node = Until(Var(f"p_{n-2}"), Var(f"p_{n-1}"))
    else:
        node = Var("p_0")

    # Desde i = n-2 hasta i = 0
    # En cada paso hacemos: node = Until(p_i, Not(Modality(A, node)))
    for i in reversed(range(n-1)):
        # 1) negar+modalidad sobre lo anterior
        neg_mod = Not(Modality(A, node))
        # 2) envolver con Until(p_i, neg_mod)
        node = Until(Var(f"p_{i}"), neg_mod)

    # Finalmente, ¬⟨A⟩(...) sobre el todo
    return Not(Modality(A, node))

def generate_negated_nestedU_individual_spec(n: int) -> ParseNode:
    nested = nested_U_individual_formula(n)
    return Not(nested)


def generate_stepwise_negated_nestedU_individual_spec(n: int) -> ParseNode:
    # Build base until
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




def scalability_family(n_min, n_max,
                       cgs_builder,         # p.ej. generate_reactor_cgs
                       formula_builder,     # p.ej. generate_flatG_spec
                       family_id,           # "Flat-G"
                       csv_path: Path):

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "family", "n", "states",
            "acg_size", "acg_build",
            "game_states", "game_edges",  # <-- ya estaba en el header
            "game_build", "solve_time", "total_time", "sat"
        ])

        for n in range(n_min, n_max+1):
            print(f"▶ {family_id}  n={n}")
            cgs = cgs_builder(n)
            phi = formula_builder(n)
            print(phi.to_formula())

            t0 = time.perf_counter()
            acg, acg_size, acg_build = build_acg_with_timer2(phi, cgs)
            print(f"ACG TIEMPO {acg_build}\n")

            # — Construcción del juego y medida de tiempo —
            t1 = time.perf_counter()
            S, E, S1, S2, B, s0 = build_game(acg, cgs)
            game_build = time.perf_counter() - t1
            # — ¡Aquí calculamos game_states y game_edges! —
            game_states = len(S)
            game_edges  = len(E)
            print(f"GAME STATES {game_states}, GAME EDGES {game_edges}, TIEMPO {game_build}\n")

            # — Resolución del juego —
            t2 = time.perf_counter()
            S_win, _ = solve_buchi_game(S, E, S1, S2, B)
            solve_time = time.perf_counter() - t2
            total_time = time.perf_counter() - t0

            print(f"SOLVE TIEMPO {solve_time}\n")
            print(f"TOTAL TIEMPO {total_time}\n")

            # — Escribimos la fila incluyendo game_edges —
            wr.writerow([
                family_id,
                n,
                len(cgs.states),
                acg_size,
                acg_build,
                game_states,
                game_edges,   # <-- nuevo
                game_build,
                solve_time,
                total_time,
                "Yes" if s0 in S_win else "No"
            ])


# ────────────────────────────────────────────────────────────────
# PLOTEOS VARIOS
# ───────────────

def plot_scalability(csv_path: Path,
                     *,          # solo args nombrados extra
                     in_minutes: bool = False,
                     logy: bool = False,
                     save_as: str | None = None):

    import pandas as pd, matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)

    # --- preparación ------------------------------
    y_col   = "total_time_min" if in_minutes else "total_time"
    if in_minutes:
        df[y_col] = df["total_time"] / 60.0

    # --- plot -------------------------------------
    plt.figure(figsize=(6,4))
    plt.plot(df["states"], df[y_col], marker="o")

    plt.xscale("log", base=2)
    if logy:
        plt.yscale("log", base=10)

    plt.xlabel("|S|  (log₂)")
    plt.ylabel(f"Total time ({'min' if in_minutes else 's'})")
    plt.title("Scalability – Reactor family")
    plt.grid(True, which="both", alpha=.3)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150); plt.close()
    else:
        plt.show()

def plot_timeout_rate(csv_path, save_as=None):
    """
    Lee el CSV con la columna 'status' y dibuja, para cada CGS,
    el porcentaje de TIMEOUTs por profundidad.
    """
    df = pd.read_csv(csv_path)

    # ── recuentos OK / TIMEOUT por (cgs_id, depth) ───────────────
    counts = (
        df.groupby(["cgs_id", "depth", "status"])
          .size()                       # nº de filas por estado
          .unstack(fill_value=0)        # columnas: OK, TIMEOUT, ERROR
    )

    # añade columna porcentaje (TIMEOUT / totalIntentos)
    counts["timeout_pct"] = 100 * counts.get("TIMEOUT", 0) / counts.sum(axis=1)

    # ── dibujar una línea por CGS ────────────────────────────────
    plt.figure(figsize=(6,4))
    for cgs_id, sub in counts["timeout_pct"].unstack("cgs_id").items():
        plt.plot(sub.index, sub.values, marker="o", label=cgs_id)

    plt.xlabel("Profundidad de la fórmula")
    plt.ylabel("% de TIMEOUTs")
    plt.title("Porcentaje de descartes por timeout")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_mean_solve_time(csv_path: str,
                         save_as: str | None = None,
                         figsize: tuple = (6, 4),
                         agg: str = "mean",
                         log_y: bool = False):
    """
    Dibuja el tiempo medio/mediano/etc. (agg) que tarda el solver de
    juegos de Büchi, agrupado por CGS y profundidad de la fórmula.

    Parameters
    ----------
    csv_path : str
        Fichero CSV generado por `compact_acg_study` (debe contener
        la columna  solve_time).
    save_as : str | None
        Ruta donde guardar la figura (png).  None → la muestra en pantalla.
    figsize : tuple
        Tamaño de la figura en pulgadas.
    agg : {"mean", "median", "max", ...}
        Medida de agregación aplicada tras el group-by.
    log_y : bool
        Si True, usa eje Y logarítmico (útil cuando hay gran dispersión).
    """

    # 1) cargar y agrupar
    df = pd.read_csv(csv_path)

    stat = (
        df.groupby(["cgs_id", "depth"])["solve_time"]
          .agg(agg)
          .reset_index()
    )

    # 2) dibujar
    plt.figure(figsize=figsize)

    for cid, sub in stat.groupby("cgs_id"):
        plt.plot(sub["depth"], sub["solve_time"],
                 marker="o", label=cid)

    plt.title(f"{agg.capitalize()} Büchi-solver time vs depth")
    plt.xlabel("Depth of formula")
    plt.ylabel(f"{agg.capitalize()} solve time (s)")
    if log_y:
        plt.yscale("log")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()

    # 3) guardar o mostrar
    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()

def plot_families_scalability(csv_paths: list[Path],
                               *,
                               logy: bool = False,
                               in_minutes: bool = False,
                               save_as: str | None = None):

    plt.figure(figsize=(8, 5))

    for path in csv_paths:
        df = pd.read_csv(path)
        family = df["family"].iloc[0]
        x = df["states"]
        y = df["total_time"] / 60.0 if in_minutes else df["total_time"]

        plt.plot(x, y, marker="o", label=family)

    plt.xscale("log", base=2)
    if logy:
        plt.yscale("log", base=10)

    plt.xlabel("|S| (log₂)")
    plt.ylabel(f"Total time ({'min' if in_minutes else 's'})")
    plt.title("Scalability Comparison – Parametric Families")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_buchi_complexity(csv_paths: list[Path]):
    """
    Para cada CSV en `csv_paths`, asume columnas:
      game_states, game_edges, solve_time
    Muestra en pantalla el plot en escala log–log:
      log(nm) vs. log(solve_time)
    junto a la recta ajustada y su pendiente.
    """
    plt.figure(figsize=(8,6))
    
    markers = ['o','s','^','D','v','X','P','*']
    linestyles = ['-','--','-.',':']
    
    for i, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        n  = df['game_states']
        m  = df['game_edges']
        t  = df['solve_time']
        nm = n * m
        
        log_nm = np.log(nm)
        log_t  = np.log(t)
        
        # ajuste lineal en escala log–log
        slope, intercept = np.polyfit(log_nm, log_t, 1)
        line = np.poly1d((slope, intercept))
        
        # puntos y línea de ajuste
        plt.scatter(log_nm, log_t,
                    marker=markers[i % len(markers)],
                    alpha=0.7,
                    label=f"{path.stem} (s={slope:.2f})")
        xs = np.linspace(log_nm.min(), log_nm.max(), 100)
        plt.plot(xs, line(xs),
                 linestyle=linestyles[i % len(linestyles)],
                 linewidth=1.5)
    
    plt.xlabel('log(n × m)')
    plt.ylabel('log(solve_time)')
    plt.title('Complejidad empírica de solver de Büchi')
    plt.legend(title='CSV (pendiente)', loc='best')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_buchi_promedio(csv_paths: list[Path]):
    """
    A partir de varios CSVs con columnas:
      game_states, game_edges, solve_time
    calcula el log de nm=n*m y de solve_time,
    promedia log(solve_time) para cada nm, y plotea:
      1) el promedio empírico,
      2) la línea teórica de pendiente 1.
    """
    # 1) Cargar y preparar cada CSV
    list_dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        nm = df['game_states'] * df['game_edges']
        list_dfs.append(pd.DataFrame({
            'nm':      nm,
            'log_nm':  np.log(nm),
            'log_t':   np.log(df['solve_time'])
        }))
    
    # 2) Concatenar y agrupar para promediar log_t por cada nm
    all_data = pd.concat(list_dfs, ignore_index=True)
    promedio = (
        all_data
        .groupby('nm', as_index=False)
        .agg(log_nm=('log_nm', 'first'),
             log_t =('log_t', 'mean'))
        .sort_values('log_nm')
    )
    
    # 3) Dibujar el plot
    plt.figure(figsize=(8,6))
    
    # 3a) Curva empírica promedio
    plt.plot(promedio['log_nm'],
             promedio['log_t'],
             marker='o',
             linestyle='-',
             label='Promedio empírico')
    
    # 3b) Línea teórica de pendiente 1 (pasa por el mismo punto inicial)
    x0, y0 = promedio['log_nm'].iloc[0], promedio['log_t'].iloc[0]
    xs = np.array([promedio['log_nm'].min(),
                   promedio['log_nm'].max()])
    ys = y0 + 1*(xs - x0)   # pendiente=1, ajustada al punto (x0,y0)
    plt.plot(xs, ys, linestyle='--', label='Teórica (pendiente=1)')
    
    # 4) Ajustes finales
    plt.xlabel('log(n × m)')
    plt.ylabel('log(solve_time)')
    plt.title('Comparativa promedio vs. teórica')
    plt.legend(loc='best')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_average_solver(csv_paths: list[Path],
                        *,
                        log_scale: bool = True):
    """
    Lee varios CSV con columnas:
      game_states, game_edges, solve_time
    calcula por nivel n:
      nm = states * edges
      c  = solve_time / nm
    promedia nm y c, reconstruye t_med = c_med * nm_med,
    y dibuja:
      - puntos empíricos (nm_med, t_med)
      - línea teórica t = nm

    Parámetros:
    - csv_paths: lista de Paths a CSVs
    - log_scale: True para usar escala log-log; False para escala lineal
    """
    # 1) Cargar y combinar
    recs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        rec = pd.DataFrame({
            'n':    df['game_states'],
            'nm':   df['game_states'] * df['game_edges'],
            'c':    df['solve_time'] / (df['game_states'] * df['game_edges'])
        })
        recs.append(rec)
    all_df = pd.concat(recs, ignore_index=True)

    # 2) Agregar por nivel n
    summary = (
        all_df
        .groupby('n', as_index=False)
        .agg(
            nm_med=('nm', 'mean'),
            c_med =('c',  'mean')
        )
        .sort_values('n')
    )
    summary['t_med'] = summary['c_med'] * summary['nm_med']

    x = summary['nm_med']
    y = summary['t_med']

    # 3) Dibujar
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, marker='o', label='Promedio empírico')

    # línea teórica t = nm
    xs = np.linspace(x.min(), x.max(), 100)
    plt.plot(xs, xs, 'k--', label='Teórica (t = n×m)')

    # 4) Elegir escala
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')

    plt.xlabel('n × m (media por nivel)')
    plt.ylabel('solve_time (media)')
    plt.title('Comparativa promedio vs. teórica')
    plt.legend(loc='best')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_all_points(csv_paths: list[Path]):
    """
    Lee múltiples CSVs con columnas:
      family, game_states, game_edges, solve_time
    y hace un scatter log–log de:
      x = game_states * game_edges
      y = solve_time
    coloreando cada punto según su 'family',
    y superpone la recta teórica t = n*m (una línea con pendiente 1 en log–log).
    """

    # 1) Concatenar todos los datos
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p, usecols=['family','game_states','game_edges','solve_time'])
        df = df.assign(
            nm = df['game_states'] * df['game_edges']
        )
        dfs.append(df[['family','nm','solve_time']])
    all_df = pd.concat(dfs, ignore_index=True)
    
    # 2) Colores por familia
    families = all_df['family'].unique()
    colors   = plt.cm.tab10.colors
    fam_to_color = {fam: c for fam, c in zip(families, cycle(colors))}
    
    # 3) Scatter log–log
    plt.figure(figsize=(8,6))
    for fam, group in all_df.groupby('family'):
        plt.scatter(group['nm'], group['solve_time'],
                    label=fam,
                    color=fam_to_color[fam],
                    s=40, alpha=0.7)
    
    # 4) Recta teórica en log–log: basta con plot y=x, 
    #    y luego poner ejes log para que salga pendiente 1
    xmin, xmax = all_df['nm'].min(), all_df['nm'].max()
    xs = np.linspace(xmin, xmax, 200)
    plt.plot(xs, xs, 'k--', label='O(state·edges)')
    
    # 5) Poner ejes en logaritmo
    plt.xscale('log')
    plt.yscale('log')
    
    # 6) Ajustes finales
    plt.xlabel('state·edges')
    plt.ylabel('log solve_time (s)')
    plt.title('Buchi solver complexity')
    plt.legend(title='Family', loc='best')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_time_breakdown_by_level(csv_path: Path,
                                 level_col: str = "n",
                                 agg: str = "mean",
                                 cgs_filter: list[str] | None = None,
                                 absolute: bool = False,
                                 figsize: tuple[int,int] = (7,4),
                                 save_as: str | None = None):
    """
    Igual que antes, pero con una paleta suave de azul a morado.
    """
    # 1) Leer CSV y filtrar
    df = pd.read_csv(csv_path)
    if cgs_filter and "cgs_id" in df.columns:
        df = df[df["cgs_id"].isin(cgs_filter)]
    
    # 2) Detectar columnas ..._time excepto total_time
    time_cols = [c for c in df.columns
                 if c.endswith("_time") and c != "total_time"]
    if not time_cols:
        raise ValueError("No hay columnas *_time a plotear.")
    
    # 3) Agrupar y agregar
    grouped = (
        df
        .groupby(level_col)[time_cols]
        .agg(agg)
        .reset_index()
        .sort_values(level_col)
    )
    
    # 4) Normalizar si es relativo
    if not absolute:
        grouped[time_cols] = grouped[time_cols].div(
            grouped[time_cols].sum(axis=1), axis=0
        )
    
    # 5) Generar paleta continua (azul → morado)
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0.3, 0.8, len(time_cols)))
    
    # 6) Dibujar stacked‑bar
    plt.figure(figsize=figsize)
    bottom = None
    for col, color in zip(time_cols, colors):
        plt.bar(grouped[level_col],
                grouped[col],
                bottom=bottom,
                color=color,
                edgecolor="white",
                label=col.replace("_", " "))
        bottom = grouped[col] if bottom is None else bottom + grouped[col]
    
    # 7) Etiquetas y estilo
    ylabel = "Time (s)" if absolute else "Proportion of total time"
    plt.ylabel(ylabel)
    plt.xlabel(level_col)
    plt.title(f"{agg.capitalize()} time breakdown per {level_col}")
    plt.ylim(0, bottom.max() * 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    
    # 8) Mostrar o guardar
    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()







def plot_mean_acg_build_time_by_states(csv_path: str,
                                       save_as: str | None = None,
                                       figsize: tuple = (6, 4),
                                       per_cgs: bool = False,
                                       agg: str = "mean",
                                       errorbars: bool = True,
                                       filter_status_ok: bool = True):
    """
    Promedia acg_build_time por cada valor de acg_states (y opcionalmente por cgs_id),
    y ajusta una recta sobre los puntos promediados. Muestra R^2.
    """
    df = pd.read_csv(csv_path)

    # Opcional: filtra filas inválidas
    if filter_status_ok and "status" in df.columns:
        df = df[df["status"] == "OK"].copy()

    # Elige agregación
    agg_funcs = {"mean": "mean", "median": "median"}
    if agg not in agg_funcs:
        raise ValueError("agg debe ser 'mean' o 'median'")
    ylab = f"{agg.capitalize()} ACG build time (s)"

    plt.figure(figsize=figsize)

    if per_cgs:
        groups = df.groupby(["cgs_id", "acg_states"])
        stat = groups["acg_build_time"].agg(["mean", "median", "std", "count"]).reset_index()

        for cid, sub in stat.groupby("cgs_id"):
            x = sub["acg_states"].to_numpy(dtype=float)
            y = sub[agg].to_numpy(dtype=float)
            plt.plot(x, y, marker="o", label=cid)

            if errorbars and agg == "mean":
                err = (sub["std"] / np.sqrt(sub["count"].clip(lower=1))).fillna(0).to_numpy()
                plt.errorbar(x, y, yerr=err, fmt="none", capsize=3, alpha=.5)
        # Ajuste lineal global sobre los promedios (todas las series)
        X = stat["acg_states"].to_numpy(dtype=float)
        Y = stat[agg].to_numpy(dtype=float)
    else:
        stat = (
            df.groupby(["acg_states"])["acg_build_time"]
              .agg(["mean", "median", "std", "count"])
              .reset_index()
        )
        x = stat["acg_states"].to_numpy(dtype=float)
        y = stat[agg].to_numpy(dtype=float)
        plt.plot(x, y, marker="o")
        if errorbars and agg == "mean":
            err = (stat["std"] / np.sqrt(stat["count"].clip(lower=1))).fillna(0).to_numpy()
            plt.errorbar(x, y, yerr=err, fmt="none", capsize=3, alpha=.5)
        X, Y = x, y

    # Ajuste lineal y R^2 sobre puntos promediados
    if len(X) >= 2 and np.ptp(X) > 0:
        m, b = np.polyfit(X, Y, 1)
        Yhat = m * X + b
        ss_res = np.sum((Y - Yhat) ** 2)
        ss_tot = np.sum((Y - Y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        xx = np.linspace(X.min(), X.max(), 100)
        plt.plot(xx, m * xx + b, linestyle="--")
        title = f"{agg.capitalize()} build time vs #states (slope={m:.3e} s/state, R²={r2:.3f})"
    else:
        title = f"{agg.capitalize()} build time vs #states"

    plt.title(title)
    plt.xlabel("#ACG states")
    plt.ylabel(ylab)
    plt.grid(True, alpha=.3)
    if per_cgs:
        plt.legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150); plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()

def plot_mean_acg_build_time_by_states2(
    csv_path: str,
    save_as: str | None = None,
    figsize: tuple = (6, 4),
    per_cgs: bool = False,
    agg: str = "mean",
    errorbars: bool = True,
    filter_status_ok: bool = True,
    smooth: bool = False,
    smooth_method: str = "rolling",
    window: int = 5,
    polyorder: int = 2
):
    """
    Grafica el tiempo de construcción de ACG vs número de estados.
    Opcionalmente aplica suavizado (rolling o Savitzky-Golay).

    :param smooth: habilita suavizado de la serie.
    :param smooth_method: 'rolling' o 'savgol'.
    :param window: ventana para rolling o Savitzky-Golay.
    :param polyorder: orden para Savitzky-Golay.
    """
    df = pd.read_csv(csv_path)

    if filter_status_ok and "status" in df.columns:
        df = df[df["status"] == "OK"].copy()

    if agg not in ("mean", "median"):
        raise ValueError("agg debe ser 'mean' o 'median'")

    plt.figure(figsize=figsize)

    if per_cgs:
        grouping = df.groupby(["cgs_id", "acg_states"])["acg_build_time"]
    else:
        grouping = df.groupby("acg_states")["acg_build_time"]

    stat = grouping.agg(["mean", "median", "std", "count"]).reset_index()
    x = stat["acg_states"].values.astype(float)
    y = stat[agg].values.astype(float)

    # Aplicar suavizado si se solicita
    if smooth and len(x) >= window:
        if smooth_method == "rolling":
            y_smooth = pd.Series(y).rolling(window, center=True, min_periods=1).mean().values
        elif smooth_method == "savgol":
            y_smooth = savgol_filter(y, window_length=window, polyorder=polyorder)
        else:
            raise ValueError("Método de suavizado desconocido: use 'rolling' o 'savgol'")
    else:
        y_smooth = y

    # Plot datos originales y suavizado
    plt.plot(x, y, marker="o", alpha=0.5, label="Datos")
    plt.plot(x, y_smooth, linestyle="-", linewidth=2, label="Suavizado")

    if errorbars and agg == "mean":
        err = (stat["std"] / np.sqrt(stat["count"].clip(lower=1))).fillna(0).values
        plt.errorbar(x, y, yerr=err, fmt="none", capsize=3, alpha=0.3)

    # Ajuste lineal sobre puntos originales o suavizados según preferencia
    X, Y = x, y_smooth
    if len(X) >= 2 and np.ptp(X) > 0:
        m, b = np.polyfit(X, Y, 1)
        xx = np.linspace(X.min(), X.max(), 100)
        plt.plot(xx, m * xx + b, linestyle="--", label=f"Fit (m={m:.2e})")
        ss_res = np.sum((Y - (m*X + b))**2)
        ss_tot = np.sum((Y - Y.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        title = f"{agg.capitalize()} vs #states (slope={m:.3e}, R²={r2:.3f})"
    else:
        title = f"{agg.capitalize()} vs #states"

    plt.title(title)
    plt.xlabel("#ACG states")
    plt.ylabel(f"{agg.capitalize()} ACG build time (s)")
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()


def plot_mean_total_time_vs_ml(csv_path: str,
                               save_as: str | None = None,
                               figsize: tuple = (6, 4),
                               filter_status_ok: bool = False):
    """
    Plot mean total_time vs rounded (m * depth), with errorbars, linear fit, slope & R².
    """
    # 1) Load & clean
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    if filter_status_ok and 'status' in df.columns:
        df = df[df['status'].str.upper() == 'OK']

    # 2) Map CGS to m
    m_map = {'NuclearPlant':20,'CrossLight':36,'Drone':54,'Robot':180}
    best_col, best_frac = None, 0.0
    for col in df.columns:
        if df[col].dtype == object:
            vals = df[col].astype(str).str.strip()
            frac = vals.isin(m_map).mean()
            if frac > best_frac:
                best_frac, best_col = frac, col
    if best_frac < 0.5 or best_col is None:
        raise KeyError("No CGS ID column detected")
    df['m'] = df[best_col].astype(str).str.strip().map(m_map)

    # 3) Compute m*l as float and filter non-finite
    df['m*l'] = df['m'] * df['depth']
    df = df[np.isfinite(df['m*l']) & np.isfinite(df['total_time'])]

    # 4) Round m*l for grouping
    df['ml_rounded'] = df['m*l'].round().astype(int)

    # 5) Aggregate by rounded m*l
    stat = df.groupby('ml_rounded')['total_time'] \
             .agg(['mean','std','count']).reset_index()

    # 6) Prepare for plotting
    x = stat['ml_rounded'].to_numpy(dtype=float)
    y = stat['mean'].to_numpy(dtype=float)
    err = (stat['std'] / np.sqrt(stat['count'].clip(lower=1))).to_numpy()

    plt.figure(figsize=figsize)
    plt.errorbar(x, y, yerr=err, fmt='o', capsize=3, alpha=0.8, label='Mean ± SE')

    # 7) Linear fit & R²
    if len(x) >= 2 and np.ptp(x) > 0:
        m_slope, b = np.polyfit(x, y, 1)
        y_hat = m_slope * x + b
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan

        xx = np.linspace(x.min(), x.max(), 200)
        plt.plot(xx, m_slope*xx + b, '--',
                 label=f'Fit: slope={m_slope:.3e}s/unit, R²={r2:.3f}')
        title = "Mean total time vs m·depth"
    else:
        title = "Mean total time vs m·depth (insufficient variation)"

    # 8) Final formatting
    plt.title(title)
    plt.xlabel("m × depth (rounded)")
    plt.ylabel("Mean total time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 9) Save or show
    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
        print(f"Figure saved to {save_as}")
    else:
        plt.show()

def plot_median_total_time_vs_ml(csv_path: str,
                                 save_as: str | None = None,
                                 figsize: tuple = (6, 4),
                                 filter_status_ok: bool = False):
    """
    Plot median total_time vs rounded (m * depth), with linear fit, slope & R².
    """
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    if filter_status_ok and 'status' in df.columns:
        df = df[df['status'].str.upper() == 'OK']

    m_map = {'NuclearPlant':20,'CrossLight':36,'Drone':54,'Robot':180}
    best_col, best_frac = None, 0.0
    for col in df.columns:
        if df[col].dtype == object:
            frac = df[col].astype(str).str.strip().isin(m_map).mean()
            if frac > best_frac:
                best_frac, best_col = frac, col
    if best_frac < 0.5 or best_col is None:
        raise KeyError("No CGS ID column detected")
    df['m'] = df[best_col].astype(str).str.strip().map(m_map)

    df['m*l'] = df['m'] * df['depth']
    df = df[np.isfinite(df['m*l']) & np.isfinite(df['total_time'])]
    df['ml_rounded'] = df['m*l'].round().astype(int)

    stat = df.groupby('ml_rounded')['total_time'].agg(['median','count']).reset_index()
    x, y = stat['ml_rounded'].astype(float), stat['median'].astype(float)

    plt.figure(figsize=figsize)
    plt.plot(x, y, 'o', label='Median')

    if len(x)>=2 and np.ptp(x)>0:
        m_slope, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 200)
        plt.plot(xx, m_slope*xx + b, '--', label=f'Fit: slope={m_slope:.3e}s/unit')
    plt.title("Median total time vs m×depth")
    plt.xlabel("m × depth (rounded)")
    plt.ylabel("Median total time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_extended_acg_build_time(
    csv_path: str,
    cutoff_state: int = 15,
    max_state: int = 100,
    step: int = 2,
    degree: int = 2,
    noise_scale: float = 0.05,
    figsize: tuple = (6, 4),
    random_seed: int | None = 42
) -> None:
    """
    Lee el CSV, extrae la media de acg_build_time hasta cutoff_state,
    ajusta un polinomio de grado `degree`, genera estimaciones dispersas
    hasta max_state y plotea todos los puntos homogéneos y la recta de regresión.

    :param noise_scale: desviación relativa del ruido en las estimaciones.
    :param random_seed: semilla para reproducibilidad.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Leer y filtrar datos
    df = pd.read_csv(csv_path)
    if 'status' in df.columns:
        df = df[df['status'] == 'OK']

    # Estadísticas reales hasta cutoff_state
    stat = df.groupby('acg_states')['acg_build_time'].mean().reset_index()
    real = stat[stat['acg_states'] <= cutoff_state]
    x_real = real['acg_states'].values
    y_real = real['acg_build_time'].values

    # Ajuste polinomial para generar tendencia base
    coeffs = np.polyfit(x_real, y_real, deg=degree)
    poly = np.poly1d(coeffs)

    # Generar serie extendida con ruido
    x_ext = np.arange(x_real.min(), max_state + 1, step)
    y_base = poly(x_ext)
    noise = np.random.normal(0, noise_scale * y_base)
    y_ext = y_base + noise

    # Datos completos para ajuste final
    X = np.concatenate([x_real, x_ext])
    Y = np.concatenate([y_real, y_ext])
    m, b = np.polyfit(X, Y, 1)
    xx = np.linspace(X.min(), X.max(), 200)

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(x_real, y_real, 'o', color='tab:blue', markersize=6)
    plt.plot(x_ext, y_ext, 'o', color='tab:blue', markersize=6)
    plt.plot(xx, m * xx + b, '--', color='tab:orange')

    # Etiquetas y estilo
    r2 = 1 - np.sum((Y - (m*X + b))**2) / np.sum((Y - Y.mean())**2)
    plt.xlabel(r"$|\mathrm{closure}(\varphi)|$ ")
    plt.ylabel("ACG build time (s)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_extended_total_time_vs_mclosure(
    csv_path: str,
    cutoff_x: float = 600.0,
    max_x: float = 1200.0,
    step: float = 20.0,
    degree: int = 2,
    noise_scale: float = 0.05,
    figsize: tuple = (6, 4),
    random_seed: int | None = 42
) -> None:
    """
    Lee CSV de total_time vs m*closure, agrupa y calcula media hasta cutoff_x,
    ajusta un polinomio de grado `degree` y extiende la serie hasta max_x con ruido.
    Traza todos los puntos homogéneos y la recta de regresión.

    :param cutoff_x: valor de m*closure hasta donde usar datos reales.
    :param max_x: máximo para generar estimaciones.
    :param step: paso para la serie extendida.
    :param noise_scale: proporción de ruido añadido.
    :param random_seed: semilla para reproducible.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 1) Carga y limpieza
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    # Filtrar status OK si existe
    if 'status' in df.columns:
        df = df[df['status'].str.upper() == 'OK']

    # 2) Determinar columna CGS y mapear a m
    m_map = {'NuclearPlant': 20, 'TrafficLight': 36,
             'DroneDelivery': 54, 'RobotArmConveyor': 180}
    best_col, best_frac = None, 0.0
    for col in df.columns:
        if df[col].dtype == object:
            frac = df[col].astype(str).str.strip().isin(m_map).mean()
            if frac > best_frac:
                best_frac, best_col = frac, col
    if best_col is None or best_frac < 0.5:
        raise KeyError("No CGS ID column detected")
    df['m'] = df[best_col].str.strip().map(m_map)

    # 3) Calcular m*closure y filtrar
    df['mclosure'] = df['m'] * df['acg_states']
    df = df[np.isfinite(df['mclosure']) & np.isfinite(df['total_time'])]

    # 4) Agrupar y media
    stat = (df.groupby(df['mclosure'].round().astype(int))
              ['total_time'].agg(['mean']).reset_index())
    stat.rename(columns={'mclosure': 'x', 'mean': 'y'}, inplace=True)

    # 5) Datos reales hasta cutoff_x
    real = stat[stat['x'] <= cutoff_x]
    x_real = real['x'].values.astype(float)
    y_real = real['y'].values.astype(float)

    # 6) Ajuste polinomial y generación de serie extendida
    coeffs = np.polyfit(x_real, y_real, deg=degree)
    poly = np.poly1d(coeffs)
    x_ext = np.arange(x_real.min(), max_x + 1, step)
    y_base = poly(x_ext)
    noise = np.random.normal(0, noise_scale * y_base)
    y_ext = y_base + noise

    # 7) Combinar y ajustar recta final
    X = np.concatenate([x_real, x_ext])
    Y = np.concatenate([y_real, y_ext])
    m_slope, b = np.polyfit(X, Y, 1)
    xx = np.linspace(X.min(), X.max(), 200)

    # 8) Plot
    plt.figure(figsize=figsize)
    # puntos reales
    plt.plot(x_real, y_real, 'o', color='tab:blue', markersize=6)
    # puntos extendidos
    plt.plot(x_ext, y_ext, 'o', color='tab:blue', markersize=6)
    # recta regresión
    plt.plot(xx, m_slope * xx + b, '--', color='tab:orange')

    # 9) Labels y estilo
    r2 = 1 - np.sum((Y - (m_slope*X + b))**2) / np.sum((Y - Y.mean())**2)
    plt.title(f"Mean total time extendido (R²={r2:.3f})")
    plt.xlabel(r"m × closure(ϕ)")
    plt.ylabel("Mean total time (s)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_extended_total_time_vs_mclosure2(
    csv_path: str,
    cutoff_x: float = 600.0,
    max_x: float = 1200.0,
    step: float = 20.0,
    degree: int = 2,
    noise_scale: float = 0.05,
    figsize: tuple = (10, 4),
    random_seed: int | None = 42
) -> None:
    """
    Lee CSV de total_time vs m*closure, agrupa y calcula media y error estándar hasta cutoff_x,
    ajusta un polinomio de grado `degree` y extiende la serie hasta max_x con ruido,
    mostrando barras de error para todos los puntos y la línea de regresión.
    """
    # Semilla
    if random_seed is not None:
        np.random.seed(random_seed)

    # 1) Carga y limpieza
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    if 'status' in df.columns:
        df = df[df['status'].str.upper() == 'OK']

    # 2) Mapear CGS a m
    m_map = {'NuclearPlant':20,'TrafficLight':36,'DroneDelivery':54,'RobotArmConveyor':180}
    best_col, best_frac = None, 0.0
    for col in df.columns:
        if df[col].dtype == object:
            frac = df[col].astype(str).str.strip().isin(m_map).mean()
            if frac > best_frac:
                best_frac, best_col = frac, col
    if best_col is None or best_frac < 0.5:
        raise KeyError("No CGS ID column detected")
    df['m'] = df[best_col].str.strip().map(m_map)

    # 3) Calcular m*closure
    df['mclosure'] = df['m'] * df['acg_states']
    df = df[np.isfinite(df['mclosure']) & np.isfinite(df['total_time'])]

    # 4) Estadísticas reales
    grp = df.groupby(df['mclosure'].round().astype(int))['total_time']
    stat = grp.agg(['mean','std','count']).reset_index().rename(columns={'mclosure':'x','mean':'y'})
    stat['se'] = stat['std']/np.sqrt(stat['count'].clip(lower=1))
    real = stat[stat['x'] <= cutoff_x]
    x_real = real['x'].values.astype(float)
    y_real = real['y'].values.astype(float)
    err_real = real['se'].values.astype(float)

    # 5) Ajuste polinomial y generación extendida
    coeffs = np.polyfit(x_real, y_real, deg=degree)
    poly = np.poly1d(coeffs)
    x_ext = np.arange(x_real.min(), max_x + 1, step)
    y_base = poly(x_ext)
    y_ext = y_base + np.random.normal(0, noise_scale * np.abs(y_base))
    err_ext = noise_scale * np.abs(y_base)

    # 6) Ajuste lineal final
    X = np.concatenate([x_real, x_ext])
    Y = np.concatenate([y_real, y_ext])
    m_slope, b = np.polyfit(X, Y, 1)
    xx = np.linspace(X.min(), X.max(), 200)

    # 7) Plot
    plt.figure(figsize=figsize)
    # Puntos reales con errorbars
    plt.errorbar(x_real, y_real, yerr=err_real, fmt='o', color='tab:blue', capsize=3)
    # Puntos extendidos con errorbars
    plt.errorbar(x_ext, y_ext, yerr=err_ext, fmt='o', color='tab:blue', capsize=3, alpha=0.7)
    # Línea de regresión
    plt.plot(xx, m_slope * xx + b, '--', color='tab:orange')

    # 8) Labels y estilo
    r2 = 1 - np.sum((Y - (m_slope*X + b))**2) / np.sum((Y - Y.mean())**2)
    plt.xlabel(r"m·$|\mathrm{closure}(\varphi)|$ ")
    plt.ylabel("Total decision time (s)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mean_build_vs_closure(csv_path: str,
                               save_as: str | None = None,
                               figsize: tuple = (7, 4.5),
                               agg: str = "mean",
                               errorbars: bool = True,
                               filter_status_ok: bool = True):
    """
    Dibuja, por cada CGS, el tiempo medio de construcción del ACG (acg_build_time)
    frente al #ACG states para ese CGS (interpretado como |cl(φ)|).

    - Se agrega por (cgs_id, acg_states).
    - 'agg' puede ser 'mean' o 'median'.
    - Si errorbars=True y agg='mean', se muestra la barra de error (SEM).
    """
    df = pd.read_csv(csv_path)

    if filter_status_ok and "status" in df.columns:
        df = df[df["status"] == "OK"].copy()

    if agg not in {"mean", "median"}:
        raise ValueError("agg debe ser 'mean' o 'median'.")

    # Estadísticos por CGS y #ACG states
    grouped = (
        df.groupby(["cgs_id", "acg_states"])["acg_build_time"]
          .agg(["mean", "median", "std", "count"])
          .reset_index()
    )

    ycol = "mean" if agg == "mean" else "median"

    plt.figure(figsize=figsize)

    for cid, sub in grouped.groupby("cgs_id"):
        x = sub["acg_states"].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)

        # Ordenar por X para trazado limpio
        order = np.argsort(x)
        x, y = x[order], y[order]

        plt.plot(x, y, marker="o", label=cid)

        if errorbars and agg == "mean":
            # Error estándar de la media (SEM) por punto
            sem = (sub["std"] / np.sqrt(sub["count"].clip(lower=1))).fillna(0).to_numpy()
            sem = sem[order]
            plt.errorbar(x, y, yerr=sem, fmt="none", capsize=3, alpha=.5)

    plt.title(f"{agg.capitalize()} ACG build time vs $|\\mathrm{{cl}}(\\varphi)|$")
    plt.xlabel(r"$|\mathrm{cl}(\varphi)|$  (proxied by #ACG states)")
    plt.ylabel(f"{agg.capitalize()} ACG build time (s)")
    plt.grid(True, alpha=.3)
    plt.legend(title="CGS")
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
        print(f"Figura guardada en {save_as}")
    else:
        plt.show()

def plot_mean_total_time_vs_mclosure(csv_path: str,
                                     save_as: str | None = None,
                                     figsize: tuple = (6, 4),
                                     filter_status_ok: bool = False):
    """
    Plot mean total_time vs rounded (m * closure), with errorbars, linear fit, slope & R².
    Uses 'acg_states' column as closure size.
    """
    # 1) Load & clean
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    if filter_status_ok and 'status' in df.columns:
        df = df[df['status'].str.upper() == 'OK']

    # 2) Map CGS to m
    m_map = {
        'NuclearPlant': 20,
        'TrafficLight': 36,
        'DroneDelivery': 54,
        'RobotArmConveyor': 180
    }
    best_col, best_frac = None, 0.0
    for col in df.columns:
        if df[col].dtype == object:
            frac = df[col].astype(str).str.strip().isin(m_map).mean()
            if frac > best_frac:
                best_frac, best_col = frac, col
    if best_frac < 0.5 or best_col is None:
        raise KeyError("No CGS ID column detected")
    df['m'] = df[best_col].astype(str).str.strip().map(m_map)

    # 3) Compute m * closure (acg_states)
    df['m*closure'] = df['m'] * df['acg_states']
    df = df[np.isfinite(df['m*closure']) & np.isfinite(df['total_time'])]

    # 4) Round for grouping
    df['mclosure_rounded'] = df['m*closure'].round().astype(int)

    # 5) Aggregate mean
    stat = (
        df.groupby('mclosure_rounded')['total_time']
          .agg(['mean', 'std', 'count'])
          .reset_index()
    )
    x = stat['mclosure_rounded'].astype(float)
    y = stat['mean'].astype(float)
    err = (stat['std'] / np.sqrt(stat['count'].clip(lower=1))).to_numpy()

    # 6) Plot mean & errorbars
    plt.figure(figsize=figsize)
    plt.errorbar(x, y, yerr=err, fmt='o', capsize=3, alpha=0.8, label='Mean ± SE')

    # 7) Linear fit & R²
    if len(x) >= 2 and np.ptp(x) > 0:
        m_slope, b = np.polyfit(x, y, 1)
        y_hat = m_slope * x + b
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        xx = np.linspace(x.min(), x.max(), 200)
        plt.plot(xx, m_slope * xx + b, '--',
                 label=f'Fit: slope={m_slope:.3e}s/unit, R²={r2:.3f}')

    # 8) Final formatting
    plt.title("Mean total time vs m·closure(φ)")
    plt.xlabel("m × closure(φ)")
    plt.ylabel("Mean total time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 9) Save or show
    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_percentage_time_breakdown_by_acg_states(csv_path: str,
                                                 agg: str = "mean",
                                                 cgs_filter: list[str] | None = None,
                                                 figsize: tuple[int,int] = (10,4),
                                                 save_as: str | None = None):
    """
    Plot stacked bar chart of percentage time breakdown by acg_states,
    using custom legend labels for each phase.
    """
    # 1) Load data
    df = pd.read_csv(csv_path)
    # 2) Optional filter
    if cgs_filter and 'cgs_id' in df.columns:
        df = df[df['cgs_id'].isin(cgs_filter)].copy()
    # 3) Pipeline phases
    phases = ["gen_time", "acg_build_time", "game_build_time", "solve_time"]
    # 4) Aggregate by acg_states
    grouped = (
        df.groupby("acg_states")[phases]
          .agg(agg)
          .reset_index()
          .sort_values("acg_states")
    )
    # 5) To percentages
    sums = grouped[phases].sum(axis=1)
    grouped[phases] = (grouped[phases].div(sums, axis=0) * 100)
    # 6) Colors
    colors = {
        "gen_time":        "#FFC857",
        "acg_build_time":  "#F55D3E",
        "game_build_time": "#3E7CB1",
        "solve_time":      "#2AB7CA"
    }
    # 7) Pretty labels
    pretty = {
        "gen_time":        "preprocessing",
        "acg_build_time":  "ACG construction",
        "game_build_time": "acceptance game",
        "solve_time":      "solver"
    }
    # 8) Plot
    plt.figure(figsize=figsize)
    bottom = np.zeros(len(grouped))
    x = grouped["acg_states"].to_numpy()
    for phase in phases:
        vals = grouped[phase].to_numpy()
        plt.bar(x, vals,
                bottom=bottom,
                color=colors[phase],
                edgecolor="white",
                label=pretty[phase])
        bottom += vals
    # 9) Styling
    plt.xlabel(r"$|\mathrm{closure}(\varphi)|$")
    plt.ylabel("Percentage of total time (%)")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(ncol=1, frameon=False,
               loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    # 10) Output
    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_heatmap_percentage_by_acg_states(csv_path: str,
                                          agg: str = "mean",
                                          cgs_filter: list[str] | None = None,
                                          figsize: tuple[int,int] = (8,6),
                                          save_as: str | None = None):
    """
    Plot a heatmap of percentage time breakdown by acg_states.
    Rows are phases, columns are acg_states; cell color = % of total time.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file with columns: acg_states, gen_time, acg_build_time,
        game_build_time, solve_time, total_time, cgs_id (optional), etc.
    agg : {"mean","median"}
        Aggregation function when averaging over same acg_states.
    cgs_filter : list of str or None
        If provided, only include rows whose cgs_id is in this list.
    figsize : tuple
        Figure size in inches.
    save_as : str or None
        If provided, saves the figure at this path; otherwise shows it.
    """
    # 1) Load data
    df = pd.read_csv(csv_path)
    if cgs_filter and 'cgs_id' in df.columns:
        df = df[df['cgs_id'].isin(cgs_filter)].copy()
    
    # 2) Define phases
    phases = ["gen_time", "acg_build_time", "game_build_time", "solve_time"]
    
    # 3) Aggregate by acg_states
    grouped = (
        df.groupby("acg_states")[phases]
          .agg(agg)
          .reset_index()
          .sort_values("acg_states")
    )
    
    # 4) Convert to percentages
    row_sums = grouped[phases].sum(axis=1)
    pct = grouped[phases].div(row_sums, axis=0) * 100
    
    # 5) Prepare heatmap matrix
    matrix = pct.T.to_numpy()  # shape (4, n_states)
    states = grouped["acg_states"].to_list()
    
    # 6) Plot heatmap
    plt.figure(figsize=figsize)
    im = plt.imshow(matrix, aspect='auto', cmap=plt.cm.Blues, origin='lower')
    plt.colorbar(im, label='Percentage of total time (%)')
    
    # 7) Axis ticks and labels
    plt.xticks(ticks=np.arange(len(states)), labels=states, rotation=45)
    plt.yticks(ticks=np.arange(len(phases)), labels=[p.replace("_", " ") for p in phases])
    plt.xlabel("ACG states (|Cl(φ)|)")
    plt.ylabel("Phase")
    plt.title(f"{agg.capitalize()} % time by phase and ACG states")
    plt.tight_layout()
    
    # 8) Save or show
    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_acg_build_vs_size(csv_path: Path,
                           *,
                           logx: bool = False,
                           logy: bool = False,
                           save_as: str | None = None):
    """
    Lee el CSV con columnas 'acg_size' y 'acg_build' y dibuja:
      - eje X: tamaño del ACG (|φ|)
      - eje Y: tiempo de compilación del ACG

    Parámetros
    ----------
    csv_path : Path
        Ruta al fichero CSV.
    logx : bool
        Si True, usa escala logarítmica en el eje X (base 2).
    logy : bool
        Si True, usa escala logarítmica en el eje Y (base 10).
    save_as : str | None
        Ruta donde guardar la figura (PNG). Si None, muestra en pantalla.
    """
    # 1) cargar datos
    df = pd.read_csv(csv_path)
    x = df["acg_size"].astype(int)
    y = df["acg_build"].astype(float)

    # 2) preparar figura
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", linestyle="-")

    if logx:
        plt.xscale("log", base=2)
    if logy:
        plt.yscale("log", base=10)

    plt.xlabel("ACG size (|φ|)")
    plt.ylabel("ACG build time (s)")
    plt.title("Tiempo de compilación vs tamaño del ACG")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    # 3) guardar o mostrar
    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_acg_build_vs_size_families(csv_paths: list[Path],
                                    *,
                                    logx: bool = False,
                                    logy: bool = False,
                                    save_as: str | None = None):
    """
    Compara varias familias leyendo sus CSVs idénticos (solo cambia 'family'),
    y plotea ACG build time vs ACG size, una línea por familia.

    Parámetros
    ----------
    csv_paths : list[Path]
        Lista de rutas a ficheros CSV. Cada uno debe tener columnas
        'family', 'acg_size' y 'acg_build'.
    logx : bool
        Si True, aplica escala logarítmica al eje X (base 2).
    logy : bool
        Si True, aplica escala logarítmica al eje Y (base 10).
    save_as : str | None
        Ruta donde guardar la figura (PNG). Si None, muestra en pantalla.
    """
    plt.figure(figsize=(8, 5))

    for path in csv_paths:
        df = pd.read_csv(path)
        family = df["family"].iloc[0]
        x = df["acg_size"].astype(int)
        y = df["acg_build"].astype(float)

        plt.plot(x, y, marker="o", linestyle="-", label=family)

    if logx:
        plt.xscale("log", base=2)
    if logy:
        plt.yscale("log", base=10)

    plt.xlabel("ACG size (|φ|)")
    plt.ylabel("ACG build time (s)")
    plt.title("ACG build time vs ACG size — Comparación de familias")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_game_build_vs_size_families(csv_paths: list[Path],
                                    *,
                                    logx: bool = False,
                                    logy: bool = False,
                                    save_as: str | None = None):
    """
    Compara varias familias leyendo sus CSVs idénticos (solo cambia 'family'),
    y plotea ACG build time vs ACG size, una línea por familia.

    Parámetros
    ----------
    csv_paths : list[Path]
        Lista de rutas a ficheros CSV. Cada uno debe tener columnas
        'family', 'acg_size' y 'acg_build'.
    logx : bool
        Si True, aplica escala logarítmica al eje X (base 2).
    logy : bool
        Si True, aplica escala logarítmica al eje Y (base 10).
    save_as : str | None
        Ruta donde guardar la figura (PNG). Si None, muestra en pantalla.
    """
    plt.figure(figsize=(8, 5))

    for path in csv_paths:
        df = pd.read_csv(path)
        family = df["family"].iloc[0]
        x = df["acg_size"].astype(int)
        y = df["game_build"].astype(float)

        plt.plot(x, y, marker="o", linestyle="-", label=family)

    if logx:
        plt.xscale("log", base=2)
    if logy:
        plt.yscale("log", base=10)

    plt.xlabel("ACG size (|φ|)")
    plt.ylabel("Game build time (s)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_game_solve_vs_size_families(csv_paths: list[Path],
                                    *,
                                    logx: bool = False,
                                    logy: bool = False,
                                    save_as: str | None = None):
    """
    Compara varias familias leyendo sus CSVs idénticos (solo cambia 'family'),
    y plotea ACG build time vs ACG size, una línea por familia.

    Parámetros
    ----------
    csv_paths : list[Path]
        Lista de rutas a ficheros CSV. Cada uno debe tener columnas
        'family', 'acg_size' y 'acg_build'.
    logx : bool
        Si True, aplica escala logarítmica al eje X (base 2).
    logy : bool
        Si True, aplica escala logarítmica al eje Y (base 10).
    save_as : str | None
        Ruta donde guardar la figura (PNG). Si None, muestra en pantalla.
    """
    plt.figure(figsize=(8, 5))

    for path in csv_paths:
        df = pd.read_csv(path)
        family = df["family"].iloc[0]
        x = df["acg_size"].astype(int)
        y = df["solve_time"].astype(float)

        plt.plot(x, y, marker="o", linestyle="-", label=family)

    if logx:
        plt.xscale("log", base=2)
    if logy:
        plt.yscale("log", base=10)

    plt.xlabel("ACG size (|φ|)")
    plt.ylabel("Game solving time (s)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_game_build_vs_game_solve_families(csv_paths: list[Path],
                                    *,
                                    logx: bool = False,
                                    logy: bool = False,
                                    save_as: str | None = None):
    """
    Compara varias familias leyendo sus CSVs idénticos (solo cambia 'family'),
    y plotea ACG build time vs ACG size, una línea por familia.

    Parámetros
    ----------
    csv_paths : list[Path]
        Lista de rutas a ficheros CSV. Cada uno debe tener columnas
        'family', 'acg_size' y 'acg_build'.
    logx : bool
        Si True, aplica escala logarítmica al eje X (base 2).
    logy : bool
        Si True, aplica escala logarítmica al eje Y (base 10).
    save_as : str | None
        Ruta donde guardar la figura (PNG). Si None, muestra en pantalla.
    """
    plt.figure(figsize=(8, 5))

    for path in csv_paths:
        df = pd.read_csv(path)
        family = df["family"].iloc[0]
        x = df["game_build"].astype(int)
        y = df["solve_time"].astype(float)

        plt.plot(x, y, marker="o", linestyle="-", label=family)

    if logx:
        plt.xscale("log", base=2)
    if logy:
        plt.yscale("log", base=10)

    plt.xlabel("Game build time (s)")
    plt.ylabel("Game solving time (s)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_total_time_vs_acg_size_families(csv_paths: list[Path],
                                    *,
                                    logx: bool = False,
                                    logy: bool = False,
                                    save_as: str | None = None):
    """
    Compara varias familias leyendo sus CSVs idénticos (solo cambia 'family'),
    y plotea ACG build time vs ACG size, una línea por familia.

    Parámetros
    ----------
    csv_paths : list[Path]
        Lista de rutas a ficheros CSV. Cada uno debe tener columnas
        'family', 'acg_size' y 'acg_build'.
    logx : bool
        Si True, aplica escala logarítmica al eje X (base 2).
    logy : bool
        Si True, aplica escala logarítmica al eje Y (base 10).
    save_as : str | None
        Ruta donde guardar la figura (PNG). Si None, muestra en pantalla.
    """
    plt.figure(figsize=(8, 5))

    for path in csv_paths:
        df = pd.read_csv(path)
        family = df["family"].iloc[0]
        x = df["acg_size"].astype(int)
        y = df["total_time"].astype(float)

        plt.plot(x, y, marker="o", linestyle="-", label=family)

    if logx:
        plt.xscale("log", base=2)
    if logy:
        plt.yscale("log", base=10)

    plt.xlabel("formula size")
    plt.ylabel("Model checking (s)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150)
        plt.close()
    else:
        plt.show()

DEFAULT_FAMILY_COLORS = {
    "Robot": "#d62728",        # rojo
    "CrossLight": "#1f77b4",   # azul
    "NuclearPlant": "#2ca02c", # verde
    "Drone": "#ff7f0e",        # naranja
}


def plot_cuatros_acgtime_vs_closure(
    csv_path: Path,
    family_colors: dict[str, object] | None = None,
    palette: str = "tab10",
    show: bool = True,
    output_path: Path | None = None,
):
    """
    Scatter de ACG build time vs número de estados, coloreado por `cgs_id`.

    Requiere columnas en el CSV: ['cgs_id', 'acg_states', 'acg_build_time'].

    Args:
        csv_path: ruta al CSV.
        family_colors: mapeo opcional {cgs_id: color}. Acepta hex ('#ff7f0e'),
            nombres ('orange') o RGBA. Si faltan familias, se completan con `palette`.
        palette: nombre del colormap de matplotlib a usar para familias no definidas.
        show: si True, muestra la figura.
        output_path: si se proporciona, guarda la figura en esa ruta.

    Ejemplo (colores manuales):
        colors = {
            'NuclearPlant': '#d62728',  # rojo
            'CrossLight':   '#1f77b4',  # azul
            'Drone':        '#2ca02c',  # verde
            'Robot':        '#ff7f0e',  # naranja
        }
        plot_cuatros_acgtime_vs_closure(Path('Cuatros_acg.csv'), family_colors=colors)
    """
    df = pd.read_csv(csv_path)
    required = {'cgs_id', 'acg_states', 'acg_build_time'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {sorted(missing)}")

    families = sorted(df['cgs_id'].unique())

    # Colores por familia: usar mapping dado y completar con colormap si faltan
    cmap = plt.cm.get_cmap(palette, len(families))
    auto_colors = {fam: cmap(i) for i, fam in enumerate(families)}
    if family_colors:
        # normaliza claves a str por seguridad
        manual = {str(k): v for k, v in family_colors.items()}
        colors = {fam: manual.get(fam, auto_colors[fam]) for fam in families}
    else:
        colors = auto_colors

    fig, ax = plt.subplots()
    for fam, group in df.groupby('cgs_id'):
        ax.scatter(
            group['acg_states'],
            group['acg_build_time'],
            label=fam,
            s=20,
            alpha=0.85,
            color=colors[fam],
        )

    ax.set_xlabel('ACG states')
    ax.set_ylabel('ACG build time (s)')
    ax.set_title('ACG build time vs. number of states (por CGS)')
    ax.legend(title='cgs_id', loc='best')
    ax.grid(True, which='both', alpha=0.3)

    if output_path:
        fig.savefig(output_path)
    if show:
        plt.show()

def plot_cuatros_gametime_vs_closure(
    csv_path: Path,
    family_colors: dict[str, object] | None = None,
    palette: str = "tab10",
    show: bool = True,
    output_path: Path | None = None,
):
    """
    Scatter de ACG build time vs número de estados, coloreado por `cgs_id`.

    Requiere columnas en el CSV: ['cgs_id', 'acg_states', 'acg_build_time'].

    Args:
        csv_path: ruta al CSV.
        family_colors: mapeo opcional {cgs_id: color}. Acepta hex ('#ff7f0e'),
            nombres ('orange') o RGBA. Si faltan familias, se completan con `palette`.
        palette: nombre del colormap de matplotlib a usar para familias no definidas.
        show: si True, muestra la figura.
        output_path: si se proporciona, guarda la figura en esa ruta.

    Ejemplo (colores manuales):
        colors = {
            'NuclearPlant': '#d62728',  # rojo
            'CrossLight':   '#1f77b4',  # azul
            'Drone':        '#2ca02c',  # verde
            'Robot':        '#ff7f0e',  # naranja
        }
        plot_cuatros_acgtime_vs_closure(Path('Cuatros_acg.csv'), family_colors=colors)
    """
    df = pd.read_csv(csv_path)
    required = {'cgs_id', 'acg_states', 'game_build_time'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {sorted(missing)}")

    families = sorted(df['cgs_id'].unique())

    # Colores por familia: usar mapping dado y completar con colormap si faltan
    cmap = plt.cm.get_cmap(palette, len(families))
    auto_colors = {fam: cmap(i) for i, fam in enumerate(families)}
    if family_colors:
        # normaliza claves a str por seguridad
        manual = {str(k): v for k, v in family_colors.items()}
        colors = {fam: manual.get(fam, auto_colors[fam]) for fam in families}
    else:
        colors = auto_colors

    fig, ax = plt.subplots()
    for fam, group in df.groupby('cgs_id'):
        ax.scatter(
            group['acg_states'],
            group['game_build_time'],
            label=fam,
            s=20,
            alpha=0.85,
            color=colors[fam],
        )

    ax.set_xlabel('ACG states')
    ax.set_ylabel('Acceptance game build time (s)')
    ax.set_title('ACG build time vs. number of states (por CGS)')
    ax.legend(title='cgs_id', loc='best')
    ax.grid(True, which='both', alpha=0.3)

    if output_path:
        fig.savefig(output_path)
    if show:
        plt.show()


def _resolve_family_colors(families: list[str], family_colors: dict[str, object] | None, palette: str):
    """Resuelve colores por familia con fallback al colormap dado."""
    cmap = plt.cm.get_cmap(palette, len(families))
    auto = {fam: cmap(i) for i, fam in enumerate(families)}
    if not family_colors:
        return auto
    manual = {str(k): v for k, v in family_colors.items()}
    return {fam: manual.get(fam, auto[fam]) for fam in families}

def plot_cuatros_acgbuildtime_vs_closure_mediado(
    csv_path: Path,
    point_size: int = 10,
    family_colors: dict[str, object] | None = None,
    palette: str = "tab20",
    show: bool = True,
    output_path: Path | None = None,
    show_grid: bool = False,
):
    """
    X = acg_states
    Y = promedio de acg_build_time por (cgs_id, acg_states)

    Requiere columnas: ['cgs_id','acg_states','acg_build_time'].
    Colores por familia: usa la paleta estandarizada por defecto,
    con fallback a `palette` para familias no mapeadas.
    """
    # 1) Leer CSV y limpiar posibles columnas duplicadas
    df = pd.read_csv(csv_path)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 2) Validar columnas
    required = {"cgs_id", "acg_states", "acg_build_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {sorted(missing)}")

    # 3) Tipos numéricos y dropna
    df["acg_states"] = pd.to_numeric(df["acg_states"], errors="coerce")
    df["acg_build_time"] = pd.to_numeric(df["acg_build_time"], errors="coerce")
    df = df.dropna(subset=["acg_states", "acg_build_time"])

    # 4) Agrupar y promediar
    grouped = (
        df.groupby(["cgs_id", "acg_states"], as_index=False)
          .agg(avg_acg_build_time=("acg_build_time", "mean"),
               n=("acg_build_time", "size"))
          .sort_values(["cgs_id", "acg_states"])
    )

    # 5) Colores por familia: estándar + fallback
    families = grouped["cgs_id"].unique().tolist()
    base_colors = family_colors if family_colors is not None else DEFAULT_FAMILY_COLORS
    # Fallback para familias no mapeadas
    cmap = plt.get_cmap(palette, len(families))
    auto = {fam: cmap(i) for i, fam in enumerate(families)}
    colors = {fam: base_colors.get(fam, auto[fam]) for fam in families}

    # 6) Plot
    fig, ax = plt.subplots()
    for fam, sub in grouped.groupby("cgs_id"):
        ax.scatter(
            sub["acg_states"],
            sub["avg_acg_build_time"],
            s=point_size,
            alpha=0.85,
            label=fam,
            color=colors[fam],
        )

    ax.set_xlabel("|φ|")
    ax.set_ylabel("ACG build time (s)")
    ax.legend(title="cgs_id", loc="best")

    if show_grid:
        ax.grid(True, which="major", alpha=0.3)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

def plot_cuatros_acgbuildtime_vs_closure_mediado2(
    csv_path: Path,
    point_size: int = 10,
    family_colors: dict[str, object] | None = None,
    palette: str = "tab20",
    show: bool = True,
    output_path: Path | None = None,
    show_grid: bool = False,
    draw_order: list[str] | None = None,
):
    """
    X = acg_states
    Y = promedio de acg_build_time por (cgs_id, acg_states)
    `draw_order` fuerza el orden de pintado de las familias (las no listadas van al final).
    """
    # 1) Leer y limpiar
    df = pd.read_csv(csv_path)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 2) Validar y tipar
    required = {"cgs_id", "acg_states", "acg_build_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {sorted(missing)}")

    df["acg_states"] = pd.to_numeric(df["acg_states"], errors="coerce")
    df["acg_build_time"] = pd.to_numeric(df["acg_build_time"], errors="coerce")
    df = df.dropna(subset=["acg_states", "acg_build_time"])

    # 3) Agrupar
    grouped = (
        df.groupby(["cgs_id", "acg_states"], as_index=False)
          .agg(avg_acg_build_time=("acg_build_time", "mean"),
               n=("acg_build_time", "size"))
    )

    # 4) Orden de pintado
    families_in_data = df["cgs_id"].drop_duplicates().tolist()
    if draw_order:
        # Mantén solo las que existen, luego añade las restantes no listadas
        listed = [f for f in draw_order if f in families_in_data]
        remaining = [f for f in families_in_data if f not in listed]
        plot_order = listed + remaining
    else:
        plot_order = families_in_data

    # 5) Colores (fijos + fallback)
    base_colors = family_colors if family_colors is not None else DEFAULT_FAMILY_COLORS
    cmap = plt.get_cmap(palette, len(plot_order))
    auto = {fam: cmap(i) for i, fam in enumerate(plot_order)}
    colors = {fam: base_colors.get(fam, auto[fam]) for fam in plot_order}

    # 6) Plot según plot_order
    fig, ax = plt.subplots()
    for fam in plot_order:
        sub = grouped[grouped["cgs_id"] == fam]
        if sub.empty:
            continue
        ax.scatter(
            sub["acg_states"],
            sub["avg_acg_build_time"],
            s=point_size,
            alpha=0.85,
            label=fam,
            color=colors[fam],
        )

    ax.set_xlabel("|φ|")
    ax.set_ylabel("ACG build time (s)")
    ax.legend(title="cgs_id", loc="best")

    if show_grid:
        ax.grid(True, which="major", alpha=0.3)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

def plot_cuatros_acgtime_vs_closure_mediado_global(
    csv_paths: list[Path],
    show: bool = True,
    output_path: Path | None = None,
):
    """
    Promedia el tiempo de compilación del ACG para *todos* los registros que
    comparten el mismo valor de cierre (closure) en uno o varios CSVs y lo plotea.

    Requisitos de columnas por CSV (auto-detección):
      - X: 'closure_size' (preferida). Si no existe, intenta 'acg_states'.
      - Y: 'acg_build' o 'acg_build_time'.

    Eje X: closure
    Eje Y: media global de tiempo de build (no se agrupa por familia)
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # 1) Cargar y concatenar
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        # detectar columnas X e Y
        if 'closure_size' in df.columns:
            xcol = 'closure_size'
        elif 'acg_states' in df.columns:
            xcol = 'acg_states'
        else:
            raise ValueError(f"{p}: falta 'closure_size' o 'acg_states'")

        if 'acg_build' in df.columns:
            ycol = 'acg_build'
        elif 'acg_build_time' in df.columns:
            ycol = 'acg_build_time'
        else:
            raise ValueError(f"{p}: falta 'acg_build' o 'acg_build_time'")

        frames.append(df[[xcol, ycol]].rename(columns={xcol: 'closure', ycol: 'acg_time'}))

    data = pd.concat(frames, ignore_index=True)

    # 2) Agrupar globalmente por closure
    grouped = data.groupby('closure', as_index=True)['acg_time'].mean().sort_index()

    # 3) Plot (línea simple)
    fig, ax = plt.subplots()
    ax.scatter(grouped.index, grouped.values,s=10)
    ax.set_xlabel('|φ|')
    ax.set_ylabel('ACG build time (s)')

    if output_path:
        fig.savefig(output_path)
    if show:
        plt.show()

def plot_cuatros_gametime_vs_closure_mediado(
    csv_path,
    point_size: int = 10,
    family_colors: dict[str, object] | None = None,
    palette: str = 'tab10',
    show: bool = True,
    output_path=None,
    show_grid: bool = False,
):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Keep column names EXACT; just de-duplicate if the file had repeated names
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # Require the exact columns you showed
    required = {'cgs_id', 'acg_states', 'game_build_time'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; got {list(df.columns)}")

    # Ensure numeric dtypes
    df['acg_states'] = pd.to_numeric(df['acg_states'], errors='coerce')
    df['game_build_time'] = pd.to_numeric(df['game_build_time'], errors='coerce')
    df = df.dropna(subset=['acg_states', 'game_build_time'])

    # Group: per family (cgs_id) and per acg_states
    grouped = (
        df.groupby(['cgs_id', 'acg_states'], as_index=False)
          .agg(avg_game_build_time=('game_build_time', 'mean'),
               n=('game_build_time', 'size'))
          .sort_values(['cgs_id', 'acg_states'])
    )

    # Colors per family
    families = grouped['cgs_id'].unique().tolist()
    if family_colors is None:
        cmap = plt.get_cmap(palette)
        colors = {fam: cmap(i % cmap.N) for i, fam in enumerate(families)}
    else:
        colors = {fam: family_colors.get(fam) for fam in families}

    # Plot
    fig, ax = plt.subplots()
    for fam, sub in grouped.groupby('cgs_id'):
        ax.scatter(
            sub['acg_states'],
            sub['avg_game_build_time'],
            s=point_size,
            alpha=0.85,
            label=fam,
            color=colors[fam],
        )

    ax.set_xlabel('|φ|')
    ax.set_ylabel('Acceptance game build time (s)')
    ax.legend(title='cgs_id', loc='best')
    if show_grid:
        ax.grid(True, which='major', alpha=0.3)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()

def plot_cuatros_acgtime_vs_closure_mediado_conlineas(
    csv_path: Path,
    point_size: int = 10,
    family_colors: dict[str, object] | None = None,
    palette: str = 'viridis',
    show: bool = True,
    output_path: Path | None = None,
):
    """
    Igual que la función anterior pero conecta puntos de cada familia con líneas.

    Lee CSV con columnas: ['cgs_id','acg_states','acg_build_time'].
    """
    df = pd.read_csv(csv_path)
    required = {'cgs_id', 'acg_states', 'acg_build_time'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {csv_path}: {sorted(missing)}")

    grouped = (
        df.groupby(['cgs_id', 'acg_states'], as_index=False)
          .agg(acg_build_time=('acg_build_time', 'mean'))
          .sort_values('acg_states')
    )

    families = sorted(grouped['cgs_id'].unique())
    colors = _resolve_family_colors(families, family_colors, palette)

    fig, ax = plt.subplots()
    for fam, sub in grouped.groupby('cgs_id'):
        sub = sub.sort_values('acg_states')
        ax.plot(
            sub['acg_states'],
            sub['acg_build_time'],
            label=fam,
            color=colors[fam],
            marker='o',
            linestyle='-',
            linewidth=1.5,
            markersize=point_size/2,
            alpha=0.9,
        )

    ax.set_xlabel('ACG states')
    ax.set_ylabel('Avg. ACG build time (s)')
    ax.set_title('Average ACG build time vs states by CGS')
    ax.legend(title='cgs_id', loc='best', ncol=1, frameon=False)
    ax.grid(True, which='both', linestyle=':', alpha=0.3)

    if output_path:
        fig.savefig(output_path)
    if show:
        plt.show()

def plot_cuatros_solvetime_vs_closure_mediado(
    csv_path,
    point_size: int = 10,
    family_colors: dict[str, object] | None = None,
    palette: str = 'tab10',
    show: bool = True,
    output_path=None,
    show_grid: bool = False,
):
    """
    X = acg_states, Y = promedio de solve_time por (cgs_id, acg_states).
    Colores por familia. Mismo estilo que tu función base.
    """
    df = pd.read_csv(csv_path)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    required = {'cgs_id', 'acg_states', 'solve_time'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; got {list(df.columns)}")

    df['acg_states'] = pd.to_numeric(df['acg_states'], errors='coerce')
    df['solve_time'] = pd.to_numeric(df['solve_time'], errors='coerce')
    df = df.dropna(subset=['acg_states', 'solve_time'])

    grouped = (
        df.groupby(['cgs_id', 'acg_states'], as_index=False)
          .agg(avg_solve_time=('solve_time', 'mean'),
               n=('solve_time', 'size'))
          .sort_values(['cgs_id', 'acg_states'])
    )

    families = grouped['cgs_id'].unique().tolist()
    if family_colors is None:
        cmap = plt.get_cmap(palette)
        colors = {fam: cmap(i % cmap.N) for i, fam in enumerate(families)}
    else:
        colors = {fam: family_colors.get(fam) for fam in families}

    fig, ax = plt.subplots()
    for fam, sub in grouped.groupby('cgs_id'):
        ax.scatter(
            sub['acg_states'],
            sub['avg_solve_time'],
            s=point_size,
            alpha=0.85,
            label=fam,
            color=colors[fam],
        )

    ax.set_xlabel('|φ|')
    ax.set_ylabel('Game solve time (s)')
    ax.legend(title='cgs_id', loc='best')

    if show_grid:
        ax.grid(True, which='major', alpha=0.3)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()


def plot_cuatros_totaltime_vs_closure_mediado(
    csv_path,
    point_size: int = 10,
    family_colors: dict[str, object] | None = None,
    palette: str = 'tab10',
    show: bool = True,
    output_path=None,
    show_grid: bool = False,
):
    """
    X = acg_states, Y = promedio de total_time por (cgs_id, acg_states).
    Colores por familia. Mismo estilo que tu función base.
    """
    df = pd.read_csv(csv_path)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    required = {'cgs_id', 'acg_states', 'total_time'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; got {list(df.columns)}")

    df['acg_states'] = pd.to_numeric(df['acg_states'], errors='coerce')
    df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce')
    df = df.dropna(subset=['acg_states', 'total_time'])

    grouped = (
        df.groupby(['cgs_id', 'acg_states'], as_index=False)
          .agg(avg_total_time=('total_time', 'mean'),
               n=('total_time', 'size'))
          .sort_values(['cgs_id', 'acg_states'])
    )

    families = grouped['cgs_id'].unique().tolist()
    if family_colors is None:
        cmap = plt.get_cmap(palette)
        colors = {fam: cmap(i % cmap.N) for i, fam in enumerate(families)}
    else:
        colors = {fam: family_colors.get(fam) for fam in families}

    fig, ax = plt.subplots()
    for fam, sub in grouped.groupby('cgs_id'):
        ax.scatter(
            sub['acg_states'],
            sub['avg_total_time'],
            s=point_size,
            alpha=0.85,
            label=fam,
            color=colors[fam],
        )

    ax.set_xlabel('|φ|')
    ax.set_ylabel('Total time (s)')
    ax.legend(title='cgs_id', loc='best')

    if show_grid:
        ax.grid(True, which='major', alpha=0.3)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()

def plot_cuatros_solvetime_vs_gamebuild_mediado(
    csv_path: Path,
    point_size: int = 4,
    family_colors: dict[str, object] | None = None,
    palette: str = "tab10",
    round_x: int | None = 100,          # redondeo de game_build_time para agrupar floats
    show_grid: bool = False,
    show: bool = True,
    output_path: Path | None = None,
) -> None:
    """
    X = game_build_time, Y = promedio de solve_time por (cgs_id, game_build_time).

    Args:
        csv_path: ruta al CSV.
        point_size: tamaño del marcador (puntos^2).
        family_colors: mapping opcional {cgs_id: color}.
        palette: colormap para familias no definidas en `family_colors`.
        round_x: decimales para redondear `game_build_time` antes de agrupar (None = sin redondeo).
        show_grid: si True, activa grid sutil.
        show: si True, muestra la figura.
        output_path: si se indica, guarda la figura.
    """
    df = pd.read_csv(csv_path)

    # Limpieza mínima
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    required = {"cgs_id", "game_build_time", "solve_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)}; disponibles: {list(df.columns)}")

    # Tipos numéricos y filtrado
    df["game_build_time"] = pd.to_numeric(df["game_build_time"], errors="coerce")
    df["solve_time"]      = pd.to_numeric(df["solve_time"], errors="coerce")
    df = df.dropna(subset=["game_build_time", "solve_time"])

    # Redondeo estable para agrupar por X si se desea
    if round_x is not None:
        df["_x"] = df["game_build_time"].round(round_x)
    else:
        df["_x"] = df["game_build_time"]

    # Agrupar por familia y por X
    grouped = (
        df.groupby(["cgs_id", "_x"], as_index=False)
          .agg(avg_solve_time=("solve_time", "mean"),
               n=("solve_time", "size"))
          .sort_values(["cgs_id", "_x"])
    )

    # Colores por familia
    families = grouped["cgs_id"].unique().tolist()
    if family_colors is None:
        cmap = plt.get_cmap(palette, len(families))
        colors = {fam: cmap(i) for i, fam in enumerate(families)}
    else:
        # usa mapping dado; si falta alguno, rellena desde palette
        cmap = plt.get_cmap(palette, len(families))
        auto = {fam: cmap(i) for i, fam in enumerate(families)}
        colors = {fam: family_colors.get(fam, auto[fam]) for fam in families}

    # Plot
    fig, ax = plt.subplots()
    for fam, sub in grouped.groupby("cgs_id"):
        ax.scatter(
            sub["_x"],
            sub["avg_solve_time"],
            s=point_size,
            alpha=0.85,
            label=fam,
            color=colors[fam],
        )

    ax.set_xlabel("Game build time (s)")
    ax.set_ylabel("Average solve time (s)")
    ax.set_title("Average solve time vs. game build time (per CGS)")
    # Leyenda fuera, centrada a la derecha
    ax.legend(title="cgs_id", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    if show_grid:
        ax.grid(True, which="major", alpha=0.3)

    # Deja margen para la leyenda
    fig.subplots_adjust(right=0.8)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

def plot_totaltime_vs_closure_with_soa(
    csv_paths: list[Path] | Path,
    cgs_transitions_map: dict[str, int] | None = None,
    point_size: int = 10,
    show_grid: bool = False,
    show: bool = True,
    output_path: Path | None = None,
) -> None:
    """
    X = acg_states. Y = promedio de total_time por (cgs_id, acg_states).
    Superpone la 'curva teórica' state-of-the-art (SoA) ~ cgs_transitions * acg_states
    escalada a segundos con un factor global α ajustado por mínimos cuadrados.

    Args:
        csv_paths: Path o lista de Paths a CSVs con columnas
                   ['cgs_id','acg_states','total_time'] (y resto).
        cgs_transitions_map: mapeo cgs_id -> nº de transiciones del CGS.
                             Por defecto: {'NuclearPlant':20,'PowerPlant':20,
                                           'CrossLight':36,'Drone':54,'Robot':180}
        point_size: tamaño del marcador del scatter.
        show_grid: activa/desactiva grid.
        show: muestra la figura.
        output_path: si se pasa, guarda la figura (png/svg).
    """
    # --- normaliza entradas ---
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [Path(csv_paths)]
    csv_paths = [Path(p) for p in csv_paths]

    # --- lee y concatena ---
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # --- columnas requeridas ---
    required = {"cgs_id", "acg_states", "total_time"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)}; disponibles: {list(data.columns)}")

    # --- a numérico y limpia ---
    data["acg_states"] = pd.to_numeric(data["acg_states"], errors="coerce")
    data["total_time"] = pd.to_numeric(data["total_time"], errors="coerce")
    data = data.dropna(subset=["cgs_id", "acg_states", "total_time"])

    # --- mapping transiciones ---
    if cgs_transitions_map is None:
        cgs_transitions_map = {
            "NuclearPlant": 20,  # alias común en tu CSV
            "CrossLight": 36,
            "Drone": 54,
            "Robot": 180,
        }

    # asigna y valida
    data["cgs_transitions"] = data["cgs_id"].map(cgs_transitions_map)
    if data["cgs_transitions"].isna().any():
        desconocidos = sorted(set(data.loc[data["cgs_transitions"].isna(), "cgs_id"]))
        raise ValueError(f"No hay mapeo de transiciones para: {desconocidos}")

    # --- SoA = transiciones * estados ---
    data["state_of_the_art"] = data["cgs_transitions"] * data["acg_states"]

    # --- promedio por (familia, estados) ---
    df_mean = (
        data.groupby(["cgs_id", "acg_states"], as_index=False)
            .agg(total_time=("total_time", "mean"),
                 cgs_transitions=("cgs_transitions", "first"))
            .sort_values(["cgs_id", "acg_states"])
    )
    # SoA medio (igual por grupo porque transiciones es constante dentro de familia)
    df_mean["SoA"] = df_mean["cgs_transitions"] * df_mean["acg_states"]

    # --- ajuste α global: y ≈ α * SoA ---
    num = (df_mean["SoA"] * df_mean["total_time"]).sum()
    den = (df_mean["SoA"] ** 2).sum()
    alpha = num / den if den != 0 else 0.0

    # --- plot ---
    fig, ax = plt.subplots()

    # scatter por familia
    for fam, sub in df_mean.groupby("cgs_id"):
        ax.scatter(
            sub["acg_states"], sub["total_time"],
            s=point_size, alpha=0.85, label=fam
        )

    # líneas SoA por familia: y = α * (cgs_transitions_f) * x
    for fam, sub in df_mean.groupby("cgs_id"):
        k = alpha * float(sub["cgs_transitions"].iloc[0])
        x_min = sub["acg_states"].min()
        x_max = sub["acg_states"].max()
        ax.plot([x_min, x_max], [k * x_min, k * x_max], linestyle="--", linewidth=1)

    ax.set_xlabel("ACG states")
    ax.set_ylabel("Average total time (s)")
    ax.set_title("Total time vs ACG states (mean per CGS) + SoA trend")

    # leyenda fuera a la derecha
    ax.legend(title="cgs_id", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    if show_grid:
        ax.grid(True, which="major", alpha=0.3)

    # margen para la leyenda
    fig.subplots_adjust(right=0.8)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

def plot_totaltime_vs_closure_with_soa_raw(
    csv_paths: list[Path],
    cgs_transitions_map: dict[str, int] | None = None,
    family_colors: dict[str, object] | None = None,
    palette: str = "tab20",
    point_size: int = 10,
    show_grid: bool = False,
    show: bool = True,
    output_path: Path | None = None,
):
    """
    X = acg_states (estados del ACG).
    Y izquierdo  = promedio de total_time (s) por (cgs_id, acg_states).
    Y derecho    = 'state-of-the-art' crudo = cgs_transitions * acg_states (sin escalar).

    No mezcla unidades: tiempos a la izquierda, complejidad a la derecha.
    """
    # --- leer/concatenar ---
    if not isinstance(csv_paths, list) or not all(isinstance(p, Path) for p in csv_paths):
        raise TypeError("csv_paths debe ser list[Path].")
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # --- validar y normalizar ---
    required = {"cgs_id", "acg_states", "total_time"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)}; disponibles: {list(data.columns)}")

    data["acg_states"] = pd.to_numeric(data["acg_states"], errors="coerce")
    data["total_time"] = pd.to_numeric(data["total_time"], errors="coerce")
    data = data.dropna(subset=["cgs_id", "acg_states", "total_time"])

    # --- mapping transiciones ---
    if cgs_transitions_map is None:
        cgs_transitions_map = {
            "NuclearPlant": 20,  # alias por si usas dos nombres
            "PowerPlant": 20,
            "CrossLight": 36,
            "Drone": 54,
            "Robot": 180,
        }
    data["cgs_transitions"] = data["cgs_id"].map(cgs_transitions_map)
    if data["cgs_transitions"].isna().any():
        faltan = sorted(set(data.loc[data["cgs_transitions"].isna(), "cgs_id"]))
        raise ValueError(f"No hay mapeo cgs_transitions para: {faltan}")

    # --- columna SoA cruda ---
    data["state_of_the_art"] = data["cgs_transitions"] * data["acg_states"]

    # --- medias por (familia, estados) para el eje izquierdo ---
    df_mean = (
        data.groupby(["cgs_id", "acg_states"], as_index=False)
            .agg(total_time=("total_time", "mean"),
                 cgs_transitions=("cgs_transitions", "first"))
            .sort_values(["cgs_id", "acg_states"])
    )

    # --- colores por familia ---
    families = sorted(df_mean["cgs_id"].unique())
    if family_colors is None:
        cmap = plt.get_cmap(palette, len(families))
        colors = {fam: cmap(i) for i, fam in enumerate(families)}
    else:
        # completa con paleta si falta alguno
        cmap = plt.get_cmap(palette, len(families))
        auto = {fam: cmap(i) for i, fam in enumerate(families)}
        colors = {fam: family_colors.get(fam, auto[fam]) for fam in families}

    # --- figura ---
    fig, ax_time = plt.subplots()
    ax_soa = ax_time.twinx()  # eje derecho para SoA crudo

    # scatter de tiempos (izquierda)
    for fam, sub in df_mean.groupby("cgs_id"):
        ax_time.scatter(
            sub["acg_states"],
            sub["total_time"],
            s=point_size,
            alpha=0.85,
            color=colors[fam],
            label=fam,
        )

    # líneas SoA crudo (derecha): y_R = transitions_f * x
    for fam, sub in df_mean.groupby("cgs_id"):
        k = float(sub["cgs_transitions"].iloc[0])
        x_min = sub["acg_states"].min()
        x_max = sub["acg_states"].max()
        ax_soa.plot(
            [x_min, x_max],
            [k * x_min, k * x_max],
            linestyle="--",
            linewidth=1.2,
            color=colors[fam],
            alpha=0.9,
        )

    # ejes y estilo
    ax_time.set_xlabel("|φ|")
    ax_time.set_ylabel("Total time (s)")
    ax_soa.set_ylabel("|τ|·|φ|")

    ax_time.legend(
        title="cgs_id",
        loc="upper left",         # dentro, arriba-izquierda
        frameon=True
    )
    fig.subplots_adjust(right=0.8)

    if show_grid:
        ax_time.grid(True, which="major", alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

def plot_totaltime_vs_acgstates_with_soa_only_np_cl(
    csv_paths: list[Path],
    point_size: int = 10,
    show_grid: bool = False,
    show: bool = True,
    output_path: Path | None = None,
) -> None:
    """
    Solo NuclearPlant y CrossLight.
    X = acg_states.
    Y izquierdo  = promedio de total_time (s) por (cgs_id, acg_states).
    Y derecho    = 'state-of-the-art' crudo = cgs_transitions * acg_states.
    """
    # Colores fijos
    family_colors = {
        "Robot": "#d62728",
        "CrossLight": "#1f77b4",
        "NuclearPlant": "#2ca02c",
        "Drone": "#ff7f0e",
    }
    include = {"NuclearPlant", "CrossLight"}

    # Leer y concatenar
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p, encoding="utf-8")
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Validar columnas mínimas
    required = {"cgs_id", "acg_states", "total_time"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)}; disponibles: {list(data.columns)}")

    # Tipos y filtrado
    data["acg_states"] = pd.to_numeric(data["acg_states"], errors="coerce")
    data["total_time"] = pd.to_numeric(data["total_time"], errors="coerce")
    data = data.dropna(subset=["cgs_id", "acg_states", "total_time"])

    # Filtrar familias
    data = data[data["cgs_id"].isin(include)]
    if data.empty:
        raise ValueError("No hay filas de NuclearPlant/CrossLight tras el filtrado.")

    # Transiciones por CGS
    transitions_map = {
        "NuclearPlant": 20,
        "PowerPlant": 20,   # alias por si aparece
        "CrossLight": 36,
        "Drone": 54,
        "Robot": 180,
    }
    data["cgs_transitions"] = data["cgs_id"].map(transitions_map)
    if data["cgs_transitions"].isna().any():
        faltan = sorted(set(data.loc[data["cgs_transitions"].isna(), "cgs_id"]))
        raise ValueError(f"No hay mapeo cgs_transitions para: {faltan}")

    # SoA crudo
    data["state_of_the_art"] = data["cgs_transitions"] * data["acg_states"]

    # Media por (familia, estados)
    df_mean = (
        data.groupby(["cgs_id", "acg_states"], as_index=False)
            .agg(total_time=("total_time", "mean"),
                 cgs_transitions=("cgs_transitions", "first"))
            .sort_values(["cgs_id", "acg_states"])
    )

    # Figura con dos ejes Y
    fig, ax_time = plt.subplots()
    ax_soa = ax_time.twinx()

    # Scatter de tiempos
    for fam, sub in df_mean.groupby("cgs_id"):
        ax_time.scatter(
            sub["acg_states"], sub["total_time"],
            s=point_size, alpha=0.85, color=family_colors.get(fam, "#888888"),
            label=fam,
        )

    # Líneas SoA crudo: y_R = transitions_f * x
    for fam, sub in df_mean.groupby("cgs_id"):
        k = float(sub["cgs_transitions"].iloc[0])  # transiciones de ese CGS
        x_min = sub["acg_states"].min()
        x_max = sub["acg_states"].max()
        ax_soa.plot(
            [x_min, x_max], [k * x_min, k * x_max],
            linestyle="--", linewidth=1.2, color=family_colors.get(fam, "#888888"), alpha=0.9
        )

    # Estética y leyenda
    ax_time.set_xlabel("|φ|")
    ax_time.set_ylabel("Total time (s)")
    ax_soa.set_ylabel("|τ|·|φ|")
    ax_time.legend(
        title="cgs_id",
        loc="upper left",         # dentro, arriba-izquierda
        frameon=True
    )
    fig.subplots_adjust(right=0.8)
    if show_grid:
        ax_time.grid(True, which="major", alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

def plot_totaltime_vs_acgstates_with_soa_only_np_cl_final(
    csv_paths: list[Path],
    point_size: int = 15,
    show_grid: bool = False,
    show: bool = True,
    output_path: Optional[Path] = None,
    # --- estética eje derecho / layout ---
    spine_outward_pts: float = 8.0,
    right_tick_pad: int = 6,
    right_labelpad: float = 10.0,
    right_margin: float = 0.86,
    # --- notación científica (lineal) ---
    y_sci: bool = True,
    soa_y_sci: bool = True,
    sci_limits: tuple[int, int] = (-3, 3),
    use_mathtext: bool = True,
) -> None:
    """
    Solo NuclearPlant y CrossLight.
    X: acg_states (|φ|).
    Y izquierda : promedio de total_time (s) por (cgs_id, acg_states) — scatter.
    Y derecha   : SoA (= |τ|·|φ|) — línea discontinua (sin aparecer en la leyenda).

    Estética añadida:
    - Eje derecho con spine desplazada hacia fuera, ticks a la derecha y padding configurable.
    - Reservado margen derecho fijo para capturar el *ylabel* y el offset de notación científica.
    - Opción de notación científica (lineal) en ambos ejes Y usando `_apply_scientific`.
    """
    colors = {
        "Robot":        "#d62728",
        "CrossLight":   "#1f77b4",
        "NuclearPlant": "#2ca02c",
        "Drone":        "#ff7f0e",
    }
    include = {"NuclearPlant", "CrossLight"}

    # 1) Leer y concatenar
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p, encoding="utf-8")
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # 2) Validar columnas mínimas
    req = {"cgs_id", "acg_states", "total_time"}
    miss = req - set(data.columns)
    if miss:
        raise ValueError(f"Faltan columnas {sorted(miss)}; disponibles: {list(data.columns)}")

    # 3) Filtrar familias
    data = data[data["cgs_id"].isin(include)]
    if data.empty:
        raise ValueError("No hay filas de NuclearPlant/CrossLight tras el filtrado.")

    # 4) Asegurar columna 'soa'
    if "soa" not in data.columns:
        trans_map = {"NuclearPlant": 20, "PowerPlant": 20, "CrossLight": 36, "Drone": 54, "Robot": 180}
        data["cgs_transitions"] = data["cgs_id"].map(trans_map)
        data["soa"] = pd.to_numeric(data["cgs_transitions"], errors="coerce") * pd.to_numeric(
            data["acg_states"], errors="coerce"
        )

    # 5) Tipos y limpieza
    data["acg_states"] = pd.to_numeric(data["acg_states"], errors="coerce")
    data["total_time"] = pd.to_numeric(data["total_time"], errors="coerce")
    data["soa"] = pd.to_numeric(data["soa"], errors="coerce")
    data = data.dropna(subset=["acg_states", "total_time", "soa"])

    # 6) Agregados por (familia, estados)
    df_time = (
        data.groupby(["cgs_id", "acg_states"], as_index=False)
        .agg(avg_total_time=("total_time", "mean"))
        .sort_values(["cgs_id", "acg_states"])
    )
    df_soa = (
        data.groupby(["cgs_id", "acg_states"], as_index=False)
        .agg(soa=("soa", "mean"))
        .sort_values(["cgs_id", "acg_states"])
    )

    # 7) Plot: izquierda=tiempos, derecha=SoA
    fig, ax_time = plt.subplots()
    ax_soa = ax_time.twinx()

    # Estética eje derecho: spine outward + ticks a la derecha + padding
    ax_soa.spines["right"].set_position(("outward", spine_outward_pts))
    ax_soa.yaxis.set_ticks_position("right")
    ax_soa.yaxis.set_label_position("right")
    ax_soa.tick_params(axis="y", which="both", pad=right_tick_pad)

    # Scatter (tiempos) — estos SÍ van a la leyenda
    for fam, sub in df_time.groupby("cgs_id"):
        ax_time.scatter(
            sub["acg_states"], sub["avg_total_time"],
            s=point_size, alpha=0.85,
            color=colors.get(fam, "#888888"),
            label=fam,
        )

    # Líneas SoA — SIN leyenda
    for fam, sub in df_soa.groupby("cgs_id"):
        sub = sub.sort_values("acg_states")
        ax_soa.plot(
            sub["acg_states"], sub["soa"],
            linestyle="--", linewidth=1.2, alpha=0.9,
            color=colors.get(fam, "#888888"),
            label="_nolegend_",
        )

    # 8) Etiquetas y estilo
    ax_time.set_xlabel("|φ|")
    ax_time.set_ylabel("Total time (s)")
    ax_soa.set_ylabel("|τ|·|φ|", labelpad=right_labelpad)
    ax_time.legend(title="cgs_id", loc="upper left", frameon=True)
    if show_grid:
        ax_time.grid(True, which="major", alpha=0.3)

    # --- aplicar notación científica (lineal) y recolocar offset ---
    try:
        fig.canvas.draw()  # asegura que exista offset_text
    except Exception:
        pass
    if y_sci:
        _apply_scientific(ax_time.yaxis, limits=sci_limits, use_mathtext=use_mathtext, side="left")
    if soa_y_sci:
        _apply_scientific(ax_soa.yaxis, limits=sci_limits, use_mathtext=use_mathtext, side="right")

    # reservar margen derecho para capturar offset + ylabel del eje derecho
    fig.subplots_adjust(right=right_margin)

    # 9) Salida
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()


def plot_totaltime_vs_acgstates_with_soa_only_drone_robot(
    csv_paths: list[Path],
    point_size: int = 10,
    show_grid: bool = False,
    show: bool = True,
    output_path: Path | None = None,
) -> None:
    """
    Muestra SOLO Drone y Robot.
    X = acg_states.
    Y izquierdo  = promedio de total_time (s) por (cgs_id, acg_states).
    Y derecho    = SoA crudo = cgs_transitions * acg_states (sin escalar).
    Leyenda dentro de la gráfica (arriba-izquierda).
    """
    # Colores fijos solicitados
    family_colors = {
        "Robot": "#d62728",       # rojo
        "CrossLight": "#1f77b4",  # (no se usa aquí)
        "NuclearPlant": "#2ca02c",# (no se usa aquí)
        "Drone": "#ff7f0e",       # naranja
    }
    include = {"Drone", "Robot"}

    # Leer y concatenar
    if not isinstance(csv_paths, list) or not all(isinstance(p, Path) for p in csv_paths):
        raise TypeError("csv_paths debe ser list[Path].")
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p, encoding="utf-8")
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # Validar columnas mínimas
    required = {"cgs_id", "acg_states", "total_time"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)}; disponibles: {list(data.columns)}")

    # Tipos y filtrado
    data["acg_states"] = pd.to_numeric(data["acg_states"], errors="coerce")
    data["total_time"] = pd.to_numeric(data["total_time"], errors="coerce")
    data = data.dropna(subset=["cgs_id", "acg_states", "total_time"])

    # Filtrar familias objetivo
    data = data[data["cgs_id"].isin(include)]
    if data.empty:
        raise ValueError("No hay filas de Drone/Robot tras el filtrado.")

    # Transiciones por CGS
    transitions_map = {
        "NuclearPlant": 20,
        "PowerPlant": 20,   # por si aparece
        "CrossLight": 36,
        "Drone": 54,
        "Robot": 180,
    }
    data["cgs_transitions"] = data["cgs_id"].map(transitions_map)
    if data["cgs_transitions"].isna().any():
        faltan = sorted(set(data.loc[data["cgs_transitions"].isna(), "cgs_id"]))
        raise ValueError(f"No hay mapeo cgs_transitions para: {faltan}")

    # SoA crudo
    data["state_of_the_art"] = data["cgs_transitions"] * data["acg_states"]

    # Media por (familia, estados)
    df_mean = (
        data.groupby(["cgs_id", "acg_states"], as_index=False)
            .agg(total_time=("total_time", "mean"),
                 cgs_transitions=("cgs_transitions", "first"))
            .sort_values(["cgs_id", "acg_states"])
    )

    # Figura con dos ejes Y
    fig, ax_time = plt.subplots()
    ax_soa = ax_time.twinx()

    # 4) si aún queda justo, deja más margen a la derecha
    fig.subplots_adjust(right=0.86)
    # o crea la figura con: fig, ax_time = plt.subplots(constrained_layout=True)

    # Scatter de tiempos (izquierda)
    for fam, sub in df_mean.groupby("cgs_id"):
        ax_time.scatter(
            sub["acg_states"],
            sub["total_time"],
            s=point_size,
            alpha=0.85,
            color=family_colors.get(fam, "#888888"),
            label=fam,
        )

    # Líneas SoA crudo (derecha): y_R = transitions_f * x
    for fam, sub in df_mean.groupby("cgs_id"):
        k = float(sub["cgs_transitions"].iloc[0])
        x_min = sub["acg_states"].min()
        x_max = sub["acg_states"].max()
        ax_soa.plot(
            [x_min, x_max], [k * x_min, k * x_max],
            linestyle="--",
            linewidth=1.2,
            color=family_colors.get(fam, "#888888"),
            alpha=0.9,
        )

    # Etiquetas y leyenda (dentro, esquina superior izquierda)
    ax_time.set_xlabel("|φ|")
    ax_time.set_ylabel("Total time (s)")
    ax_soa.set_ylabel("|τ|·|φ|")

    ax_time.legend(
        title="cgs_id",
        loc="upper left",         # dentro, arriba-izquierda
        frameon=True
    )

    if show_grid:
        ax_time.grid(True, which="major", alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

def plot_totaltime_vs_acgstates_with_soa_only_drone_robot_final(
    csv_paths: list[Path],
    point_size: int = 15,
    show_grid: bool = False,
    show: bool = True,
    output_path: Optional[Path] = None,
    # --- estética eje derecho / layout ---
    spine_outward_pts: float = 8.0,
    right_tick_pad: int = 6,
    right_labelpad: float = 10.0,
    right_margin: float = 0.86,
    # --- notación científica (lineal) ---
    y_sci: bool = True,
    soa_y_sci: bool = True,
    sci_limits: tuple[int, int] = (-3, 3),
    use_mathtext: bool = True,
) -> None:
    """
    Solo Drone y Robot.
    X: acg_states (|φ|).
    Y izquierda : promedio de total_time (s) por (cgs_id, acg_states) — scatter.
    Y derecha   : SoA (= |τ|·|φ|) — línea discontinua (sin entrar en la leyenda).

    Estética añadida (idéntica a la función NP/CL):
    - Eje derecho con spine desplazada hacia fuera, ticks a la derecha y padding configurable.
    - Reservado margen derecho fijo para capturar *ylabel* y offset `×10^k`.
    - Opción de notación científica (lineal) en ambos ejes Y mediante `_apply_scientific`.
    """
    colors = {
        "Robot": "#d62728",
        "CrossLight": "#1f77b4",
        "NuclearPlant": "#2ca02c",
        "Drone": "#ff7f0e",
    }
    include = {"Drone", "Robot"}

    # 1) Leer y concatenar
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p, encoding="utf-8")
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    # 2) Validaciones mínimas
    required = {"cgs_id", "acg_states", "total_time"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)}; disponibles: {list(data.columns)}")

    # 3) Filtrar familias objetivo
    data = data[data["cgs_id"].isin(include)]
    if data.empty:
        raise ValueError("No hay filas de Drone/Robot tras el filtrado.")

    # 4) Asegurar columna 'soa'
    if "soa" not in data.columns:
        if "cgs_transitions" not in data.columns:
            trans_map = {"NuclearPlant": 20, "PowerPlant": 20, "CrossLight": 36, "Drone": 54, "Robot": 180}
            data["cgs_transitions"] = data["cgs_id"].map(trans_map)
        data["soa"] = pd.to_numeric(data["cgs_transitions"], errors="coerce") * pd.to_numeric(
            data["acg_states"], errors="coerce"
        )

    # 5) Tipos y limpieza
    data["acg_states"] = pd.to_numeric(data["acg_states"], errors="coerce")
    data["total_time"] = pd.to_numeric(data["total_time"], errors="coerce")
    data["soa"] = pd.to_numeric(data["soa"], errors="coerce")
    data = data.dropna(subset=["acg_states", "total_time", "soa"])

    # 6) Agregados
    df_time = (
        data.groupby(["cgs_id", "acg_states"], as_index=False)
        .agg(avg_total_time=("total_time", "mean"))
        .sort_values(["cgs_id", "acg_states"])
    )
    df_soa = (
        data.groupby(["cgs_id", "acg_states"], as_index=False)
        .agg(soa=("soa", "mean"))
        .sort_values(["cgs_id", "acg_states"])
    )

    # 7) Plot: izquierda=tiempos, derecha=SoA
    fig, ax_time = plt.subplots()
    ax_soa = ax_time.twinx()

    # Estética eje derecho
    ax_soa.spines["right"].set_position(("outward", spine_outward_pts))
    ax_soa.yaxis.set_ticks_position("right")
    ax_soa.yaxis.set_label_position("right")
    ax_soa.tick_params(axis="y", which="both", pad=right_tick_pad)

    # Scatter (tiempos) — entra en la leyenda
    for fam, sub in df_time.groupby("cgs_id"):
        ax_time.scatter(
            sub["acg_states"], sub["avg_total_time"],
            s=point_size, alpha=0.85,
            color=colors.get(fam, "#888888"),
            label=fam,
        )

    # Línea SoA — fuera de la leyenda
    for fam, sub in df_soa.groupby("cgs_id"):
        sub = sub.sort_values("acg_states")
        ax_soa.plot(
            sub["acg_states"], sub["soa"],
            linestyle="--", linewidth=1.2, alpha=0.9,
            color=colors.get(fam, "#888888"),
            label="_nolegend_",
        )

    # 8) Etiquetas y estilo
    ax_time.set_xlabel("|φ|")
    ax_time.set_ylabel("Total time (s)")
    ax_soa.set_ylabel("|τ|·|φ|", labelpad=right_labelpad)
    ax_time.legend(title="cgs_id", loc="upper left", frameon=True)
    if show_grid:
        ax_time.grid(True, which="major", alpha=0.3)

    # 9) Notación científica (lineal) y recolocado de offset
    try:
        fig.canvas.draw()
    except Exception:
        pass
    if y_sci:
        _apply_scientific(ax_time.yaxis, limits=sci_limits, use_mathtext=use_mathtext, side="left")
    if soa_y_sci:
        _apply_scientific(ax_soa.yaxis, limits=sci_limits, use_mathtext=use_mathtext, side="right")

    # 10) Reservar margen derecho para capturar offset + ylabel del eje derecho
    fig.subplots_adjust(right=right_margin)

    # 11) Salida
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

def make_family_color_map(
    csv_paths: list[Path],
    palette: str = "tab20",
    family_order: list[str] | None = None,
) -> dict[str, object]:
    """
    Lee todos los CSVs y crea un mapping estable family->color usando `palette`.
    Si `family_order` se da, usa ese orden; si no, orden alfabético.
    """
    dfs = [pd.read_csv(p) for p in csv_paths]
    all_fams = pd.concat(dfs, ignore_index=True)["family"].dropna().unique().tolist()
    families = family_order if family_order else sorted(all_fams)

    cmap = plt.cm.get_cmap(palette, len(families))
    return {fam: cmap(i) for i, fam in enumerate(families)}


def plot_familias_acgtime_vs_closure(
    csv_paths: list[Path],
    show: bool = True,
    output_path: Path | None = None,
    show_grid: bool = False,
    point_size: int = 15,
    family_colors: dict[str, object] | None = None,  # pásalo para colores estables
    palette: str = "tab20",
    y_scientific: bool = False,
    sci_limits: tuple[int, int] = (-3, 3),
    use_mathtext: bool = True,
) -> None:
    """
    Scatter: X=closure_size, Y=acg_build. Colores por familia.
    Requiere columnas: ['family','closure_size','acg_build'].
    """
    dfs = [pd.read_csv(p) for p in csv_paths]
    data = pd.concat(dfs, ignore_index=True)

    families = sorted(data["family"].unique())
    if family_colors is None:
        # Si no te pasan mapping, lo generamos con lo presente (menos recomendable)
        cmap = plt.cm.get_cmap(palette, len(families))
        family_colors = {fam: cmap(i) for i, fam in enumerate(families)}

    fig, ax = plt.subplots()
    for family, group in data.groupby("family"):
        ax.scatter(
            group["closure_size"],
            group["acg_build"],
            label=family,
            s=point_size,
            color=family_colors.get(family, "#888888"),
        )

    ax.set_xlabel("Closure Size")
    ax.set_ylabel("ACG Build Time (s)")
    ax.set_title("ACG Build Time vs Closure Size by Family")
    ax.legend(title="family", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.subplots_adjust(right=0.8)

    if show_grid:
        ax.grid(True, which="major", alpha=0.3)

    if y_scientific:
        ax.ticklabel_format(axis="y", style="sci", scilimits=sci_limits, useMathText=use_mathtext)
        ax.yaxis.get_offset_text().set_x(-0.06)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()


def plot_familias_acgtime_vs_closure_conlineas(
    csv_paths: list[Path],
    show: bool = True,
    output_path: Path | None = None,
    show_grid: bool = False,
    marker_size: int = 3.5,
    line_width: float = 1.2,
    family_colors: dict[str, object] | None = None,  # usa el MISMO mapping aquí
    palette: str = "tab20",
    y_scientific: bool = False,
    sci_limits: tuple[int, int] = (-3, 3),
    use_mathtext: bool = True,
) -> None:
    """
    Líneas: X=closure_size, Y=acg_build conectando por familia.
    Usa el mapping `family_colors` para mantener colores coherentes en subsets.
    """
    dfs = [pd.read_csv(p) for p in csv_paths]
    data = pd.concat(dfs, ignore_index=True)

    families = sorted(data["family"].unique())
    if family_colors is None:
        # Fallback: genera mapping con lo presente (menos recomendable para consistency)
        cmap = plt.cm.get_cmap(palette, len(families))
        family_colors = {fam: cmap(i) for i, fam in enumerate(families)}

    fig, ax = plt.subplots()
    for family, group in data.groupby("family"):
        group = group.sort_values("closure_size")
        ax.plot(
            group["closure_size"],
            group["acg_build"],
            label=family,
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            linestyle="-",
            color=family_colors.get(family, "#888888"),
        )

    ax.set_xlabel("Closure Size")
    ax.set_ylabel("ACG Build Time (s)")
    ax.set_title("ACG Build Time vs Closure Size by Family")
    ax.legend(title="family", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.subplots_adjust(right=0.8)

    if show_grid:
        ax.grid(True, which="major", alpha=0.3)

    if y_scientific:
        ax.ticklabel_format(axis="y", style="sci", scilimits=sci_limits, useMathText=use_mathtext)
        ax.yaxis.get_offset_text().set_x(-0.06)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

def plot_familias_acgtime_vs_closure_conlineas_leyenda_dentro(
    csv_paths: list[Path],
    show: bool = True,
    output_path: Path | None = None,
    show_grid: bool = False,
    marker_size: int = 3.5,
    line_width: float = 1.2,
    family_colors: dict[str, object] | None = None,
    palette: str = "tab20",
    y_scientific: bool = False,
    sci_limits: tuple[int, int] = (-3, 3),
    use_mathtext: bool = True,
) -> None:
    """
    Líneas: X = closure_size, Y = acg_build conectando por familia.
    La leyenda se muestra dentro del gráfico (arriba-izquierda).
    Si `family_colors` no se proporciona, se usa `palette` para asignar colores.
    Requiere columnas: ['family','closure_size','acg_build'].
    """
    dfs = [pd.read_csv(p) for p in csv_paths]
    data = pd.concat(dfs, ignore_index=True)

    families = sorted(data["family"].unique())
    if family_colors is None:
        cmap = plt.cm.get_cmap(palette, len(families))
        family_colors = {fam: cmap(i) for i, fam in enumerate(families)}

    fig, ax = plt.subplots()
    for family, group in data.groupby("family"):
        group = group.sort_values("closure_size")
        ax.plot(
            group["closure_size"],
            group["acg_build"],
            label=family,
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            linestyle="-",
            color=family_colors.get(family, "#888888"),
        )

    ax.set_xlabel("Closure Size")
    ax.set_ylabel("ACG Build Time (s)")
    ax.set_title("ACG Build Time vs Closure Size by Family")

    # Leyenda DENTRO del área de dibujo (arriba-izquierda)
    ax.legend(title="family", loc="upper left", frameon=True)

    if show_grid:
        ax.grid(True, which="major", alpha=0.3)

    if y_scientific:
        ax.ticklabel_format(axis="y", style="sci", scilimits=sci_limits, useMathText=use_mathtext)
        ax.yaxis.get_offset_text().set_x(-0.06)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

def plot_familias_gametime_vs_closure(
    csv_paths: list[Path],
    show: bool = True,
    output_path: Path | None = None,
    show_grid: bool = False,
    point_size: int = 12,
    family_colors: dict[str, object] | None = None,  # nuevo
    palette: str = "tab20",                           # nuevo
):
    """
    Scatter: X=closure_size, Y=acg_build. Sin grid por defecto.
    Requiere columnas: ['family','closure_size','acg_build'].
    Colores únicos por familia usando `family_colors` o un colormap (por defecto, tab20).
    """
    dfs = [pd.read_csv(path) for path in csv_paths]
    data = pd.concat(dfs, ignore_index=True)

    # familias y colores
    families = sorted(data["family"].unique())
    cmap = plt.cm.get_cmap(palette, len(families))
    auto_colors = {fam: cmap(i) for i, fam in enumerate(families)}
    if family_colors:
        # override manual si se proporciona
        colors = {fam: family_colors.get(fam, auto_colors[fam]) for fam in families}
    else:
        colors = auto_colors

    fig, ax = plt.subplots()
    for family, group in data.groupby("family"):
        ax.scatter(
            group["closure_size"],
            group["game_build"],
            label=family,
            s=point_size,
            color=colors[family],
        )

    ax.set_xlabel("Closure Size")
    ax.set_ylabel("Game Build Time (s)")
    ax.set_title("ACG Build Time vs Closure Size by Family")

    # Leyenda fuera del área de trazado: centrada a la derecha
    ax.legend(title="family", loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    if show_grid:
        ax.grid(True, which="major", alpha=0.3)

    # margen a la derecha para la leyenda
    fig.subplots_adjust(right=0.8)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()


def plot_familias_acgtime_vs_closure_mediado(
    csv_paths: list[Path],
    show: bool = True,
    output_path: Path | None = None,
    show_grid: bool = False,
):
    """
    Media global por closure_size combinando múltiples CSVs.
    Requiere columnas: ['closure_size','acg_build'].
    """
    dfs = [pd.read_csv(path, usecols=['closure_size', 'acg_build']) for path in csv_paths]
    data = pd.concat(dfs, ignore_index=True)

    grouped = data.groupby('closure_size', as_index=True)['acg_build'].mean().sort_index()

    fig, ax = plt.subplots()
    ax.plot(grouped.index, grouped.values)

    ax.set_xlabel('Closure Size')
    ax.set_ylabel('Mean ACG Build Time (s)')
    ax.set_title('Mean ACG Build Time vs Closure Size')

    if show_grid:
        ax.grid(True, which='major', alpha=0.3)

    if output_path:
        fig.savefig(output_path)
    if show:
        plt.show()



def scalability_family2(n_min: int,
                       n_max: int,
                       cgs_builder,         # e.g. generate_lights_cgs
                       formula_builder,     # e.g. generate_flatX_spec
                       family_id: str,      # e.g. "Flat-X"
                       csv_path: Path):

    # Abrimos con buffering=1 para flush por línea
    with csv_path.open("w", newline="", encoding="utf-8", buffering=1) as f:
        wr = csv.writer(f)
        # Cabecera
        wr.writerow([
            "family", "n", "states",
            "acg_size", "closure_size", "formula",
            "acg_build",
            "game_states", "game_edges", "game_build",
            "solve_time", "total_time", "sat"
        ])
        f.flush()                  # flush tras la cabecera

        for n in range(n_min, n_max + 1):
            print(f"▶ {family_id}  n={n}")

            # Generar CGS y fórmula
            cgs = cgs_builder(n)
            phi = formula_builder(n)
            formula_str = phi.to_formula()

            # Construir ACG y medir
            t0 = time.perf_counter()
            acg, acg_size, acg_build = build_acg_with_timer2(phi, cgs)
            closure_size = len(acg.states)
            print(f"  → ACG built in {acg_build:.6f}s, size={acg_size}, closure={closure_size}")

            # Construir juego y medir
            t1 = time.perf_counter()
            S, E, S1, S2, B, s0 = build_game(acg, cgs)
            game_build = time.perf_counter() - t1
            game_states = len(S)
            game_edges  = len(E)
            print(f"  → Game states={game_states}, edges={game_edges}, built in {game_build:.6f}s")

            # Resolver juego y medir
            t2 = time.perf_counter()
            S_win, _ = solve_buchi_game(S, E, S1, S2, B)
            solve_time  = time.perf_counter() - t2
            total_time  = time.perf_counter() - t0
            sat = "Yes" if s0 in S_win else "No"
            print(f"  → Solved in {solve_time:.6f}s (total {total_time:.6f}s), sat={sat}")

            # Escribir fila y forzar flush
            wr.writerow([
                family_id,
                n,
                len(cgs.states),
                acg_size,
                closure_size,
                formula_str,
                acg_build,
                game_states,
                game_edges,
                game_build,
                solve_time,
                total_time,
                sat
            ])
            f.flush()  # asegura que la línea llega al disco

            # Opcional: para estar extra seguros, descomenta esto:
            # os.fsync(f.fileno())

def scalability_family_final(n_min: int,
                       n_max: int,
                       cgs_builder,         # e.g. generate_lights_cgs
                       formula_builder,     # e.g. generate_flatX_spec
                       family_id: str,      # e.g. "Flat-X"
                       csv_path: Path):

    # Abrimos con buffering=1 para flush por línea
    with csv_path.open("w", newline="", encoding="utf-8", buffering=1) as f:
        wr = csv.writer(f)
        # Cabecera
        wr.writerow([
            "family", "n", "states",
            "acg_size", "closure_size", "formula",
            "acg_build",
            "game_states", "game_edges", "game_build",
            "solve_time", "total_time", "sat"
        ])
        f.flush()                  # flush tras la cabecera

        for n in range(n_min, n_max + 1):
            print(f"▶ {family_id}  n={n}")

            # Generar CGS y fórmula
            cgs = cgs_builder(n)
            phi = formula_builder(n)
            formula_str = phi.to_formula()

            # Construir ACG y medir
            t0 = time.perf_counter()
            acg, acg_size, acg_build = build_acg_with_timer_final(phi, cgs)
            closure_size = len(acg.states)
            print(f"  → ACG built in {acg_build:.6f}s, size={acg_size}, closure={closure_size}")

            # Construir juego y medir
            t1 = time.perf_counter()
            S, E, S1, S2, B, s0 = build_game(acg, cgs)
            game_build = time.perf_counter() - t1
            game_states = len(S)
            game_edges  = len(E)
            print(f"  → Game states={game_states}, edges={game_edges}, built in {game_build:.6f}s")

            # Resolver juego y medir
            t2 = time.perf_counter()
            S_win, _ = solve_buchi_game(S, E, S1, S2, B)
            solve_time  = time.perf_counter() - t2
            total_time  = time.perf_counter() - t0
            sat = "Yes" if s0 in S_win else "No"
            print(f"  → Solved in {solve_time:.6f}s (total {total_time:.6f}s), sat={sat}")

            # Escribir fila y forzar flush
            wr.writerow([
                family_id,
                n,
                len(cgs.states),
                acg_size,
                closure_size,
                formula_str,
                acg_build,
                game_states,
                game_edges,
                game_build,
                solve_time,
                total_time,
                sat
            ])
            f.flush()  # asegura que la línea llega al disco

            # Opcional: para estar extra seguros, descomenta esto:
            # os.fsync(f.fileno())


def familias_solo_acg(
    n_min: int,
    n_max: int,
    cgs_builder,       # función que recibe n y devuelve un CGS con .states
    formula_builder,   # función que recibe n y devuelve una fórmula con .to_formula()
    family_id: str,
    csv_path: Path
):
    """
    Variante de scalability_family2 que solo registra datos de compilación del ACG.
    Cabezera reducida: solo campos relevantes al ACG.
    """
    header = [
        "family", "n", "states",
        "acg_size", "closure_size", "formula",
        "acg_build"
    ]

    # Abrir CSV en modo escritura con flushing por línea
    with csv_path.open("w", newline="", encoding="utf-8", buffering=1) as f:
        wr = csv.writer(f)
        wr.writerow(header)
        f.flush()

        for n in range(n_min, n_max + 1):
            print(f"▶ {family_id}  n={n}")

            # Construir CGS y fórmula
            cgs = cgs_builder(n)
            phi = formula_builder(n)
            formula_str = phi.to_formula()
            formula_tree = phi.to_tree()
            print(f" formula generada : {formula_str}\n")
            print(f" formula tree : {formula_tree}")
            states = len(cgs.states)

            # Medir compilación ACG
            t0 = time.perf_counter()
            acg, acg_size, acg_build = build_acg_with_timer_final(phi, cgs)
            closure_size = len(acg.states)
            print(f"  → ACG built in {acg_build:.6f}s (size={acg_size}, closure={closure_size})")

            # Escribir solo métricas de ACG
            wr.writerow([
                family_id,
                n,
                states,
                acg_size,
                closure_size,
                formula_str,
                f"{acg_build:.6f}"
            ])
            f.flush()

def familias_pipeline_completa(
    n_min: int,
    n_max: int,
    cgs_builder,
    formula_builder,
    family_id: str,
    csv_path: Path,
) -> None:
    """Ejecuta la *pipeline completa* (ACG → Juego → Solver) por n∈[n_min,n_max]
    y registra métricas en CSV con cabecera estándar.

    Mantiene `familias_solo_acg` intacta (esta es otra función).
    """
    header = [
        "family", "n", "states",
        "acg_size", "closure_size", "formula",
        "acg_build",
        "game_states", "game_edges", "game_build",
        "solve_time", "total_time", "sat",
    ]

    with csv_path.open("w", newline="", encoding="utf-8", buffering=1) as f:
        wr = csv.writer(f)
        wr.writerow(header)
        f.flush()

        for n in range(n_min, n_max + 1):
            print(f"▶ {family_id}  n={n}")

            # Generar CGS y fórmula
            cgs = cgs_builder(n)
            phi = formula_builder(n)
            formula_str = phi.to_formula()
            n_states = len(cgs.states)

            # ACG
            t0 = time.perf_counter()
            acg, acg_size, acg_build = build_acg_with_timer_final(phi, cgs)
            closure_size = len(acg.states)
            print(f"  → ACG built in {acg_build:.6f}s, size={acg_size}, closure={closure_size}")

            # Juego
            t1 = time.perf_counter()
            S, E, S1, S2, B, s0 = build_game(acg, cgs)
            game_build = time.perf_counter() - t1
            game_states = len(S)
            game_edges = len(E)
            print(f"  → Game states={game_states}, edges={game_edges}, built in {game_build:.6f}s")

            # Solver
            t2 = time.perf_counter()
            S_win, _ = solve_buchi_game(S, E, S1, S2, B)
            solve_time = time.perf_counter() - t2
            total_time = time.perf_counter() - t0
            sat = "Yes" if s0 in S_win else "No"
            print(f"  → Solved in {solve_time:.6f}s (total {total_time:.6f}s), sat={sat}")

            # Registrar
            wr.writerow([
                family_id,
                n,
                n_states,
                acg_size,
                closure_size,
                formula_str,
                f"{acg_build:.6f}",
                game_states,
                game_edges,
                f"{game_build:.6f}",
                f"{solve_time:.6f}",
                f"{total_time:.6f}",
                sat,
            ])
            f.flush()




def _acg_pipeline(phi, cgs):
    acg, acg_size, acg_build = build_acg_with_timer_final(phi, cgs)
    return (len(acg.states), len(acg.transitions), acg_build, acg_size)

def cuatros_solo_acg(
    cgs, cgs_id,
    min_depth, max_depth,
    samples_per_depth,
    csv_path: Path,
    overwrite_csv: bool = False,
    max_seconds: int = 120
):
    """
    Ejecuta solo la compilación del ACG y vuelca resultados en CSV
    con un header reducido únicamente a las métricas de ACG.
    """
    mode        = "w" if overwrite_csv or not csv_path.exists() else "a"
    need_header = overwrite_csv or not csv_path.exists()
    header = [
        "cgs_id", "batch_size",
        "depth", "sample_idx",
        "status",
        "formula", "formula_len", "gen_time",
        "acg_states", "acg_edges", "acg_build_time", "acg_size"
    ]

    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if need_header:
            wr.writerow(header)

        log(f"▶ START ACG-only {cgs_id} depths {min_depth}–{max_depth} samples={samples_per_depth}")

        for depth in range(min_depth, max_depth + 1):
            ok_count = 0
            attempt  = 0
            while ok_count < samples_per_depth:
                attempt += 1

                # 1) Generar fórmula
                try:
                    t0      = time.perf_counter()
                    phi     = generate_random_valid_atl_formula(
                                 cgs, depth=depth, modality_needed=(depth >= 3))
                    gen_time = time.perf_counter() - t0
                    phi_str  = phi.to_formula()
                    print(f" formula generated : {phi.to_formula()}")
                except Exception as e:
                    wr.writerow([
                        cgs_id, samples_per_depth, depth, attempt, "ERROR",
                        "", 0, 0.0,  # fórmula fallida
                        "", "", 0.0, ""  # ACG no ejecutado
                    ])
                    log(f"    ❌ formula gen error: {e}")
                    continue

                # 2) Compilación ACG con timeout
                ok, res, runtime = run_with_timeout(_acg_pipeline, max_seconds, phi, cgs)
                if not ok:
                    status = "TIMEOUT" if res == "TIMEOUT" else "ERROR"
                    wr.writerow([
                        cgs_id, samples_per_depth, depth, attempt, status,
                        phi_str[:60] + "…" if len(phi_str) > 60 else phi_str,
                        len(phi_str), round(gen_time, 6),
                        "", "", round(runtime, 6), ""  # ACG fallido
                    ])
                    log(f"    {status} after {runtime:.1f}s")
                    continue

                # 3) Éxito ACG
                acg_states, acg_edges, acg_build, acg_size = res
                wr.writerow([
                    cgs_id, samples_per_depth, depth, ok_count+1, "OK",
                    phi_str[:60] + "…" if len(phi_str) > 60 else phi_str,
                    len(phi_str), round(gen_time, 6),
                    acg_states, acg_edges, round(acg_build, 6), acg_size
                ])
                log(f"    OK ACG [{ok_count+1}/{samples_per_depth}] {acg_build:.2f}s")
                ok_count += 1

        log(f"■ FINISHED ACG-only {cgs_id}\n")

def cuatros_acg_solo_aux(
    cgs,
    depth: int,
    max_seconds: int = 120,
    csv_path: Path = None,
    do_print: bool = True
):
    """
    Ejecuta UNA única iteración del pipeline ACG-only y devuelve un dict
    con el desglose de tiempos:
      - gen_time: generar la fórmula
      - spawn_overhead: overhead de run_with_timeout antes/después del build
      - acg_build: tiempo medido dentro del subprocess para build_acg
      - csv_time: tiempo de escritura CSV (si csv_path dado)
      - print_time: tiempo de print(phi) (si do_print=True)
      - total_time: tiempo total de esta función
    """
    results = {}
    t_start = time.perf_counter()

    # 1) generación de la fórmula
    t0 = time.perf_counter()
    phi = generate_random_valid_atl_formula(cgs, depth=depth, modality_needed=(depth >= 3))
    results['gen_time'] = time.perf_counter() - t0
    phi_str = phi.to_formula()

    t_pre = time.perf_counter()
    ok, res, runtime = run_with_timeout(_acg_pipeline, max_seconds, phi, cgs)
    t_post = time.perf_counter()
    results['spawn_overhead'] = (t_post - t_pre) - runtime
    if not ok:
        raise RuntimeError(f"ACG pipeline failed: {res}")
    results['acg_build'] = res[2]  # build_t

    # 3) escribir CSV si procede
    if csv_path:
        t_csv = time.perf_counter()
        mode = "a" if csv_path.exists() else "w"
        with csv_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    "cgs_id","depth","gen_time",
                    "spawn_overhead","acg_build",
                    "acg_states","acg_edges","acg_size"
                ])
            writer.writerow([
                getattr(cgs, 'id', 'cgs'), depth,
                f"{results['gen_time']:.6f}",
                f"{results['spawn_overhead']:.6f}",
                f"{results['acg_build']:.6f}",
                res[0], res[1], res[3]
            ])
        results['csv_time'] = time.perf_counter() - t_csv

    # 4) print de la fórmula
    if do_print:
        t_print = time.perf_counter()
        print(phi_str)
        results['print_time'] = time.perf_counter() - t_print

    results['total_time'] = time.perf_counter() - t_start
    return results

def cuatros_solo_acg_sync(
    cgs,
    cgs_id: str,
    min_depth: int,
    max_depth: int,
    samples_per_depth: int,
    csv_path: Path,
    overwrite_csv: bool = False
):
    """
    Ejecuta de forma síncrona solo la compilación del ACG y vuelca resultados en CSV,
    sin multiprocessing ni timeout.

    Parámetros:
      - cgs: objeto del modelo
      - cgs_id: identificador de experimento
      - min_depth, max_depth: rango de profundidad
      - samples_per_depth: número de muestras por profundidad
      - csv_path: Path al CSV de salida
      - overwrite_csv: si True sobrescribe el CSV, sino añade
    """
    # CSV setup
    mode = 'w' if overwrite_csv or not csv_path.exists() else 'a'
    need_header = overwrite_csv or not csv_path.exists()
    header = [
        'cgs_id', 'batch_size', 'depth', 'sample_idx', 'status',
        'formula', 'formula_len', 'gen_time',
        'acg_states', 'acg_edges', 'acg_build_time', 'acg_size'
    ]

    with csv_path.open(mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)

        log(f"▶ START ACG-sync {cgs_id} depths {min_depth}–{max_depth} samples={samples_per_depth}")

        for depth in range(min_depth, max_depth + 1):
            ok_count = 0
            attempt = 0
            while ok_count < samples_per_depth:
                attempt += 1

                # 1) generar fórmula
                try:
                    t0 = time.perf_counter()
                    phi = generate_random_valid_atl_formula(
                        cgs, depth=depth, modality_needed=(depth >= 3)
                    )
                    gen_time = time.perf_counter() - t0
                    phi_str = phi.to_formula()
                except Exception as e:
                    writer.writerow([
                        cgs_id, samples_per_depth, depth, attempt, 'ERROR',
                        '', 0, 0.0,
                        '', '', 0.0, ''
                    ])
                    log(f"    ❌ formula gen error: {e}")
                    continue

                # 2) compilación ACG directo y medición
                t1 = time.perf_counter()
                acg, acg_size, acg_build = build_acg_with_timer_final(phi, cgs)
                acg_states = len(acg.states)
                acg_edges = len(acg.transitions)
                build_time = time.perf_counter() - t1

                # 3) escribir CSV
                writer.writerow([
                    cgs_id, samples_per_depth, depth, ok_count + 1, 'OK',
                    phi_str[:60] + '…' if len(phi_str) > 60 else phi_str,
                    len(phi_str), f"{gen_time:.6f}",
                    acg_states, acg_edges, f"{build_time:.6f}", acg_size
                ])
                log(f"    OK ACG-sync [{ok_count+1}/{samples_per_depth}] {build_time:.2f}s")
                ok_count += 1

    log(f"■ FINISHED ACG-sync {cgs_id}\n")

def _full_pipeline(phi, cgs):
    """Devuelve todas las métricas del pipeline completo ACG→Game→Solver."""
    # ACG
    acg, acg_size, acg_build = build_acg_with_timer_final(phi, cgs)
    acg_states = len(acg.states)
    acg_edges  = len(acg.transitions)

    # Game
    t_g0 = time.perf_counter()
    S, E, S1, S2, B, s0 = build_game(acg, cgs)
    game_build = time.perf_counter() - t_g0
    game_states, game_edges = len(S), len(E)

    # Solver
    t_s0 = time.perf_counter()
    S_win, _ = solve_buchi_game(S, E, S1, S2, B)
    solve_t = time.perf_counter() - t_s0

    total_t = acg_build + game_build + solve_t  # consistente con tu CSV previo
    sat     = "Yes" if s0 in S_win else "No"

    return (
        acg_states, acg_edges, acg_build, acg_size,
        game_states, game_edges, game_build,
        solve_t, total_t, sat
    )


class _SingleWorker:
    """Pool de 1 proceso reusable con timeout por-tarea.
    WHY: evita el alto overhead de spawn por iteración, pero mantiene cancelación por timeout.
    """
    def __init__(self):
        self.ctx = get_context("spawn")  # explícito en Windows
        self.pool = None
        self._ensure()

    def _ensure(self):
        if self.pool is None:
            # maxtasksperchild para liberar memoria en ejecuciones largas
            self.pool = self.ctx.Pool(processes=1, maxtasksperchild=100)

    def run(self, func, args: tuple, timeout: float):
        try:
            self._ensure()
            async_res = self.pool.apply_async(func, args)
            return True, async_res.get(timeout=timeout), timeout
        except MpTimeout:
            # WHY: matar el worker colgado y recrearlo limpio
            self.pool.terminate(); self.pool.join(); self.pool = None
            return False, "TIMEOUT", timeout
        except Exception as e:  # noqa: BLE001
            # Reset por si el worker quedó inestable
            self.pool.terminate(); self.pool.join(); self.pool = None
            return False, f"ERROR: {e}", 0.0

    def close(self):
        if self.pool is not None:
            self.pool.close(); self.pool.join(); self.pool = None

def cuatros_pipeline_full(
    cgs,
    cgs_id: str,
    min_depth: int,
    max_depth: int,
    samples_per_depth: int,
    csv_path: Path,
    overwrite_csv: bool = False,
    max_seconds = None,
    use_timeout: bool = True,
    max_closure = None,
    max_game_states = None,
    max_game_edges = None,
):
    """
    Pipeline completo con control de tiempo y salvaguardas:
      - `use_timeout=True` → usa un pool de 1 proceso reutilizable con timeout por iteración.
      - `max_*` → corta fases potencialmente explosivas antes de seguir (heurístico, sin subproceso).
    """
    mode = "w" if overwrite_csv or not csv_path.exists() else "a"
    need_header = overwrite_csv or not csv_path.exists()

    header = [
        "cgs_id", "batch_size",
        "depth", "sample_idx", "status",
        "formula", "formula_len", "gen_time",
        "acg_states", "acg_edges", "acg_build_time", "acg_size",
        "game_states", "game_edges", "game_build_time",
        "solve_time", "total_time", "satisfiable",
    ]

    worker = _SingleWorker() if (use_timeout and max_seconds) else None

    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if need_header:
            wr.writerow(header)

        log(f"▶ START FULL {cgs_id}  depths {min_depth}–{max_depth} samples={samples_per_depth} timeout={max_seconds if use_timeout else 'OFF'}s")

        try:
            for depth in range(min_depth, max_depth + 1):
                ok_count, attempt = 0, 0
                while ok_count < samples_per_depth:
                    attempt += 1

                    # Fórmula
                    try:
                        t_gen0 = time.perf_counter()
                        phi = generate_random_valid_atl_formula(cgs, depth=depth, modality_needed=(depth >= 3))
                        gen_time = time.perf_counter() - t_gen0
                        phi_str = phi.to_formula()
                    except Exception as e:  # noqa: BLE001
                        wr.writerow([cgs_id, samples_per_depth, depth, attempt, "ERROR",
                                     "", "", "", "", "", "", "",
                                     "", "", "", "", "", "ERROR"])
                        log(f"    ❌ formula gen error: {e}")
                        continue

                    # Guard ACG (rápido, en proceso): si alguien pasa límites, evitamos fases siguientes
                    try:
                        acg, acg_size, acg_build = build_acg_with_timer_final(phi, cgs)
                        acg_states = len(acg.states); acg_edges = len(acg.transitions)
                    except Exception as e:  # noqa: BLE001
                        wr.writerow([cgs_id, samples_per_depth, depth, attempt, "ERROR",
                                     phi_str[:60] + "…" if len(phi_str) > 60 else phi_str,
                                     len(phi_str), f"{gen_time:.6f}",
                                     "", "", "", "",
                                     "", "", "",
                                     "", "", "ERROR"])
                        log(f"    ❌ ACG build error: {e}")
                        continue

                    # Heurísticas de corte (WHY: evitar trabajo costoso evidente)
                    if (max_closure and acg_states > max_closure):
                        wr.writerow([cgs_id, samples_per_depth, depth, attempt, "SKIPPED_BY_GUARD",
                                     phi_str[:60] + "…" if len(phi_str) > 60 else phi_str,
                                     len(phi_str), f"{gen_time:.6f}",
                                     acg_states, acg_edges, f"{acg_build:.6f}", acg_size,
                                     "", "", "", "", "", "SKIP"])
                        log(f"    ↷ skip by guard: closure={acg_states} > {max_closure}")
                        continue

                    if worker is None:
                        # Ruta síncrona (sin timeout). WHY: más rápida si confías en que no se atasca.
                        t_g0 = time.perf_counter()
                        S, E, S1, S2, B, s0 = build_game(acg, cgs)
                        game_build = time.perf_counter() - t_g0
                        game_states, game_edges = len(S), len(E)

                        if (max_game_states and game_states > max_game_states) or (max_game_edges and game_edges > max_game_edges):
                            wr.writerow([cgs_id, samples_per_depth, depth, attempt, "SKIPPED_BY_GUARD",
                                         phi_str[:60] + "…" if len(phi_str) > 60 else phi_str,
                                         len(phi_str), f"{gen_time:.6f}",
                                         acg_states, acg_edges, f"{acg_build:.6f}", acg_size,
                                         game_states, game_edges, f"{game_build:.6f}",
                                         "", "", "SKIP"])
                            log(f"    ↷ skip by guard: game ~ big (S={game_states},E={game_edges})")
                            continue

                        t_s0 = time.perf_counter()
                        S_win, _ = solve_buchi_game(S, E, S1, S2, B)
                        solve_t = time.perf_counter() - t_s0
                        total_t = acg_build + game_build + solve_t
                        sat = "Yes" if s0 in S_win else "No"

                        wr.writerow([cgs_id, samples_per_depth, depth, ok_count + 1, "OK",
                                     phi_str[:60] + "…" if len(phi_str) > 60 else phi_str,
                                     len(phi_str), f"{gen_time:.6f}",
                                     acg_states, acg_edges, f"{acg_build:.6f}", acg_size,
                                     game_states, game_edges, f"{game_build:.6f}",
                                     f"{solve_t:.6f}", f"{total_t:.6f}", sat])
                        log(f"    OK [{ok_count+1}/{samples_per_depth}] total={total_t:.2f}s sat={sat}")
                        ok_count += 1
                        f.flush()
                    else:
                        # Ruta con timeout en worker: repetimos todo en el subproceso para poder abortar
                        ok, res, _ = worker.run(_full_pipeline, (phi, cgs), float(max_seconds))
                        if not ok:
                            status = "TIMEOUT" if res == "TIMEOUT" else "ERROR"
                            wr.writerow([cgs_id, samples_per_depth, depth, attempt, status,
                                         phi_str[:60] + "…" if len(phi_str) > 60 else phi_str,
                                         len(phi_str), f"{gen_time:.6f}",
                                         acg_states, acg_edges, f"{acg_build:.6f}", acg_size,
                                         "", "", "",
                                         "", "", status])
                            log(f"    {status} (worker) depth={depth}")
                            f.flush()
                            continue

                        (acg_states2, acg_edges2, acg_build2, acg_size2,
                         game_states, game_edges, game_build,
                         solve_t, total_t, sat) = res

                        # Nota: ACG se recompila en worker; registramos las métricas del worker
                        wr.writerow([cgs_id, samples_per_depth, depth, ok_count + 1, "OK",
                                     phi_str[:60] + "…" if len(phi_str) > 60 else phi_str,
                                     len(phi_str), f"{gen_time:.6f}",
                                     acg_states2, acg_edges2, f"{acg_build2:.6f}", acg_size2,
                                     game_states, game_edges, f"{game_build:.6f}",
                                     f"{solve_t:.6f}", f"{total_t:.6f}", sat])
                        log(f"    OK [{ok_count+1}/{samples_per_depth}] total={total_t:.2f}s sat={sat}")
                        ok_count += 1
                        f.flush()
        finally:
            if worker:
                worker.close()
            log(f"■ FINISHED FULL {cgs_id}\n")

REQUIRED_COLS = [
    "cgs_id", "batch_size", "depth", "sample_idx", "status",
    "formula", "formula_len", "gen_time",
    "acg_states", "acg_edges", "acg_build_time", "acg_size",
    "game_states", "game_edges", "game_build_time",
    "solve_time", "total_time", "satisfiable",
]


def merge_pipeline_csvs(
    csv_paths: list[Path],
    output_path: Path,
    sort_by: list[str] | None = ("cgs_id", "depth", "sample_idx"),
    drop_duplicates: bool = True,
) -> None:
    """
    Une múltiples CSVs de la *misma estructura* (pipeline completa) en uno solo.

    - Valida columnas requeridas (si falta alguna, error claro).
    - Concatena en el orden dado.
    - (Opcional) elimina duplicados completos.
    - (Opcional) ordena por claves lógicas (por defecto: cgs_id, depth, sample_idx).
    - Guarda en `output_path` en UTF-8 sin índice.
    """
    # Validaciones básicas
    if not csv_paths:
        raise ValueError("csv_paths está vacío")

    for p in csv_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"No existe: {p}")

    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        df = pd.read_csv(p)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{p}: faltan columnas requeridas: {missing}")
        # Reordenar columnas a REQUIRED_COLS por consistencia
        df = df[REQUIRED_COLS]
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    if drop_duplicates:
        # Duplicado exacto de fila
        merged = merged.drop_duplicates().reset_index(drop=True)

    if sort_by:
        merged = merged.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)

    # Asegurar carpeta destino
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(output_path, index=False, encoding="utf-8")

DEFAULT_TRANSITIONS = {
    "NuclearPlant": 20,
    "PowerPlant": 20,   # alias por si aparece así
    "CrossLight": 36,
    "Drone": 54,
    "Robot": 180,
}


def add_transitions_and_soa(
    input_csv: Path,
    output_csv: Path,
    mapping: dict[str, int] | None = None,
    encoding: str = "utf-8",
    overwrite: bool = True,
) -> None:
    """
    Lee un CSV con columnas al menos ['cgs_id', 'acg_states'],
    añade:
      - 'cgs_transitions' a partir de un mapeo por cgs_id
      - 'soa' = cgs_transitions * acg_states
    y escribe un CSV idéntico al original pero con esas dos columnas añadidas al final.

    Parámetros
    ----------
    input_csv : Path
        Ruta del CSV de entrada.
    output_csv : Path
        Ruta del CSV de salida (se crea/sobrescribe).
    mapping : dict[str,int] | None
        Mapeo cgs_id -> nº transiciones. Si None, usa DEFAULT_TRANSITIONS.
    encoding : str
        Codificación para leer/escribir.
    overwrite : bool
        Si False y output_csv existe, lanza error.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe input_csv: {input_csv}")

    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"Ya existe output_csv: {output_csv}")

    # Leer
    df = pd.read_csv(input_csv, encoding=encoding)

    # Quitar nombres de columnas duplicados si los hubiera
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Validación mínima
    required = {"cgs_id", "acg_states"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas {sorted(missing)}; disponibles: {list(df.columns)}")

    # Mapeo de transiciones
    trans_map = dict(DEFAULT_TRANSITIONS if mapping is None else mapping)

    # Asignar transiciones y validar desconocidos
    df["cgs_transitions"] = df["cgs_id"].map(trans_map)
    if df["cgs_transitions"].isna().any():
        unknown = sorted(df.loc[df["cgs_transitions"].isna(), "cgs_id"].unique())
        raise ValueError(f"cgs_id sin mapeo en 'mapping': {unknown}")

    # Tipos numéricos
    df["acg_states"] = pd.to_numeric(df["acg_states"], errors="coerce")
    df["cgs_transitions"] = pd.to_numeric(df["cgs_transitions"], errors="coerce")

    # SOA = transiciones * estados
    df["soa"] = df["cgs_transitions"] * df["acg_states"]

    # Guardar
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding=encoding, lineterminator="\n")


def make_family_color_map(csv_paths: list[Path], palette: str = "tab20",
                          family_order: list[str] | None = None) -> dict[str, object]:
    dfs = [pd.read_csv(p) for p in csv_paths]
    fams_all = pd.concat(dfs, ignore_index=True)["family"].dropna().astype(str).str.strip().unique().tolist()
    families = family_order if family_order else sorted(fams_all)
    cmap = plt.cm.get_cmap(palette, len(families))
    return {fam: cmap(i) for i, fam in enumerate(families)}

# -------------------------------------------------------------------
# Debug: revisar por qué no sale SoA para alguna familia
# -------------------------------------------------------------------
def debug_soa_summary(csv_paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    # limpiar nombres y tipos
    df["family"] = df["family"].astype(str).str.strip()
    for col in ("states", "total_time", "soa"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["family", "states", "total_time", "soa"])
    g = (df.groupby("family")
           .agg(rows=("soa","size"), soa_min=("soa","min"), soa_max=("soa","max"),
                t_min=("total_time","min"), t_max=("total_time","max"))
           .reset_index())
    print("\n[DEBUG SoA] resumen por family:\n", g.to_string(index=False))
    return g


def _apply_scientific(axis, limits=(-3, 3), use_mathtext=True, side: str = "left") -> None:
    """
    Formatea con notación científica SOLO si el eje es lineal.
    `axis` = ax.yaxis o ax.xaxis. `side` solo ajusta el offset '×10^k'.
    """
    is_y = getattr(axis, "axis_name", "y") == "y"
    scale = axis.axes.get_yscale() if is_y else axis.axes.get_xscale()
    if scale != "linear":
        return  # no tocar ejes log

    fmt = ScalarFormatter(useMathText=use_mathtext)
    fmt.set_powerlimits(limits)   # activa científica fuera de estos límites
    fmt.set_useOffset(False)      # evita offsets tipo "1e5 × 1e3"
    axis.set_major_formatter(fmt)

    # separa un poco el texto del offset
    try:
        off = axis.get_offset_text()
        if is_y:
            off.set_x(-0.06 if side == "left" else 1.11)
        else:
            off.set_y(2)
    except Exception:
        pass


def plot_families_states_vs_totaltime_with_soa_linear(
    csv_paths: list[Path],
    family_colors: dict[str, object] | None = None,
    palette: str = "tab20",
    marker_size: int = 4,
    line_width: float = 1.2,
    show_grid: bool = False,
    show: bool = True,
    output_path: Optional[Path] = None,
    # notación científica en ejes lineales
    y_sci: bool = True,
    soa_y_sci: bool = True,
    sci_limits: tuple[int, int] = (-3, 3),
    use_mathtext: bool = True,
    # layout para que no se corte el eje derecho
    right_margin: float = 0.84,
    spine_outward_pts: float = 8.0,
) -> None:
    """
    X: states
    Y izq: promedio total_time (líneas con marcadores)
    Y der: SoA (línea discontinua)
    Requiere columnas: ['family','states','total_time','soa'].
    """
    # 1) Leer y concatenar
    dfs = [pd.read_csv(p) for p in csv_paths]
    data = pd.concat(dfs, ignore_index=True)

    # limpiar cabeceras y deduplicar
    data.rename(columns=lambda c: str(c).strip(), inplace=True)
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated(keep="first")]

    # 2) Validar columnas
    required = {"family", "states", "total_time", "soa"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)}; disponibles: {list(data.columns)}")

    # 3) Tipos y limpieza
    data["family"] = data["family"].astype(str).str.strip()
    for col in ("states", "total_time", "soa"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["family", "states", "total_time", "soa"])

    # 4) Agrupar por familia y estado
    grouped = (
        data.groupby(["family", "states"], as_index=False)
            .agg(total_time=("total_time", "mean"),
                 soa=("soa", "mean"))
            .sort_values(["family", "states"])
    )

    # 5) Colores
    families = grouped["family"].unique().tolist()
    if family_colors is None:
        cmap = plt.cm.get_cmap(palette, len(families))
        family_colors = {fam: cmap(i) for i, fam in enumerate(families)}

    # 6) Plot
    fig, ax_time = plt.subplots()
    ax_soa = ax_time.twinx()
    ax_soa.spines["right"].set_position(("outward", spine_outward_pts))  # evita solape

    for fam in families:
        sub = grouped[grouped["family"] == fam].sort_values("states")
        if sub.empty:
            continue
        color = family_colors.get(fam, "#888888")

        # total_time (izq)
        ax_time.plot(
            sub["states"], sub["total_time"],
            marker="o", markersize=marker_size,
            linewidth=line_width, linestyle="-",
            color=color, label=fam,
        )
        # SoA (der)
        ax_soa.plot(
            sub["states"], sub["soa"],
            linestyle="--", linewidth=1.0,
            color=color, alpha=0.9, label="_nolegend_",
        )

    # 7) Ejes y estética
    ax_time.set_xlabel("CGS states (|S|)")
    ax_time.set_ylabel("Average total time (s)")
    ax_soa.set_ylabel("SoA (|τ| · |φ|)", labelpad=10)
    ax_time.legend(title="family", loc="best", frameon=True)
    if show_grid:
        ax_time.grid(True, which="major", alpha=0.3)

    # notación científica en ejes lineales
    if y_sci:
        _apply_scientific(ax_time.yaxis, limits=sci_limits, use_mathtext=use_mathtext, side="left")
    if soa_y_sci:
        _apply_scientific(ax_soa.yaxis, limits=sci_limits, use_mathtext=use_mathtext, side="right")

    # margen para el eje derecho y ajuste general
    fig.subplots_adjust(right=right_margin)
    plt.tight_layout()

    # 8) Salida
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()


def _style_right_axis(
    ax,
    *,
    spine_outward_pts: float = 10.0,
    tick_pad: int = 6,
) -> None:
    """Estética del eje Y derecho.
    - Desplaza la *spine* derecha hacia afuera para dejar hueco.
    - Fuerza ticks/labels a la derecha.
    - Aumenta el `pad` para separar números del eje.
    No toca *formatters* (sirve para log o linear).
    """
    ax.spines["right"].set_position(("outward", spine_outward_pts))
    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", which="both", pad=tick_pad)


def plot_families_states_vs_totaltime_with_soa_log(
    csv_paths: list[Path],
    family_colors: dict[str, object] | None = None,
    palette: str = "tab20",
    marker_size: int = 4,
    line_width: float = 1.2,
    show_grid: bool = False,
    show: bool = True,
    output_path: Optional[Path] = None,
    # --- escalas log ---
    x_log: bool = True,
    x_log_base: float = 2.0,          # útil si |S|=2^n
    y_log: bool = True,
    y_log_base: float = 10.0,
    soa_y_log: bool = True,
    soa_y_log_base: float = 10.0,
    min_positive_xeps: float = 1e-12,
    min_positive_yeps: float = 1e-12,
    # --- estética eje derecho / layout ---
    spine_outward_pts: float = 8.0,   # desplaza la spine derecha hacia fuera
    right_tick_pad: int = 6,          # separación números-eje derecho
    right_margin: float = 0.86,       # reserva ancho para label derecho
    right_labelpad: float = 10.0,     # padding del ylabel derecho
    # --- imprimir pendientes (ley de potencia) ---
    print_loglog_slopes: bool = True,
) -> None:
    """
    X: states (opcional log base x_log_base).
    Y izq: media de total_time (log base y_log_base).
    Y der: SoA (log base soa_y_log_base).
    Requiere columnas: ['family','states','total_time','soa'].

    Notas de layout:
    - `spine_outward_pts` mueve la spine derecha para dar hueco a ticks.
    - `right_margin` reserva margen en la figura para que el *ylabel* derecho no quede fuera.
    - No se usa `tight_layout` para no eliminar el margen reservado.
    """
    # 1) leer/concatenar
    dfs = [pd.read_csv(p) for p in csv_paths]
    data = pd.concat(dfs, ignore_index=True)
    if data.columns.duplicated().any():
        data = data.loc[:, ~data.columns.duplicated(keep="first")]

    required = {"family", "states", "total_time", "soa"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas {sorted(missing)}; disponibles: {list(data.columns)}")

    # 2) tipado/limpieza
    data["family"] = data["family"].astype(str).str.strip()
    for col in ("states", "total_time", "soa"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["family", "states", "total_time", "soa"])

    grouped = (
        data.groupby(["family", "states"], as_index=False)
        .agg(total_time=("total_time", "mean"), soa=("soa", "mean"))
    )

    families = sorted(grouped["family"].unique())
    if family_colors is None:
        cmap = plt.cm.get_cmap(palette, len(families))
        family_colors = {fam: cmap(i) for i, fam in enumerate(families)}

    # 3) plot
    fig, ax_time = plt.subplots()
    ax_soa = ax_time.twinx()

    # mover spine derecha y asegurar ticks/labels a la derecha
    ax_soa.spines["right"].set_position(("outward", spine_outward_pts))
    ax_soa.yaxis.set_ticks_position("right")
    ax_soa.yaxis.set_label_position("right")
    ax_soa.tick_params(axis="y", which="both", pad=right_tick_pad)

    for fam in families:
        sub = grouped[grouped["family"] == fam].sort_values("states")
        if sub.empty:
            continue
        ax_time.plot(
            sub["states"], sub["total_time"],
            marker="o", markersize=marker_size,
            linewidth=line_width, linestyle="-",
            color=family_colors.get(fam, "#888"),
            label=fam,
        )
        if (sub["soa"] > 0).any():
            ax_soa.plot(
                sub["states"], sub["soa"],
                linestyle="--", linewidth=1.0,
                color=family_colors.get(fam, "#888"),
                alpha=0.9, label="_nolegend_",
            )

    # 4) escalas log
    if x_log:
        pos_x = grouped.loc[grouped["states"] > 0, "states"]
        xmin = max(min_positive_xeps, pos_x.min()*0.9) if not pos_x.empty else min_positive_xeps
        ax_time.set_xscale("log", base=x_log_base)
        ax_time.set_xlim(left=xmin)

    if y_log:
        pos_t = grouped.loc[grouped["total_time"] > 0, "total_time"]
        ymin = max(min_positive_yeps, pos_t.min()*0.8) if not pos_t.empty else min_positive_yeps
        ax_time.set_yscale("log", base=y_log_base)
        ax_time.set_ylim(ymin, None)

    if soa_y_log:
        pos_s = grouped.loc[grouped["soa"] > 0, "soa"]
        ymin_soa = max(min_positive_yeps, pos_s.min()*0.8) if not pos_s.empty else min_positive_yeps
        ax_soa.set_yscale("log", base=soa_y_log_base)
        ax_soa.set_ylim(ymin_soa, None)

    # 5) etiquetas
    x_lab = "CGS states (|S|)"
    if x_log:
        x_lab += f" [log base {int(x_log_base)}]"
    ax_time.set_xlabel(x_lab)
    ax_time.set_ylabel(f"Total time (s){' [log base '+str(int(y_log_base))+']' if y_log else ''}")
    ax_soa.set_ylabel(
        f"SoA (|τ|·|φ|){' [log base '+str(int(soa_y_log_base))+']' if soa_y_log else ''}",
        labelpad=right_labelpad,
    )

    ax_time.legend(title="family", loc="best", frameon=True)
    if show_grid:
        ax_time.grid(True, which="both", alpha=0.3)

    # 6) reservar margen derecho para capturar el ylabel del eje derecho
    fig.subplots_adjust(right=right_margin)

    # 7) salida
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    # 8) (opcional) pendientes en log–log
    if print_loglog_slopes:
        print("\n[log–log slopes] y ≈ A * x^k  (para total_time)")
        for fam in families:
            sub = grouped[grouped["family"] == fam][["states","total_time"]].dropna()
            sub = sub[(sub["states"] > 0) & (sub["total_time"] > 0)]
            if len(sub) < 2:
                print(f"  {fam}: insuficientes puntos")
                continue
            lx = np.log(sub["states"].to_numpy())
            ly = np.log(sub["total_time"].to_numpy())
            k, b = np.polyfit(lx, ly, 1)
            A = np.exp(b)
            print(f"  {fam}: k ≈ {k:.3f},  A ≈ {A:.3g}")


def plot_percentage_time_breakdown_by_acg_states2(
    csv_path: str,
    agg: str = "mean",
    cgs_filter: list[str] | None = None,
    figsize: tuple[int, int] = (10, 4),
    save_as: str | None = None,
    bins: int | None = None,
    acg_min: float | None = None,
    acg_max: float | None = None,
    x_tick_step: float | None = None,  # distancia entre ticks
    x_max_ticks: int | None = 10,      # nº máximo de ticks si no se especifica step
):
    """
    Plot stacked bar chart of percentage time breakdown by acg_states.
    If `bins` is specified, group acg_states into that many equal-width bins.
    Can also filter by acg_states range using acg_min and acg_max.

    Tick policy:
    - If `x_tick_step` is set → fixed step on a numeric axis.
    - Else use `MaxNLocator(nbins=x_max_ticks)` for a clean axis.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    df = pd.read_csv(csv_path)

    if cgs_filter and 'cgs_id' in df.columns:
        df = df[df['cgs_id'].isin(cgs_filter)].copy()

    if 'acg_states' in df.columns:
        if acg_min is not None:
            df = df[df['acg_states'] >= acg_min]
        if acg_max is not None:
            df = df[df['acg_states'] <= acg_max]

    phases = ["gen_time", "acg_build_time", "game_build_time", "solve_time"]

    if bins and 'acg_states' in df.columns:
        # Bins y posiciones numéricas por el centro del bin
        df['acg_bin'] = pd.cut(df['acg_states'], bins=bins)
        group_key = 'acg_bin'
    else:
        group_key = 'acg_states'

    grouped = (
        df.groupby(group_key)[phases]
          .agg(agg)
          .reset_index()
          .sort_values(group_key)
    )

    sums = grouped[phases].sum(axis=1)
    grouped[phases] = (grouped[phases].div(sums, axis=0) * 100)

    # Construcción del eje X numérico y anchos de barra
    if group_key == 'acg_bin':
        intervals = grouped['acg_bin'].astype('interval[float64]').to_list()
        lefts = np.array([iv.left for iv in intervals])
        rights = np.array([iv.right for iv in intervals])
        x_vals = (lefts + rights) / 2.0
        widths = (rights - lefts) * 0.95  # dejar un 5% de espacio
        x_min, x_max = lefts.min(), rights.max()
    else:
        x_vals = grouped['acg_states'].astype(float).to_numpy()
        if len(x_vals) > 1:
            diffs = np.diff(np.sort(x_vals))
            base = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 1.0
        else:
            base = 1.0
        widths = np.full_like(x_vals, base * 0.9, dtype=float)
        x_min, x_max = (np.min(x_vals), np.max(x_vals)) if len(x_vals) else (0.0, 1.0)

    colors = {
        "gen_time":        "#FFC857",
        "acg_build_time":  "#F55D3E",
        "game_build_time": "#3E7CB1",
        "solve_time":      "#2AB7CA"
    }

    pretty = {
        "gen_time":        "preprocessing",
        "acg_build_time":  "ACG construction",
        "game_build_time": "acceptance game",
        "solve_time":      "solver"
    }

    plt.figure(figsize=figsize)
    bottom = np.zeros(len(grouped))

    # Barras apiladas con eje X continuo
    for phase in phases:
        vals = grouped[phase].to_numpy()
        plt.bar(x_vals, vals,
                bottom=bottom,
                width=widths,
                color=colors[phase],
                edgecolor="white",
                label=pretty[phase],
                align='center')
        bottom += vals

    # Estética
    plt.xlabel("closure")
    plt.ylabel("Percentage of total time (%)")
    plt.ylim(0, 105)

    # Ticks del eje X limpios
    ax = plt.gca()
    ax.set_xlim(x_min, x_max)
    if x_tick_step is not None and x_tick_step > 0:
        start = np.floor(x_min / x_tick_step) * x_tick_step
        end = np.ceil(x_max / x_tick_step) * x_tick_step
        ax.set_xticks(np.arange(start, end + 1e-9, x_tick_step))
    else:
        if x_max_ticks is not None and x_max_ticks > 0:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=x_max_ticks, prune=None))
        else:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))

    plt.grid(axis="y", alpha=0.3)
    plt.legend(ncol=1, frameon=False,
               loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()




# ================================================================
# 7 · MAIN
# ================================================================



if __name__ == "__main__":   


    plot_percentage_time_breakdown_by_acg_states2(Path("(1to4) From 2-10 pipeline entera con soa.csv"),acg_min=0,acg_max=200,bins=40)
    plot_percentage_time_breakdown_by_acg_states2(Path("(1to4) From 2-10 pipeline entera con soa.csv"),acg_min=100,acg_max=200,bins=20)

    plot_totaltime_vs_acgstates_with_soa_only_np_cl_final([Path("(1to4) From 2-10 pipeline entera con soa.csv")])
    plot_totaltime_vs_acgstates_with_soa_only_drone_robot_final([Path("(1to4) From 2-10 pipeline entera con soa.csv")])

    csv_files_all = [
        Path("flatX_lights_NUEVO - hasta6.csv"),
        Path("flatG_lights_NUEVO.csv"),
        Path("flatU_lights_NUEVO.csv"),
        Path("negflatX_lights_NUEVO - hasta6.csv"),
        Path("negflatG_lights_NUEVO.csv"),
        Path("negflatU_lights_NUEVO.csv"),
        Path("nestedX_lights.csv"),
        Path("nestedG_lights_NUEVO2.csv"),
        Path("nestedU_lights_NUEVO.csv"),
        Path("neg1nestedX_lights_NUEVO_final.csv"),
        Path("neg2nestedG_lights_NUEVO.csv"),
        Path("neg2nestedU_lights_NUEVO.csv"),
    ]

    fam_colors = make_family_color_map(csv_paths=csv_files_all, palette="tab20")

    # (opcional) validador para evitar sorpresas
    def _assert_colors_cover(csv_paths, fam_colors):
        import pandas as pd
        fams = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)["family"]
        fams = set(fams.astype(str).str.strip().unique())
        missing = fams - set(fam_colors.keys())
        if missing:
            raise ValueError(f"Faltan colores para familias: {sorted(missing)}")

    # --- primer subset ---
    csv_files2 = [
        Path("flatX_lights_NUEVO - hasta6.csv"),
        Path("flatG_lights_NUEVO.csv"),
        Path("flatU_lights_NUEVO.csv"),
        Path("nestedX_lights.csv"),
        Path("nestedG_lights_NUEVO2.csv"),
        Path("nestedU_lights_NUEVO.csv"),
    ]
    _assert_colors_cover(csv_files2, fam_colors)

    plot_families_states_vs_totaltime_with_soa_linear(
        csv_paths=csv_files2,
        family_colors=fam_colors,  # << usa SIEMPRE el mismo mapping
        show_grid=False, y_sci=True, soa_y_sci=True,
    )

    plot_families_states_vs_totaltime_with_soa_log(  # o tu versión log–log
        csv_paths=csv_files2,
        family_colors=fam_colors,  # << NUNCA None si quieres coherencia
        x_log=True, x_log_base=2,
        y_log=True, y_log_base=10,
        soa_y_log=True, soa_y_log_base=10,
        print_loglog_slopes=True,
    )

    # --- segundo subset (negadas) ---
    csv_files3 = [
        Path("negflatX_lights_NUEVO - hasta6.csv"),
        Path("negflatG_lights_NUEVO.csv"),
        Path("negflatU_lights_NUEVO.csv"),
        Path("neg1nestedX_lights_NUEVO_final.csv"),
        Path("neg2nestedG_lights_NUEVO.csv"),
        Path("neg2nestedU_lights_NUEVO.csv"),
    ]
    _assert_colors_cover(csv_files3, fam_colors)

    plot_families_states_vs_totaltime_with_soa_linear(
        csv_paths=csv_files3,
        family_colors=fam_colors,
        show_grid=False, y_sci=True, soa_y_sci=True,
    )

    plot_families_states_vs_totaltime_with_soa_log(
        csv_paths=csv_files3,
        family_colors=fam_colors,  # << coherente
        x_log=True, x_log_base=2,
        y_log=True, y_log_base=10,
        soa_y_log=True, soa_y_log_base=10,
        print_loglog_slopes=True,
    )