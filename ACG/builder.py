from __future__ import annotations
import time
from copy import deepcopy
from preprocessing.ast_nodes import ParseNode, T, F, Var, And, Or, Not, Next, Globally, Eventually, Until, Modality, DualModality, Conj, Disj, Top, Bottom
from preprocessing.transformer import push_negations_to_nnf
from .model import ACG, EpsilonAtom, ExistentialAtom, UniversalAtom

def _is_atomic_state(state):
    return isinstance(state, Var) or (isinstance(state, Not) and isinstance(state.sub, Var))

def _delta_atomic(state, sigma):
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
    WILDCARD = acg.WILDCARD
    for state in acg.states:
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
        elif isinstance(state, Modality) and isinstance(state.sub, Next):
            next_state = state.sub.sub
            agents = frozenset(state.agents)
            acg.add_transition(state, sigma, UniversalAtom(next_state, agents))
        elif isinstance(state, Not) and isinstance(state.sub, Modality) and isinstance(state.sub.sub, Next):
            inner = state.sub.sub.sub
            neg_inner = push_negations_to_nnf(Not(inner))
            agents = frozenset(state.sub.agents)
            Omega = cgs.agents
            Agentsbuenos = Omega - agents
            acg.add_transition(state, sigma, ExistentialAtom(neg_inner, Agentsbuenos))
        elif isinstance(state, Modality) and isinstance(state.sub, Globally):
            phi = state.sub.sub
            agents = frozenset(state.agents)
            acg.add_transition(state, sigma, Conj(EpsilonAtom(phi), UniversalAtom(state, agents)))
        elif isinstance(state, Not) and isinstance(state.sub, Modality) and isinstance(state.sub.sub, Globally):
            phi = state.sub.sub.sub
            neg_phi = push_negations_to_nnf(Not(phi))
            agents = frozenset(state.sub.agents)
            Omega = cgs.agents
            Agentsbuenos = Omega - agents
            acg.add_transition(state, sigma, Disj(EpsilonAtom(neg_phi), ExistentialAtom(state, Agentsbuenos)))
        elif isinstance(state, Modality) and isinstance(state.sub, Until):
            phi1, phi2 = state.sub.lhs, state.sub.rhs
            agents = frozenset(state.agents)
            acg.add_transition(
                state, sigma,
                Disj(EpsilonAtom(phi2), Conj(EpsilonAtom(phi1), UniversalAtom(state, agents)))
            )
        elif isinstance(state, Not) and isinstance(state.sub, Modality) and isinstance(state.sub.sub, Until):
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
    ap_set = extract_propositions(transformed_ast)
    acg.propositions = ap_set
    if materialize_alphabet:
        acg.generate_alphabet()
    else:
        acg.alphabet = set()
    closure = generate_closure(transformed_ast)
    acg.states = closure
    acg.initial_state = transformed_ast
    for node in closure:
        if isinstance(node, Modality) and isinstance(node.sub, Globally):
            acg.final_states.add(node)
        elif isinstance(node, Not) and isinstance(node.sub, Modality) and isinstance(node.sub.sub, Until):
            acg.final_states.add(node)
    generate_transitions_final(acg, cgs)
    _orig_get = acg.get_transition
    def _get_transition_monkey(self, state, sigma):
        if _is_atomic_state(state):
            return _delta_atomic(state, sigma)
        return _orig_get(state, sigma)
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