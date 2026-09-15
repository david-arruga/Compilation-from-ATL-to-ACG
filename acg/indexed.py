"""Occurrence-indexed compiler for normalized ATL; no structural AST hashing.

Each state-formula occurrence has positive/negative IDs 2*i and 2*i+1.
Repeated equal formulas may have different IDs. Compilation does not render
or reconstruct formula labels; decode_labels is a separate diagnostic operation.
"""
from .model import ACG, EpsilonAtom, UniversalAtom, ExistentialAtom
from preprocessing.ast_nodes import (Var, T, F, Not, And, Or, Modality,
                                    Next, Globally, Until, Top, Bottom, Conj, Disj)


class StateID(int):
    def to_formula(self):
        return f'q{int(self)}'

    def __str__(self):
        return self.to_formula()


class IndexedACG(ACG):
    def __init__(self):
        super().__init__()
        self.records = []
        self.input_visits = 0

    def get_transition(self, state, input_symbol):
        if state not in self.states:
            raise ValueError('Unknown indexed ACG state.')
        kind, data, children = self.records[int(state)//2]
        if kind == 'var':
            truth = (data in input_symbol) != bool(int(state) % 2)
            return Top() if truth else Bottom()
        return super().get_transition(state, input_symbol)

    def decode_labels(self):
        """Reconstruct labels outside compilation; never used by the game.

        Records are postorder. Sharing here avoids copying children, but later
        hashing/rendering these diagnostic ASTs can still be expensive.
        """
        labels = []
        for kind, data, children in self.records:
            args = [labels[int(c)] for c in children]
            negargs = [labels[int(c)^1] for c in children]
            if kind == 'var': pos, neg = Var(data), Not(Var(data))
            elif kind == 'true': pos, neg = T(), F()
            elif kind == 'false': pos, neg = F(), T()
            elif kind == 'and': pos, neg = And(*args), Or(*negargs)
            elif kind == 'or': pos, neg = Or(*args), And(*negargs)
            else:
                path = {'next':Next, 'globally':Globally, 'until':Until}[kind](*args)
                pos = Modality(sorted(data), path)
                neg = Not(pos)
            labels.extend((pos, neg))
        return labels


def build_acg_indexed(ast, cgs, materialize_alphabet=False):
    """Compile a normalized state AST. Cost O(N*(1+|agents|)) in a unit-cost
    reference/container model; fixed agents give O(N). No CPython time theorem.
    Explicit alphabet enumeration is optional and excluded from that bound.
    """
    out = IndexedACG()
    universe = frozenset(cgs.agents)
    tasks, results = [('visit', ast)], []
    while tasks:
        task = tasks.pop()
        if task[0] == 'flip':
            results[-1] = StateID(int(results[-1]) ^ 1)
            continue
        if task[0] == 'finish':
            _, kind, data, count = task
            children = tuple(results[-count:]) if count else ()
            if count: del results[-count:]
            idx = len(out.records)
            out.records.append((kind, data, children))
            results.append(StateID(2*idx))
            continue
        node = task[1]
        out.input_visits += 1
        if isinstance(node, Not):
            if not isinstance(node.sub, (Var, Modality)):
                raise ValueError('Expected normalized negation before literal or modality.')
            tasks.extend([('flip',), ('visit', node.sub)])
            continue
        data, children = None, ()
        if isinstance(node, Var):
            kind, data = 'var', node.name
            out.propositions.add(data)
        elif isinstance(node, T): kind = 'true'
        elif isinstance(node, F): kind = 'false'
        elif isinstance(node, (And, Or)):
            kind = 'and' if isinstance(node, And) else 'or'
            children = (node.lhs, node.rhs)
        elif isinstance(node, Modality):
            if not isinstance(node.agents, (list, tuple, set, frozenset)) or not all(
                isinstance(a,str) for a in node.agents):
                raise ValueError('Invalid coalition.')
            data = frozenset(node.agents)
            if not data <= universe: raise ValueError('Unknown coalition agent.')
            path = node.sub
            if isinstance(path, Next): kind, children = 'next', (path.sub,)
            elif isinstance(path, Globally): kind, children = 'globally', (path.sub,)
            elif isinstance(path, Until): kind, children = 'until', (path.lhs,path.rhs)
            else: raise ValueError('Unsupported normalized strategic form.')
            out.input_visits += 1  # the temporal AST wrapper
        else:
            raise ValueError('Unsupported normalized ATL node: '+type(node).__name__)
        tasks.append(('finish',kind,data,len(children)))
        tasks.extend(('visit',child) for child in reversed(children))
    out.states = {StateID(i) for i in range(2*len(out.records))}
    out.initial_state = results[0]
    eps, box, dia = EpsilonAtom, UniversalAtom, ExistentialAtom
    for i,(kind,data,ch) in enumerate(out.records):
        p, n = StateID(2*i), StateID(2*i+1)
        if kind == 'var': continue
        neg = tuple(StateID(int(c)^1) for c in ch)
        if kind == 'true': dp,dn = Top(),Bottom()
        elif kind == 'false': dp,dn = Bottom(),Top()
        elif kind == 'and': dp,dn = Conj(*(eps(c) for c in ch)),Disj(*(eps(c) for c in neg))
        elif kind == 'or': dp,dn = Disj(*(eps(c) for c in ch)),Conj(*(eps(c) for c in neg))
        else:
            complement = universe-data
            if kind == 'next': dp,dn = box(ch[0],data),dia(neg[0],complement)
            elif kind == 'globally':
                dp,dn = Conj(eps(ch[0]),box(p,data)),Disj(eps(neg[0]),dia(n,complement))
                out.final_states.add(p)
            else:
                dp = Disj(eps(ch[1]),Conj(eps(ch[0]),box(p,data)))
                dn = Conj(eps(neg[1]),Disj(eps(neg[0]),dia(n,complement)))
                out.final_states.add(n)
        out.add_transition(p,None,dp)
        out.add_transition(n,None,dn)
    if materialize_alphabet: out.generate_alphabet()
    return out
