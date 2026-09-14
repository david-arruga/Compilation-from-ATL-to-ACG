"""Run from repository root: python3 -m scripts.chapter4_trace.

Checks the manually transcribed chapter-4 example against current code and
prints reproducible JSON. These checks are not a universal proof of Python.
"""
import json
from copy import deepcopy
from itertools import product
from preprocessing import (parse, tokenize, normalize_formula, Var, Not, And,
                           Or, Next, Globally, Modality, Conj, Disj, Top, Bottom)
from acg import build_acg_final, EpsilonAtom, UniversalAtom, ExistentialAtom
from acceptance_game import build_game
from acceptance_game.examples import cgs1
from buchi_solver import solve_buchi_game

FORMULA = '<Valve> globally (underpowered implies <Reactor> next efficient)'


def trace():
    u, e = Var('underpowered'), Var('efficient')
    x = Modality(['Reactor'], Next(e))
    b = Or(Not(u), x)
    nb = And(u, Not(x))
    g = Modality(['Valve'], Globally(b))
    ast = normalize_formula(parse(tokenize(FORMULA)))
    assert ast == g
    aliases = {'g':g, 'not_g':Not(g), 'b':b, 'not_b':nb, 'x':x,
               'not_x':Not(x), 'u':u, 'not_u':Not(u), 'e':e, 'not_e':Not(e)}
    a = build_acg_final(ast,cgs1)
    assert a.states == set(aliases.values()) and a.final_states == {g}
    assert a.propositions == {'underpowered','efficient'}
    eps, box, dia = EpsilonAtom, UniversalAtom, ExistentialAtom
    expected = {
        g: Conj(eps(b),box(g,{'Valve'})),
        Not(g): Disj(eps(nb),dia(Not(g),{'Reactor'})),
        b: Disj(eps(Not(u)),eps(x)), nb: Conj(eps(u),eps(Not(x))),
        x:box(e,{'Reactor'}), Not(x):dia(Not(e),{'Valve'})}
    for bits in product((False,True),repeat=2):
        lab=frozenset(p for p,yes in zip(('underpowered','efficient'),bits) if yes)
        for q, delta in expected.items(): assert a.get_transition(q,lab)==delta
        for p in (u,e):
            for neg in (False,True):
                q=Not(p) if neg else p
                truth=(p.name not in lab) if neg else (p.name in lab)
                assert a.get_transition(q,lab)==(Top() if truth else Bottom())
    # Columns: heat/open, heat/lock, cool/open, cool/lock.
    rows={'s0':['s0','s1','s0','s0'], 's1':['s1','s3','s2','s2'],
          's2':['s2','s1','s0','s0'], 's3':['s3','s4','s2','s1'],
          's4':['s0','s0','s0','s0']}
    for s, destinations in rows.items():
        for (r,v),dst in zip(product(('heat','cool'),('open','lock')),destinations):
            assert cgs1.get_successor(s,{'Reactor':r,'Valve':v})==dst
    # The advertised strategy remains at start for either opposing move.
    assert 'underpowered' not in cgs1.labeling_function['s0']
    for r in ('heat','cool'):
        assert cgs1.get_successor('s0',{'Reactor':r,'Valve':'open'})=='s0'
    results={}
    summary=None
    for s in sorted(cgs1.states):
        model=deepcopy(cgs1); model.set_initial_state(s)
        V,E,A,R,B,initial=build_game(a,model)
        win,lose=solve_buchi_game(V,E,A,R,B)
        results[s]=initial in win
        if s=='s0':
            summary=dict(vertices=len(V),edges=len(E),accept_owned=len(A),
                         reject_owned=len(R),buchi_vertices=len(B),
                         winning_vertices=len(win),losing_vertices=len(lose))
    assert results=={'s0':True,'s1':False,'s2':False,'s3':False,'s4':True}
    return dict(formula=FORMULA,normalized=str(ast),states=len(a.states),
                accepting_states=['g'],state_aliases={k:str(v) for k,v in aliases.items()},
                alphabet=[[],['efficient'],['underpowered'],['efficient','underpowered']],
                transition_schemas={k:str(expected[q]) for k,q in aliases.items() if q in expected},
                note='Literal transitions checked on every valuation; alphabet remains symbolic in code.',
                game_from_start=summary,satisfaction_by_state=results)


if __name__=='__main__':
    print(json.dumps(trace(),ensure_ascii=False,indent=2,sort_keys=True))
