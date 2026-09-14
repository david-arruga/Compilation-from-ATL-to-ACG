import unittest
from acg import ACG, EpsilonAtom, UniversalAtom
from preprocessing import Var, Top, Conj, Disj, normalize_formula, parse, tokenize
from acceptance_game import build_game
from acceptance_game.utils import generate_possibilities, pretty_node
from acceptance_game.examples import cgs1
from acg import build_acg_final
from buchi_solver import solve_buchi_game
from test_arena_contract import model


class PresentationTests(unittest.TestCase):
    def test_supports_are_minimal_not_just_satisfying(self):
        a,b=EpsilonAtom(Var('a')),EpsilonAtom(Var('b'))
        self.assertEqual(set(generate_possibilities(Disj(a,Conj(a,b)))),{frozenset({a})})
        self.assertEqual(generate_possibilities(Disj(Top(),a)),[frozenset()])

    def test_source_and_support_are_part_of_identity(self):
        a=ACG(); q,r,t=map(Var,('q','r','t'))
        a.states={q,r,t};a.initial_state=q
        shared=UniversalAtom(t,{'a'})
        a.add_transition(q,None,Disj(Conj(shared,EpsilonAtom(r)),Conj(shared,EpsilonAtom(t))))
        a.add_transition(r,None,shared);a.add_transition(t,None,Top())
        V,E,*_=build_game(a,model(),full_arena=True)
        atoms=[v for v in V if isinstance(v,tuple) and v[0]=='atom_applied']
        self.assertEqual(len(atoms),3)
        self.assertEqual(len({pretty_node(v) for v in atoms}),3)
        for node in atoms:
            _,source,s,H,atom=node
            self.assertIn((("atom_selection",source,s,H),node),E)
            self.assertIn(atom,H)

    def test_full_and_reachable_graphs_agree_exactly_on_reachable_vertices(self):
        formulas=['<Valve> globally (underpowered implies <Reactor> next efficient)',
                  'not <Reactor> (safe until efficient)', '<> G safe']
        for formula in formulas:
            a=build_acg_final(normalize_formula(parse(tokenize(formula))),cgs1)
            small=build_game(a,cgs1)
            full=build_game(a,cgs1,full_arena=True)
            V,E,A,R,B,initial=full
            reached={initial};todo=[initial]
            while todo:
                v=todo.pop()
                for src,dst in E:
                    if src==v and dst not in reached:
                        reached.add(dst);todo.append(dst)
            kept=reached|{'true_sink','false_sink'}
            self.assertEqual(small[0],kept)
            self.assertEqual(small[1],{edge:val for edge,val in E.items() if edge[0] in kept})
            self.assertEqual(small[2],A&kept)
            self.assertEqual(small[3],R&kept)
            self.assertEqual(small[4],B&kept)
            wf,_=solve_buchi_game(*full[:5]);ws,_=solve_buchi_game(*small[:5])
            self.assertEqual(ws,wf&kept)
            configurations={v for v in V if isinstance(v,tuple) and v[0]=='state'}
            self.assertEqual(configurations,{('state',q,s) for q in a.states for s in cgs1.states})


if __name__=='__main__':unittest.main()
