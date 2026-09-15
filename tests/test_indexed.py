import random
import unittest
from unittest.mock import patch
from preprocessing import (ParseNode, Var, Not, And, Or, Modality, Globally,
    Next, Until, T, F, Conj, Disj, Top, Bottom, normalize_formula,parse,tokenize)
from acg import build_acg_final,build_acg_indexed,EpsilonAtom,UniversalAtom,ExistentialAtom
from acceptance_game import build_game
from acceptance_game.examples import cgs1
from buchi_solver import solve_buchi_game
from test_arena_contract import model


def project(t,labels):
    if isinstance(t,EpsilonAtom): return EpsilonAtom(labels[int(t.state)])
    if isinstance(t,UniversalAtom): return UniversalAtom(labels[int(t.state)],t.agents)
    if isinstance(t,ExistentialAtom): return ExistentialAtom(labels[int(t.state)],t.agents)
    if isinstance(t,Conj): return Conj(project(t.lhs,labels),project(t.rhs,labels))
    if isinstance(t,Disj): return Disj(project(t.lhs,labels),project(t.rhs,labels))
    return t


class IndexedTests(unittest.TestCase):
    def test_transition_projection_and_acceptance(self):
        rng=random.Random(31)
        def gen(depth):
            if depth==0:return rng.choice([Var('p'),Var('q'),T(),F()])
            c=rng.randrange(7)
            if c<2:return (And if c==0 else Or)(gen(depth-1),gen(depth-1))
            if c==2:return Not(gen(depth-1))
            path=(Next(gen(depth-1)) if c==3 else Globally(gen(depth-1)) if c==4
                  else Until(gen(depth-1),gen(depth-1)))
            return Modality(rng.choice([[],['a']]),path)
        g=model()
        formulas=[normalize_formula(gen(3)) for _ in range(80)]
        formulas += [And(Var('p'),Var('p')),Not(Modality([],Globally(Var('p'))))]
        for f in formulas:
            old,new=build_acg_final(f,g),build_acg_indexed(f,g)
            labels=new.decode_labels()
            self.assertEqual(labels[int(new.initial_state)],f)
            self.assertEqual(set(labels),old.states)
            for q in new.states:
                label=labels[int(q)]
                self.assertEqual(q in new.final_states,label in old.final_states)
                for lab in (frozenset(),frozenset({'p'}),frozenset({'q'}),frozenset({'p','q'})):
                    self.assertEqual(project(new.get_transition(q,lab),labels),old.get_transition(label,lab))
            def wins(a):
                arena=build_game(a,g);w,_=solve_buchi_game(*arena[:5]);return arena[5] in w
            self.assertEqual(wins(old),wins(new))

    def test_no_ast_hash_or_copy_and_no_recursive_compilation(self):
        f=Var('p')
        for _ in range(5000):f=Modality(['a'],Globally(f))
        with (patch.object(ParseNode,'__hash__',side_effect=AssertionError('AST hash')),
             patch('acg.builder.deepcopy',side_effect=AssertionError('AST copy'))):
            a=build_acg_indexed(f,model())
        self.assertEqual(a.input_visits,10001)
        self.assertEqual(len(a.states),10002)

    def test_rejects_unsupported_input(self):
        for f in (Next(Var('p')),Modality(['unknown'],Next(Var('p'))),Not(And(Var('p'),Var('q')))):
            with self.assertRaises(ValueError):build_acg_indexed(f,model())

    def test_chapter4_indexed_and_reference(self):
        f=normalize_formula(parse(tokenize('<Valve> globally (underpowered implies <Reactor> next efficient)')))
        for constructor in (build_acg_final,build_acg_indexed):
            a=constructor(f,cgs1);arena=build_game(a,cgs1)
            w,_=solve_buchi_game(*arena[:5]);self.assertIn(arena[5],w)


if __name__=='__main__':unittest.main()
