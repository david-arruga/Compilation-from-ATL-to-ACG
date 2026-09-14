import itertools
import unittest
from acg import CGS, ACG, EpsilonAtom, UniversalAtom
from preprocessing import Var, Top, Bottom, Conj, Disj
from acceptance_game import build_game
from acceptance_game.utils import generate_possibilities, evaluate_boolean_formula
from buchi_solver import solve_buchi_game


def model():
    g = CGS()
    g.add_agent('a'); g.add_decisions('a', {'stay'})
    g.set_initial_state('s'); g.label_state('s', set())
    g.add_transition('s', [('a', 'stay')], 's')
    return g


class ArenaContractTests(unittest.TestCase):
    def test_invalid_cgs_fail_with_validation_error(self):
        changes = [lambda g: g.decisions.clear(),
                   lambda g: g.decisions.update(a=set()),
                   lambda g: g.transition_function.clear(),
                   lambda g: g.transition_function.update({('outside', frozenset({('a','stay')})): 's'}),
                   lambda g: g.transition_function.update({('s', frozenset({('a','illegal')})): 's'}),
                   lambda g: g.transition_function.update({('s', frozenset({('a','stay'),('a','illegal')})): 's'}),
                   lambda g: g.labeling_function.update(s={'unknown'}),
                   lambda g: g.labeling_function.clear()]
        for change in changes:
            g = model(); change(g)
            with self.assertRaises(ValueError): g.validate()

    def test_conflicting_successors_cannot_be_overwritten(self):
        g = model()
        with self.assertRaises(ValueError):
            g.add_transition('s', [('a', 'stay')], 'other')
        self.assertEqual(g.get_successor('s', {'a':'stay'}), 's')
        self.assertEqual(g.states, {'s'})
        with self.assertRaises(ValueError): g.get_successor('s', {'a':'missing'})

    def test_reference_models_are_total(self):
        from acceptance_game.examples import cgs1, cgs2, cgs3, cgs4
        for g in (cgs1, cgs2, cgs3, cgs4):
            g.validate()
        self.assertEqual(cgs2.decisions['PedLight'], {'walk', 'dontWalk'})

    def test_unreachable_states_are_allowed(self):
        g = model(); g.add_state('u'); g.label_state('u',set())
        g.add_transition('u', [('a','stay')], 'u')
        g.validate()
        with self.assertRaises(ValueError): g.validate(check_reachability=True)

    def test_supports_match_boolean_truth(self):
        a, b = EpsilonAtom(Var('a')), EpsilonAtom(Var('b'))
        leaves = [Top(), Bottom(), a, b]
        depth1 = leaves + [op(x,y) for op in (Conj,Disj) for x in leaves for y in leaves]
        formulas = [op(x,y) for op in (Conj,Disj) for x in depth1 for y in leaves]
        for f in formulas:
            supports = generate_possibilities(f)
            for bits in itertools.product((False, True), repeat=2):
                assignment = frozenset(x for x, yes in zip((a,b),bits) if yes)
                self.assertEqual(any(s <= assignment for s in supports),
                                 evaluate_boolean_formula(f, assignment))
        with self.assertRaises(ValueError): generate_possibilities('not a transition')

    def test_nested_constants_generate_total_correct_games(self):
        # q loops through epsilon and is not Büchi accepting, hence is losing.
        q = Var('q'); atom = EpsilonAtom(q)
        cases = [(Conj(Top(),atom), False), (Disj(Top(),atom), True),
                 (Conj(Bottom(),atom), False), (Disj(Bottom(),atom), False),
                 (Conj(Top(),Top()), True), (Disj(Bottom(),Bottom()),False)]
        for delta, expected in cases:
            a = ACG(); a.states={q}; a.initial_state=q
            a.add_transition(q, None, delta)
            V,E,S1,S2,B,initial = build_game(a,model())
            self.assertTrue(all(any(src == v for src,dst in E) for v in V))
            win,_ = solve_buchi_game(V,E,S1,S2,B)
            self.assertEqual(initial in win,expected)

    def test_arena_checks_inputs_and_atom_targets(self):
        q = Var('q'); a = ACG(); a.states={q}; a.initial_state=q
        a.add_transition(q,None,Top())
        g = model(); g.transition_function.clear()
        with self.assertRaises(ValueError): build_game(a,g)
        for atom in (EpsilonAtom(Var('missing')),UniversalAtom(q,{'unknown'})):
            a.add_transition(q,None,atom)
            with self.assertRaises(ValueError): build_game(a,model())
        a.initial_state=Var('missing')
        with self.assertRaises(ValueError): build_game(a,model())


if __name__ == '__main__': unittest.main()
