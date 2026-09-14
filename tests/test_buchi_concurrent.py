import itertools
import random
import unittest
from buchi_solver import solve_buchi_game
from preprocessing import Var, Globally, Until, Not, Modality, normalize_formula
from acg import CGS, build_acg_final
from acceptance_game import build_game
from test_parser_temporal import buchi_oracle


class BuchiTests(unittest.TestCase):
    def test_removed_vertices_cannot_witness_an_attractor(self):
        V = {0, 1, 2, 3}
        E = {edge: None for edge in [(0, 0), (1, 0), (2, 1), (2, 2), (3, 0), (3, 2)]}
        # B={1,3}: both can be visited only finitely, along every play.
        self.assertEqual(solve_buchi_game(V, E, {2, 3}, {0, 1}, {1, 3}), (set(), V))

    def test_all_total_three_vertex_arenas(self):
        V = {0, 1, 2}
        subsets = [{v for v in V if m >> v & 1} for m in range(8)]
        for succ in itertools.product(subsets[1:], repeat=3):
            E = {(v, w): None for v in V for w in succ[v]}
            for accept in subsets:
                for B in subsets:
                    actual, losing = solve_buchi_game(V, E, accept, V-accept, B)
                    expected = {v for v in V if buchi_oracle((V, E, accept, V-accept, B, v))}
                    self.assertEqual(actual, expected)
                    self.assertEqual(losing, V-expected)

    def test_invalid_arenas_and_boundary_conditions(self):
        for args in [({0}, {}, {0}, set(), set()),
                     ({0}, {(0, 0): None}, {0}, {0}, set()),
                     ({0}, {(0, 1): None}, {0}, set(), set()),
                     ({0}, {(0, 0): None}, {0}, set(), {1})]:
            with self.assertRaises(ValueError): solve_buchi_game(*args)
        self.assertEqual(solve_buchi_game(set(), {}, set(), set(), set()), (set(), set()))
        E = {(0, 0): None}
        self.assertEqual(solve_buchi_game({0}, E, {0}, set(), {0}), ({0}, set()))
        self.assertEqual(solve_buchi_game({0}, E, {0}, set(), set()), (set(), {0}))


class ConcurrentTemporalTests(unittest.TestCase):
    def test_gu_against_direct_atl_fixed_points(self):
        # 64 distinct sampled two-state CGSs, all four coalitions,
        # both roots, G/U and their negations: 2048 comparisons.
        rng = random.Random(20260914)
        actions = list(itertools.product((0, 1), repeat=2))
        maps = rng.sample(range(256), 64)
        for encoded in maps:
            transition = {(s, joint): (encoded >> (4*s+i)) & 1
                          for s in (0, 1) for i, joint in enumerate(actions)}
            labels = [{p for p in ('p', 'q') if rng.randrange(2)} for _ in range(2)]
            g = CGS()
            for a in ('a', 'b'):
                g.add_agent(a); g.add_decisions(a, {0, 1})
            for p in ('p', 'q'): g.add_proposition(p)
            for s in (0, 1):
                g.add_state(s); g.label_state(s, labels[s])
                for joint in actions:
                    g.add_transition(s, list(zip(('a', 'b'), joint)), transition[s, joint])
            for coalition in ([], ['a'], ['b'], ['a', 'b']):
                indices = [('a', 'b').index(a) for a in coalition]
                choices = list(itertools.product((0, 1), repeat=len(indices)))
                def pre(X):
                    return {s for s in (0, 1) if any(
                        all(transition[s, joint] in X for joint in actions
                            if all(joint[i] == a for i, a in zip(indices, choice)))
                        for choice in choices)}
                P = {s for s in (0, 1) if 'p' in labels[s]}
                Q = {s for s in (0, 1) if 'q' in labels[s]}
                for path in (Globally(Var('p')), Until(Var('p'), Var('q'))):
                    winning = {0, 1} if isinstance(path, Globally) else set()
                    while True:
                        updated = P & pre(winning) if isinstance(path, Globally) else Q | (P & pre(winning))
                        if updated == winning: break
                        winning = updated
                    for neg in (False, True):
                        formula = Modality(coalition, path)
                        if neg: formula = Not(formula)
                        automaton = build_acg_final(normalize_formula(formula), g)
                        for root in (0, 1):
                            g.set_initial_state(root)
                            arena = build_game(automaton, g)
                            actual, _ = solve_buchi_game(*arena[:5])
                            expected = (root not in winning) if neg else (root in winning)
                            self.assertEqual(arena[5] in actual, expected,
                                             (encoded, labels, coalition, root, str(formula)))


if __name__ == '__main__': unittest.main()
