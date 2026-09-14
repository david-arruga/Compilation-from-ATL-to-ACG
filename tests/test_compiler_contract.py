import itertools
import unittest

from preprocessing import (parse, tokenize, normalize_formula, Var, T, F, Not,
    And, Or, Next, Globally, Until, Eventually, Release, Modality, DualModality,
    Implies, Iff, Conj, Disj, Top, Bottom)
from preprocessing.validator import validate_core_atl, filter
from acg import build_acg_final, CGS, EpsilonAtom, UniversalAtom, ExistentialAtom


class CompilerContract(unittest.TestCase):
    def setUp(self):
        self.p, self.q = Var("p"), Var("q")
        self.cgs = CGS()
        for a in ("a", "b"):
            self.cgs.add_agent(a)
            self.cgs.add_decisions(a, {0, 1})

    def test_reject_unsupported_forms(self):
        for f in (Eventually(self.p), Until(self.p, self.q, True),
                  Modality(["a"], Release(self.p, self.q)),
                  DualModality(["a"], Until(self.p, self.q)),
                  Modality(["a"], Next(Globally(self.p)))):
            with self.subTest(formula=str(f)):
                with self.assertRaises(ValueError):
                    normalize_formula(f)

    def test_builder_rejects_path_negation_and_unknown_coalition(self):
        for f in (Modality(["a"], Not(Until(self.p, self.q))),
                  Modality(["unknown"], Next(self.p))):
            with self.subTest(formula=str(f)):
                with self.assertRaises(ValueError):
                    build_acg_final(f, self.cgs)

    def test_eventually_marker_cannot_bypass_grammar(self):
        self.assertNotEqual(filter(Until(T(), self.p, True)), "ATL")

    def test_eventually_expansion_and_idempotence(self):
        f = normalize_formula(Modality(["a"], Eventually(self.p)))
        self.assertEqual(f, Modality(["a"], Until(T(), self.p)))
        self.assertEqual(normalize_formula(f), f)

    def test_boolean_normalization_truth_tables(self):
        def ev(f, valuation):
            if isinstance(f, Var): return valuation[f.name]
            if isinstance(f, T): return True
            if isinstance(f, F): return False
            if isinstance(f, Not): return not ev(f.sub, valuation)
            l, r = ev(f.lhs, valuation), ev(f.rhs, valuation)
            if isinstance(f, And): return l and r
            if isinstance(f, Or): return l or r
            if isinstance(f, Implies): return not l or r
            if isinstance(f, Iff): return l == r
            raise AssertionError(type(f))
        for f in (Not(And(self.p, self.q)), Not(Or(self.p, self.q)),
                  Iff(self.p, self.q), Not(Implies(self.p, self.q)),
                  Not(T()), Not(F())):
            normalized = normalize_formula(f)
            for bits in itertools.product((False, True), repeat=2):
                valuation = dict(zip(("p", "q"), bits))
                self.assertEqual(ev(f, valuation), ev(normalized, valuation))

    def test_thesis_example_core(self):
        f = normalize_formula(parse(tokenize(
            "<Valve> globally (underpowered implies <Reactor> next efficient)")))
        self.cgs.agents = {"Valve", "Reactor"}
        acg = build_acg_final(f, self.cgs)
        self.assertEqual(len(acg.states), 10)
        self.assertEqual(acg.final_states, {f})
        # Check the complete closure, including unreachable negative states.
        def targets(t):
            if isinstance(t, (EpsilonAtom, UniversalAtom, ExistentialAtom)):
                return {t.state}
            if isinstance(t, (Conj, Disj)):
                return targets(t.lhs) | targets(t.rhs)
            return set()
        for q in acg.states:
            for bits in itertools.product((False, True), repeat=2):
                label = frozenset(p for p, yes in zip(
                    ("underpowered", "efficient"), bits) if yes)
                self.assertLessEqual(targets(acg.get_transition(q, label)), acg.states)

    def test_six_temporal_cases_have_total_closed_transitions(self):
        for coalition in ([], ["a"], ["a", "b"]):
            for path in (Next(self.p), Globally(self.p), Until(self.p, self.q)):
                for negated in (False, True):
                    f = Modality(coalition, path)
                    if negated: f = Not(f)
                    acg = build_acg_final(normalize_formula(f), self.cgs)
                    for q in acg.states:
                        for lab in (frozenset(), frozenset({"p", "q"})):
                            self.assertIsNotNone(acg.get_transition(q, lab))

    def test_next_quantifiers_against_all_two_player_boolean_games(self):
        # Exhaust all 16 payoff tables; direct exists/forall semantics is
        # independent of the compiler's chosen transition atom and coalition.
        moves = list(itertools.product((0, 1), repeat=2))
        for truth in itertools.product((False, True), repeat=4):
            table = dict(zip(moves, truth))
            def outcomes(partial):
                return [m for m in moves if all(m[("a", "b").index(a)] == v
                                               for a, v in partial.items())]
            def decisions(agents):
                agents = sorted(agents)
                return [dict(zip(agents, m)) for m in
                        itertools.product((0, 1), repeat=len(agents))]
            def atom_value(atom):
                coalition = set(atom.agents)
                def good(move):
                    value = table[move]
                    return not value if isinstance(atom.state, Not) else value
                if isinstance(atom, UniversalAtom):
                    return any(all(good(m) for m in outcomes(d))
                               for d in decisions(coalition))
                return all(any(good(m) for m in outcomes(d))
                           for d in decisions({"a", "b"} - coalition))
            for coalition in ([], ["a"], ["b"], ["a", "b"]):
                positive = any(all(table[m] for m in outcomes(d))
                               for d in decisions(coalition))
                for kind in ("positive", "negative", "dual"):
                    f = Modality(coalition, Next(self.p))
                    expected = positive
                    if kind == "negative":
                        f, expected = Not(f), not positive
                    elif kind == "dual":
                        f = DualModality(coalition, Next(self.p))
                        expected = all(any(table[m] for m in outcomes(d))
                                       for d in decisions(coalition))
                    f = normalize_formula(f)
                    acg = build_acg_final(f, self.cgs)
                    self.assertEqual(atom_value(acg.get_transition(f, frozenset())), expected)


if __name__ == "__main__":
    unittest.main()
