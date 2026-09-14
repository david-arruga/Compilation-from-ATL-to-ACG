import itertools
import unittest
from preprocessing import (parse, tokenize, normalize_formula, Var, T, F,
    Not, And, Or, Implies, Iff, Next, Globally, Eventually, Until, Modality)
from acg import CGS, build_acg_final
from acceptance_game import build_game


def read(s):
    return parse(tokenize(s))


class ParserTests(unittest.TestCase):
    def test_names_are_whole_tokens(self):
        for name in ('Fuel', 'Global', 'Ready', 'Xray', 'p_0'):
            self.assertEqual(read(name), Var(name))
        self.assertEqual(read('<ctrl_0> G p_0'), Modality(['ctrl_0'], Globally(Var('p_0'))))

    def test_constants_and_printed_symbols(self):
        self.assertEqual(read('⊤'), T())
        self.assertEqual(read('false'), F())
        f = Modality(['a'], Until(Var('p'), Not(Var('q'))))
        self.assertEqual(read(f.to_formula()), f)
        self.assertEqual(read('p -> q'), read('p implies q'))
        self.assertEqual(read('p <-> q'), read('p iff q'))

    def test_no_modality_moved_across_parentheses(self):
        f = read('(<a> X p) U q')
        self.assertIsInstance(f, Until)
        with self.assertRaises(ValueError):
            normalize_formula(f)
        self.assertEqual(read('<a>p U q'), read('<a>(p U q)'))

    def test_precedence(self):
        p, q, r = map(Var, ('p', 'q', 'r'))
        self.assertEqual(read('p implies q implies r'), Implies(p, Implies(q, r)))
        self.assertEqual(read('p or q and r'), Or(p, And(q, r)))
        self.assertEqual(read('not <a> G p'), Not(Modality(['a'], Globally(p))))
        self.assertEqual(read('p iff q implies r'), Iff(p, Implies(q, r)))

    def test_malformed_inputs(self):
        for text in ('<a,> X p', '<a b> X p', '<,a> X p', 'p q', 'p @ q',
                     '', '(p', 'p)', '<a X p', '[a,] G p'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                read(text)
        self.assertEqual(read('<> G p'), Modality([], Globally(Var('p'))))


def buchi_oracle(arena):
    """Independent nested fixed-point oracle; never calls buchi_solver.

    nu Z. mu Y. ((B intersect CPre(Z)) union CPre(Y)), on a total finite arena.
    Intended only for small regression fixtures, not production performance.
    """
    V, E, accept, reject, B, initial = arena
    assert accept.isdisjoint(reject) and accept | reject == V
    successors = {v: set() for v in V}
    for src, dst in E:
        successors[src].add(dst)
    assert all(successors.values())
    def pre(X):
        return {v for v in V if (bool(successors[v] & X) if v in accept
                                 else successors[v] <= X)}
    Z = set(V)
    while True:
        Y = set()
        base = B & pre(Z)
        while True:
            new = base | pre(Y)
            if new == Y: break
            Y = new
        if Y == Z: return initial in Z
        Z = Y


class TemporalTests(unittest.TestCase):
    def test_gu_against_unique_path_semantics(self):
        # Every deterministic transition map on two states, all p/q labels,
        # both roots: 1024 comparisons covering finite prefixes and cycles.
        p, q = Var('p'), Var('q')
        bodies = [Globally(p), Until(p, q), Eventually(q),
                  Globally(Modality(['a'], Eventually(q)))]
        formulas = [f for path in bodies for f in
                    (Modality(['a'], path), Not(Modality(['a'], path)))]
        def semantic(f, s, labels, next_state):
            if isinstance(f, Var): return f.name in labels[s]
            if isinstance(f, Not): return not semantic(f.sub, s, labels, next_state)
            if isinstance(f, Modality): return semantic(f.sub, s, labels, next_state)
            seen = set()
            while s not in seen:
                seen.add(s)
                if isinstance(f, Globally):
                    if not semantic(f.sub, s, labels, next_state): return False
                elif isinstance(f, Eventually):
                    if semantic(f.sub, s, labels, next_state): return True
                elif isinstance(f, Until):
                    if semantic(f.rhs, s, labels, next_state): return True
                    if not semantic(f.lhs, s, labels, next_state): return False
                else: raise AssertionError(type(f))
                s = next_state[s]
            return isinstance(f, Globally)
        for next_state in itertools.product((0, 1), repeat=2):
            for bits in itertools.product((False, True), repeat=4):
                labels = [{p for p, yes in zip(('p', 'q'), bits[2*s:2*s+2]) if yes}
                          for s in (0, 1)]
                g = CGS()
                g.add_agent('a')
                g.add_decisions('a', {'stay'})
                for name in ('p', 'q'): g.add_proposition(name)
                for s in (0, 1):
                    g.add_state(s)
                    g.label_state(s, labels[s])
                    g.add_transition(s, [('a', 'stay')], next_state[s])
                for root in (0, 1):
                    g.set_initial_state(root)
                    for original in formulas:
                        compiled = build_acg_final(normalize_formula(original), g)
                        actual = buchi_oracle(build_game(compiled, g))
                        self.assertEqual(actual, semantic(original, root, labels, next_state),
                                         (next_state, labels, root, str(original)))


if __name__ == '__main__':
    unittest.main()
