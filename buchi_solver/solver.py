"""Büchi winning regions on finite total turn-based arenas."""
from .attractor import attractor_1, attractor_2


def avoid_set_classical(Sj, Bj, S1, S2, E):
    # Every predecessor quantifier must range over the current subarena.
    edges = {(src, dst): value for (src, dst), value in E.items()
             if src in Sj and dst in Sj}
    accept, reject = S1 & Sj, S2 & Sj
    reachable_buchi = attractor_1(edges, accept, reject, Bj & Sj)
    return attractor_2(edges, accept, reject, Sj - reachable_buchi)


def solve_buchi_game(S, E, S1, S2, B):
    S, S1, S2, B = map(set, (S, S1, S2, B))
    if S1 & S2 or S1 | S2 != S:
        raise ValueError("Owners must partition the arena vertices.")
    if not B <= S:
        raise ValueError("Büchi vertices must belong to the arena.")
    sources = set()
    for src, dst in E:
        if src not in S or dst not in S:
            raise ValueError("Edge endpoint outside the arena.")
        sources.add(src)
    if sources != S:
        raise ValueError("The Büchi solver requires a total arena (no dead ends).")
    remaining, removed = set(S), set()
    while remaining:
        losing = avoid_set_classical(remaining, B & remaining, S1, S2, E)
        if not losing:
            break
        remaining -= losing
        removed |= losing
    return remaining, removed
