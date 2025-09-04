from __future__ import annotations
from itertools import product
from acg import CGS

class Coalition(list):
    def __sub__(self, other):
        return set(self) - set(other)

def full_coalition(n: int) -> Coalition:
    return Coalition(f"ctrl_{i}" for i in range(n))

def generate_lights_cgs(n: int) -> CGS:
    g = CGS()
    for i in range(n):
        g.add_proposition(f"p_{i}")
    for i in range(n):
        a = f"ctrl_{i}"
        g.add_agent(a)
        g.add_decisions(a, {f"toggle_{i}", "wait"})
    states = [tuple(bs) for bs in product([0,1], repeat=n)]
    for s in states:
        g.add_state(s)
        lab = {f"p_{i}" for i,bit in enumerate(s) if bit}
        g.label_state(s, lab)
    init = tuple(0 for _ in range(n))
    g.set_initial_state(init)
    for s in states:
        for joint in product(*[[(a,d) for d in g.decisions[a]] for a in sorted(g.agents)]):
            new_bits = list(s)
            for (agent, act) in joint:
                idx = int(agent.split("_")[1])
                if act == f"toggle_{idx}":
                    new_bits[idx] = 1 - new_bits[idx]
            s_next = tuple(new_bits)
            g.add_transition(s, joint, s_next)
    return g