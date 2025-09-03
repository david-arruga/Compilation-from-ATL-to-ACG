from acg import ACG, CGS
from .utils import pretty_node

class GameProduct:
    def __init__(self, acg: ACG, cgs: CGS):
        self.acg = acg
        self.cgs = cgs
        self.states = set()
        self.transitions = dict()
        self.initial_states = set()
        self.S1 = set()
        self.S2 = set()
        self.B = set()

    def __str__(self):
        pretty_initial = [pretty_node(s) for s in sorted(self.initial_states, key=str)]
        pretty_s1 = [pretty_node(s) for s in sorted(self.S1, key=str)]
        pretty_s2 = [pretty_node(s) for s in sorted(self.S2, key=str)]
        pretty_B = [pretty_node(s) for s in sorted(self.B, key=str)]
        lines = [
            "GameProduct(",
            f"  Initial States: {pretty_initial}",
            f"  Total States: {len(self.states)}",
            f"  Player Accept States (S1): {pretty_s1}",
            f"  Player Reject States (S2): {pretty_s2}",
            f"  Büchi Final States (B): {pretty_B}",
            f"  Transitions (|E| = {len(self.transitions)}):"
        ]
        for (src, dst), val in sorted(self.transitions.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            lines.append(f"    ({pretty_node(src)}) → ({pretty_node(dst)})")
        lines.append(")")
        return "\n".join(lines)