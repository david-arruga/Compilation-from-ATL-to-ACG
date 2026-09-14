from __future__ import annotations
from itertools import product

class CGS:
    def __init__(self):
        self.propositions = set()
        self.agents = set()
        self.states = set()
        self.initial_state = None
        self.labeling_function = {}
        self.decisions = {}
        self.transition_function = {}
        self.strategies = {}

    def add_proposition(self, proposition):
        self.propositions.add(proposition)

    def add_state(self, state):
        self.states.add(state)

    def set_initial_state(self, state):
        if state not in self.states:
            self.add_state(state)
        self.initial_state = state

    def label_state(self, state, propositions):
        if state not in self.states:
            self.add_state(state)
        self.labeling_function[state] = set(propositions)

    def add_agent(self, agent):
        self.agents.add(agent)

    def add_decisions(self, agent, decision_set):
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the CGS.")
        self.decisions[agent] = set(decision_set)

    def add_transition(self, state, joint_decision, next_state):
        ordered_joint_action = frozenset(sorted(joint_decision, key=lambda x: x[0]))
        key = (state, ordered_joint_action)
        if key in self.transition_function and self.transition_function[key] != next_state:
            raise ValueError("Conflicting successors for the same joint decision.")
        self.states.update((state, next_state))
        self.transition_function[key] = next_state

    def get_all_agent_choices(self, agent_subset):
        agent_subset = sorted(agent_subset)
        all_choices = [self.decisions[agent] for agent in agent_subset]
        combinations = product(*all_choices)
        return [
            dict(zip(agent_subset, combo)) for combo in combinations
        ]

    def get_joint_actions_for_agents(self, agent_subset):
        agent_subset = sorted(agent_subset)
        all_choices = [self.decisions[agent] for agent in agent_subset]
        combos = product(*all_choices)
        return [dict(zip(agent_subset, combo)) for combo in combos]

    def get_successor(self, state, joint_decision_dict):
        joint_action = frozenset(sorted(joint_decision_dict.items()))
        try:
            return self.transition_function[(state, joint_action)]
        except KeyError as exc:
            raise ValueError("Missing CGS transition for state and joint decision.") from exc
    
    def get_propositions(self):
        return sorted(self.propositions)

    def get_agents(self):
        return sorted(self.agents)

    def __str__(self):
        formatted_transitions = []
        for (state, joint_decision) in self.transition_function:
            decision_str = ", ".join([f"({agent}, {decision})" for agent, decision in joint_decision])
            next_state = self.transition_function[(state, joint_decision)]
            formatted_transitions.append(f"    τ({state}, {{{decision_str}}}) → {next_state}")
        return (
            f"CGS(\n"
            f"  Propositions: {self.propositions}\n"
            f"  Agents: {sorted(self.agents)}\n"
            f"  States: {sorted(self.states)}\n"
            f"  Initial State: {self.initial_state}\n"
            f"  Labeling Function: {self.labeling_function}\n"
            f"  Decisions: {self.decisions}\n"
            f"  Transitions:\n" +
            "\n".join(formatted_transitions) +
            f"\n)"
        )
    
    def validate(self, *, check_reachability=False, verbose=False):
        errors = []
        if self.initial_state is None or self.initial_state not in self.states:
            errors.append("Initial state missing or not in self.states.")
        missing_dec_sets = [a for a in self.agents
                            if a not in self.decisions or not self.decisions[a]]
        if missing_dec_sets:
            errors.append(f"Agents without decision sets: {missing_dec_sets}")
        if not self.agents:
            errors.append("The thesis scope requires a nonempty agent set.")
        if None in self.states:
            errors.append("None is reserved for missing states in this representation.")
        if set(self.decisions) - self.agents:
            errors.append("Decision sets for unknown agents.")
        if set(self.labeling_function) - self.states:
            errors.append("Labels for unknown states.")
        joint_actions = []
        if not missing_dec_sets:
            joint_actions = list(product(*[[(a, d) for d in self.decisions[a]]
                                           for a in sorted(self.agents)]))
            for s in self.states:
                for ja in joint_actions:
                    if (s, frozenset(ja)) not in self.transition_function:
                        errors.append(f"Missing transition from {s} with {dict(ja)}.")
        for (src, decision), dst in self.transition_function.items():
            if src not in self.states or dst not in self.states:
                errors.append("Transition endpoint outside the state set.")
            pairs = list(decision)
            if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs):
                errors.append("Malformed joint decision.")
                continue
            names = [a for a, _ in pairs]
            if len(names) != len(set(names)) or set(names) != self.agents:
                errors.append("A joint decision must assign exactly one action to each agent.")
            if any(d not in self.decisions.get(a, set()) for a, d in pairs):
                errors.append("Joint decision contains an unavailable action.")
        for st in self.states:
            if st not in self.labeling_function:
                errors.append(f"State {st} has no label.")
            else:
                unknown_props = self.labeling_function[st] - self.propositions
                if unknown_props:
                    errors.append(f"Unknown propositions in label of {st}: {unknown_props}")
        if check_reachability and not errors:
            seen = {self.initial_state}
            frontier = [self.initial_state]
            while frontier:
                cur = frontier.pop()
                for ja in joint_actions:
                    nxt = self.transition_function[(cur, frozenset(ja))]
                    if nxt not in seen:
                        seen.add(nxt)
                        frontier.append(nxt)
            unreachable = self.states - seen
            if unreachable:
                errors.append(f"Unreachable states: {unreachable}")
        if errors:
            msg = "CGS validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(msg)
        if verbose:
            print("CGS validation successful: all checks passed.")