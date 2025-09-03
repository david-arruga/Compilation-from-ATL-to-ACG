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
        if state not in self.states:
            self.add_state(state)
        if next_state not in self.states:
            self.add_state(next_state)
        ordered_joint_action = frozenset(sorted(joint_decision, key=lambda x: x[0]))
        self.transition_function[(state, ordered_joint_action)] = next_state

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
        return self.transition_function.get((state, joint_action), None)
    
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
        joint_actions = list(product(*[[(a,d) for d in sorted(self.decisions[a])]
                                    for a in sorted(self.agents)]))
        for s in self.states:
            for ja in joint_actions:
                ja_key = frozenset(ja)
                if (s, ja_key) not in self.transition_function:
                    errors.append(f"Missing transition from {s} with {dict(ja)}.")
        for (src, _), dst in self.transition_function.items():
            if dst not in self.states:
                errors.append(f"Transition points to undefined state {dst}.")
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