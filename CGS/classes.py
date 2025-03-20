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

class Strategy:
    def __init__(self, agents, decision_map=None):
        self.agents = set(agents)
        self.decision_map = decision_map if decision_map else {}

    def add_decision(self, history, decisions):
        if not isinstance(decisions, dict):
            raise ValueError("Decisions must be a dictionary {agent: decision}.")
        self.decision_map[tuple(history)] = decisions

    def get_decision(self, history):
        return self.decision_map.get(tuple(history), {})

    def __str__(self):
        formatted_decisions = [
            f"  History: ({' → '.join(history)}) → Decisions: {decisions}"
            for history, decisions in self.decision_map.items()
        ]
        return f"Strategy({self.agents}):\n" + "\n".join(formatted_decisions)

class CounterStrategy:
    def __init__(self, agents, decision_map=None):
        self.agents = set(agents)
        self.decision_map = decision_map if decision_map else {}

    def add_decision(self, history, decision_A_prime, decision_A_complement):
        key = (tuple(history), frozenset(decision_A_prime.items()))
        self.decision_map[key] = decision_A_complement

    def get_decision(self, history, decision_A_prime):
        key = (tuple(history), frozenset(decision_A_prime.items()))
        return self.decision_map.get(key, {})

    def __str__(self):
        formatted_decisions = [
            f"  History: ({' → '.join(history)}), Decision_A': {decision_A_prime} → Decision_A\A': {decisions_A_complement}"
            for (history, decision_A_prime), decisions_A_complement in self.decision_map.items()
        ]
        return f"CounterStrategy({self.agents}):\n" + "\n".join(formatted_decisions)
