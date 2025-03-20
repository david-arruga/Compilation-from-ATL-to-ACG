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


def play(game, initial_state, strategy, counterstrategy, max_steps=10):
    history = [initial_state]
    current_state = initial_state

    for _ in range(max_steps):
        print(f"\n🟢 Current State: {current_state}")

        decision_A = strategy.get_decision([current_state])
        if not decision_A:
            print(f" Strategy failed to return a decision at {current_state}. Stopping.")
            break  

        decision_A_complement = counterstrategy.get_decision([current_state], decision_A)
        if not decision_A_complement:
            print(f" CounterStrategy failed to return a decision at {current_state}. Stopping.")
            break  

        joint_action = frozenset(list(decision_A.items()) + list(decision_A_complement.items()))

        if (current_state, joint_action) in game.transition_function:
            current_state = game.transition_function[(current_state, joint_action)]
            history.append(current_state)
            print(f" Transition found! Moving to: {current_state}")
        else:
            print(" No matching transition found. Stopping.")
            break  

    return history



game = CGS()

game.add_proposition("safe")
game.add_proposition("critical")
game.add_proposition("operational")
game.add_proposition("Explosion")
game.add_proposition("shutdown")

game.add_agent("Reactor")
game.add_agent("Pressure")

game.add_decisions("Reactor", {"heat", "cool", "emergency_shutdown"})
game.add_decisions("Pressure", {"hold", "vent", "release_pressure"})

game.add_state("Start")
game.add_state("Stable")
game.add_state("Cold")
game.add_state("Critical")
game.add_state("Shut Down")
game.add_state("Emergency Shutdown")

game.set_initial_state("Start")

game.label_state("Start", {"safe"})
game.label_state("Critical", {"critical"})
game.label_state("Stable", {"safe", "operational"})
game.label_state("Cold", {"safe"})
game.label_state("Shut Down", {"Explosion"})
game.label_state("Emergency Shutdown", {"shutdown"})

game.add_transition("Start", {("Reactor", "heat"), ("Pressure", "hold")}, "Stable")
game.add_transition("Start", {("Reactor", "heat"), ("Pressure", "vent")}, "Stable")
game.add_transition("Start", {("Reactor", "cool"), ("Pressure", "hold")}, "Start")
game.add_transition("Start", {("Reactor", "cool"), ("Pressure", "vent")}, "Start")

game.add_transition("Stable", {("Reactor", "heat"), ("Pressure", "vent")}, "Stable")
game.add_transition("Stable", {("Reactor", "cool"), ("Pressure", "hold")}, "Cold")
game.add_transition("Stable", {("Reactor", "cool"), ("Pressure", "vent")}, "Cold")
game.add_transition("Stable", {("Reactor", "heat"), ("Pressure", "hold")}, "Critical")
game.add_transition("Stable", {("Reactor", "cool"), ("Pressure", "release_pressure")}, "Start")

game.add_transition("Cold", {("Reactor", "heat"), ("Pressure", "hold")}, "Stable")
game.add_transition("Cold", {("Reactor", "heat"), ("Pressure", "vent")}, "Stable")
game.add_transition("Cold", {("Reactor", "cool"), ("Pressure", "hold")}, "Start")
game.add_transition("Cold", {("Reactor", "cool"), ("Pressure", "vent")}, "Start")

game.add_transition("Critical", {("Reactor", "cool"), ("Pressure", "hold")}, "Critical")
game.add_transition("Critical", {("Reactor", "heat"), ("Pressure", "vent")}, "Stable")
game.add_transition("Critical", {("Reactor", "cool"), ("Pressure", "vent")}, "Stable")
game.add_transition("Critical", {("Reactor", "heat"), ("Pressure", "hold")}, "Shut Down")
game.add_transition("Critical", {("Reactor", "emergency_shutdown"), ("Pressure", "release_pressure")}, "Emergency Shutdown")

game.add_transition("Shut Down", {("Reactor", "heat"), ("Pressure", "hold")}, "Start")
game.add_transition("Shut Down", {("Reactor", "heat"), ("Pressure", "vent")}, "Start")
game.add_transition("Shut Down", {("Reactor", "cool"), ("Pressure", "vent")}, "Start")
game.add_transition("Shut Down", {("Reactor", "cool"), ("Pressure", "hold")}, "Start")

strategy_reactor = Strategy({"Reactor"})  
strategy_reactor.add_decision(["Start"], {"Reactor": "heat"})
strategy_reactor.add_decision(["Stable"], {"Reactor": "cool"})
strategy_reactor.add_decision(["Critical"], {"Reactor": "cool"})
strategy_reactor.add_decision(["Shut Down"], {"Reactor": "emergency_shutdown"})

counterstrategy_pressure = CounterStrategy({"Pressure"})  
counterstrategy_pressure.add_decision(["Start"], {"Reactor": "heat"}, {"Pressure": "hold"})
counterstrategy_pressure.add_decision(["Stable"], {"Reactor": "cool"}, {"Pressure": "release_pressure"})
counterstrategy_pressure.add_decision(["Critical"], {"Reactor": "cool"}, {"Pressure": "vent"})
counterstrategy_pressure.add_decision(["Critical"], {"Reactor": "emergency_shutdown"}, {"Pressure": "release_pressure"})

print("\n🔍 Simulating a Play from 'Start'...\n")
play_trace = play(game, "Start", strategy_reactor, counterstrategy_pressure, max_steps=10)

print("\nCGS Structure:\n", game)
print("\n", strategy_reactor)
print("\n", counterstrategy_pressure)
print("\nResulting Play Trace:", " → ".join(play_trace))
