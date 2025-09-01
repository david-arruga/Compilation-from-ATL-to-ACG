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
