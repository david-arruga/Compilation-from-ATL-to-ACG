def generate_transitions(acg):
    
    for state in acg.states:
        for sigma in acg.alphabet:  
            
            if isinstance(state, Var):  
                if state.name in sigma:
                    acg.add_transition(state, sigma, Top())  
                else:
                    acg.add_transition(state, sigma, Bottom())  

            elif isinstance(state, Not) and isinstance(state.sub, Var):  
                negated_prop = state.sub.name
                if negated_prop in sigma:
                    acg.add_transition(state, sigma, Bottom())  
                else:
                    acg.add_transition(state, sigma, Top())  

            elif isinstance(state, And):  
                acg.add_transition(state, sigma, Conj(EpsilonAtom(state.lhs), EpsilonAtom(state.rhs)))  

            elif isinstance(state, Or):  
                acg.add_transition(state, sigma, Disj(EpsilonAtom(state.lhs), EpsilonAtom(state.rhs)))  

            elif isinstance(state, Modality) and isinstance(state.sub, Next):
                next_state = state.sub.sub  
                agents = frozenset(state.agents)  
                acg.add_transition(state, sigma, ExistentialAtom(next_state, agents))  

            elif isinstance(state, Modality) and isinstance(state.sub, Globally):
                phi = state.sub.sub  
                agents = frozenset(state.agents)  
                acg.add_transition(state, sigma, Conj(EpsilonAtom(phi), ExistentialAtom(state, agents)))

            elif isinstance(state, Modality) and isinstance(state.sub, Until):
                phi1 = state.sub.lhs  
                phi2 = state.sub.rhs  
                agents = frozenset(state.agents)  
                acg.add_transition(state, sigma, Disj(EpsilonAtom(phi2), Conj(EpsilonAtom(phi1), ExistentialAtom(state, agents))))
