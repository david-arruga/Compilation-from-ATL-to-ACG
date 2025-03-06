def build_acg(formula):
    acg = ACG()  

    def collect_subformulas(node):

        if node not in acg.states:
            acg.add_state(node)  

            if isinstance(node, Var):  
                acg.add_proposition(node.name)  

            for child in getattr(node, "__dict__", {}).values():
                if isinstance(child, ParseNode):
                    collect_subformulas(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, ParseNode):
                            collect_subformulas(item)

    collect_subformulas(formula)
    acg.add_initial_state(formula)  

    for state in acg.states:
        for sigma in acg.alphabet:  
            
            if isinstance(state, Var):  
                if state.name in sigma:
                    acg.add_transition(state, sigma, state, "epsilon")  # True 
                else:
                    acg.add_transition(state, sigma, state, "epsilon")  # False 
            
            elif isinstance(state, Not) and isinstance(state.sub, Var):  
                negated_prop = state.sub.name
                if negated_prop in sigma:
                    acg.add_transition(state, sigma, state, "epsilon")  # False 
                else:
                    acg.add_transition(state, sigma, state, "epsilon")  # True 

            elif isinstance(state, And):  
                acg.add_transition(state, sigma, state.lhs, "epsilon")  
                acg.add_transition(state, sigma, state.rhs, "epsilon")  
            
            elif isinstance(state, Or):  
                acg.add_transition(state, sigma, state.lhs, "epsilon")  
                acg.add_transition(state, sigma, state.rhs, "epsilon")  


            elif isinstance(state, Modality) and isinstance(state.sub, Next):
                next_state = state.sub.sub  
                agents = frozenset(state.agents)  
                acg.add_transition(state, sigma, next_state, "existential", agents)  

            elif isinstance(state, Modality) and isinstance(state.sub, Globally):
                phi = state.sub.sub  
                agents = frozenset(state.agents)  
                acg.add_transition(state, sigma, phi, "epsilon")
                acg.add_transition(state, sigma, state, "existential", agents)

            elif isinstance(state, Modality) and isinstance(state.sub, Until):
                phi1 = state.sub.lhs  
                phi2 = state.sub.rhs  
                agents = frozenset(state.agents)  
                acg.add_transition(state, sigma, phi2, "epsilon")
                acg.add_transition(state, sigma, phi1, "epsilon")
                acg.add_transition(state, sigma, state, "existential", agents)


    return acg
