def build_acg(formula):
  
    acg = ACG()  # Create an empty ACG

    def collect_subformulas(node):
        if node not in acg.states:
            acg.add_state(node)  # Add the subformula as a state

            if isinstance(node, Var):  
                acg.add_proposition(node.name)  # Directly add atomic propositions

            # Recursively explore child nodes
            for child in getattr(node, "__dict__", {}).values():
                if isinstance(child, ParseNode):
                    collect_subformulas(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, ParseNode):
                            collect_subformulas(item)

    collect_subformulas(formula)  # Populate ACG while computing closure
    acg.add_initial_state(formula)  # Set the original formula as the initial state

    return acg  # Return the constructed ACG
