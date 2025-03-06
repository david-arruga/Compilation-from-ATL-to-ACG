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

    return acg  
