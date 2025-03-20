def filter(ast, strict_ATL=True):

    has_modality = False  
    agents_set = None  

    def check_invalid(node):
        nonlocal has_modality

        if isinstance(node, Modality):
            has_modality = True  

            if not isinstance(node.sub, ParseNode) or not isinstance(node.agents, list):
                print(" ERROR: Modality must have valid agents and subformula.")
                return "INVALID"

        if isinstance(node, Until):
            if not isinstance(node.lhs, ParseNode) or not isinstance(node.rhs, ParseNode):
                print("ERROR: Until must have well defined arguments.")
                return "INVALID"

        for child in getattr(node, "__dict__", {}).values():
            if isinstance(child, ParseNode):
                result = check_invalid(child)
                if result:
                    return result 

        return None  

    resultado = check_invalid(ast)

    if resultado:
        return resultado  

    if not strict_ATL:
        return "ATL*" 

    def check_strict_ATL(node, parent=None):
        nonlocal has_modality, agents_set

        if isinstance(node, Modality):
            has_modality = True  

            if agents_set is None:
                agents_set = set(node.agents)  
            else:
                if set(node.agents) != agents_set:
                    print(f"ERROR: Modalities with different agents: {agents_set} vs {set(node.agents)}.")
                    return "ATL* but not ATL"  

            if not isinstance(node.sub, (Next, Globally, Until, Not)):
                print("ERROR: Modality can only be applied to Next, Globally or Until.")
                return "ATL* but not ATL"

        if isinstance(node, Until):
            if getattr(node, "generated_from_eventually", False):
                pass  
            # else:
            #    if not isinstance(node.rhs, (Var, Modality)):
            #        print(" ERROR: φ₂ in Until must be modal or atomic.")
            #        return "ATL* but not ATL"

        if isinstance(node, (Next, Globally, Until)):
            if not isinstance(parent, Modality):  
                print(f" ERROR: {node.__class__.__name__}  is not immediately preceded by a modality.")
                return "ATL* but not ATL"

        for key, child in node.__dict__.items():
            if isinstance(child, ParseNode):
                result = check_strict_ATL(child, node)  
                if result:
                    return result  

        return None  


    result_ATL = check_strict_ATL(ast)
    if result_ATL:
        return result_ATL  

    return "ATL"  
