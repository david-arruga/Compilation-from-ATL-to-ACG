def build_acg(transformed_ast):
    
    ap_set = extract_propositions(transformed_ast)
    alphabet = set(frozenset(s) for s in chain.from_iterable(combinations(ap_set, r) for r in range(len(ap_set) + 1)))
    acg = ACG()
    acg.propositions = ap_set
    acg.alphabet = alphabet 
    closure = generate_closure(transformed_ast)
    acg.states = closure  
    acg.initial_state = transformed_ast  
    generate_transitions(acg)

    return acg  
