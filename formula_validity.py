def formula_validity(tokens, strict_ATL=True):

    cursor = 0
    limit = len(tokens)
    open_parentheses = 0
    open_modalities = 0  
    last_modality_agents = None  
    inside_modality = False  
    inside_boolean = False  

    def advance():
        nonlocal cursor
        if cursor < limit:
            cursor += 1

    def current_token():
        return tokens[cursor] if cursor < limit else (END_OF_INPUT, "")

    while cursor < limit:
        token = current_token()

        if token[0] == LPAREN:
            open_parentheses += 1
        elif token[0] == RPAREN:
            if open_parentheses > 0:
                open_parentheses -= 1
            else:
                print("Syntax error: Unmatched closing parenthesis.")
                return None  

        elif token[0] == LTRI:
            open_modalities += 1  
            inside_modality = True  
            advance()


            agents = []
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()

            if current_token()[0] != RTRI:
                print("Syntax error: Modality incorrectly closed.")
                return None  
            advance()  
            open_modalities -= 1  

            next_token = current_token()
            if strict_ATL and next_token[0] not in {NEXT, GLOBALLY, EVENTUALLY, UNTIL, RELEASE}:
                print(f"Error: Modality <A> applied to '{next_token[1]}', which is not allowed in ATL.")
                return None  

            if strict_ATL:
                agent_set = frozenset(agents)  
                if last_modality_agents is None:
                    last_modality_agents = agent_set  
                elif last_modality_agents != agent_set:
                    print("Error: Different agent sets detected in ATL.")
                    return None  

        elif token[0] in {AND, OR}:
            inside_boolean = True  
            next_token = tokens[cursor + 1] if cursor + 1 < limit else (END_OF_INPUT, "")

            if strict_ATL and next_token[0] == LTRI and not inside_modality:
                print("Error: Modality <A> found inside OR/AND without proper coverage.")
                return None  

        elif token[0] == RTRI:
            inside_modality = False  

        inside_boolean = False  
        advance()

    if open_parentheses != 0:
        print(f"Syntax error: Unmatched opening parenthesis ({open_parentheses} left open).")
        return None  
    if open_modalities != 0:
        print(f"Syntax error: Unmatched modality opening ({open_modalities} left open).")
        return None  

    return tokens  
