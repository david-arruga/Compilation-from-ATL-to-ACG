def parse(tokens):
    cursor = 0
    limit = len(tokens)

    def advance():
        nonlocal cursor
        if cursor < limit:
            cursor += 1

    def current_token():
        return tokens[cursor] if cursor < limit else (END_OF_INPUT, "")

    def parse_atomic_formula():
 
        token = current_token()

        if token[0] == LPAREN:
            advance()
            result = parse_formula()  
            if current_token()[0] != RPAREN:
                raise ValueError("Parse error: Expected ')' but not found.")
            advance()
            return result

        elif token[0] == PROPOSITION:
            advance()
            return Var(token[1])  

        elif token[0] == NOT:
            advance()
            return Not(parse_atomic_formula())  

        elif token[0] == LTRI:
            agents = []
            advance()
            while current_token()[0] == AGENT_NAME:
                agents.append(current_token()[1])
                advance()
                if current_token()[0] == COMMA:
                    advance()
            if current_token()[0] != RTRI:
                raise ValueError("Parse error: Expected '>' but not found.")
            advance()
            return Modality(agents, parse_temporal())  

        else:
            raise ValueError(f"Parse error: unexpected token '{token[1]}'.")

    def parse_temporal():
        
        token = current_token()

        if token[0] == NEXT:
            advance()
            return Next(parse_until()) 

        elif token[0] == GLOBALLY:
            advance()
            return Globally(parse_temporal())

        elif token[0] == EVENTUALLY:
            advance()
            return Eventually(parse_temporal())

        else:
            return parse_atomic_formula()  

    def parse_until():
       
        result = parse_temporal()
        while current_token()[0] in {UNTIL, RELEASE}:
            token_type = current_token()[0]
            advance()
            rhs = parse_temporal() 
            if token_type == UNTIL:
                result = Until(result, rhs)
            elif token_type == RELEASE:
                result = Release(result, rhs)
        return result

    def parse_conjunction():
      
        result = parse_until()
        while current_token()[0] == AND:
            advance()
            rhs = parse_until()
            result = And(result, rhs)
        return result

    def parse_disjunction():

        result = parse_conjunction()
        while current_token()[0] == OR:
            advance()
            rhs = parse_conjunction()
            result = Or(result, rhs)
        return result

    def parse_implies():
     
        result = parse_disjunction()

        while current_token()[0] in {IMPLIES, IFF}:
            token_type = current_token()[0]
            advance()
            rhs = parse_disjunction()

            if token_type == IMPLIES:
                result = Implies(result, rhs)
            elif token_type == IFF:
                result = Iff(result, rhs)

        return result

    def parse_formula():
        
        return parse_implies()

    return parse_formula()
