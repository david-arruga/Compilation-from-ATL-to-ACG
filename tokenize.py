def tokenize(source): 
    tokens = []
    cursor = 0
    length = len(source)

    keywords = {
        "and": AND,
        "or": OR,
        "not": NOT,
        "next": NEXT,
        "until": UNTIL,
        "release": RELEASE,
        "globally": GLOBALLY,
        "eventually": EVENTUALLY,
        "implies": IMPLIES
    }

    inside_agents = False  

    while cursor < length:
        symbol = source[cursor]

        if symbol in " \t\n":
            pass  

        elif symbol == "(":
            tokens.append((LPAREN, symbol))
        elif symbol == ")":
            tokens.append((RPAREN, symbol))
        elif symbol == "<":
            tokens.append((LTRI, symbol))
            inside_agents = True  
        elif symbol == ">":
            tokens.append((RTRI, symbol))
            inside_agents = False  
        elif symbol == ",":
            tokens.append((COMMA, symbol))
        elif symbol == "|":
            tokens.append((OR, symbol))
        elif symbol == "&":
            tokens.append((AND, symbol))
        elif symbol in SYMBOL_MAP:
            tokens.append((SYMBOL_MAP[symbol], symbol))
        
        elif symbol.isalpha():
            buffer = symbol
            cursor += 1
            while cursor < length and source[cursor].isalnum():
                buffer += source[cursor]
                cursor += 1

            if inside_agents:
                tokens.append((AGENT_NAME, buffer))
            else:
                token_type = keywords.get(buffer.lower(), PROPOSITION)  
                tokens.append((token_type, buffer))

            cursor -= 1  

        else:
            tokens.append((UNKNOWN, symbol))

        cursor += 1

    return tokens
