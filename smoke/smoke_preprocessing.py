from preprocessing import tokenize, parse, apply_modal_dualities, normalize_formula, filter

def main():
    formula = input("ATL formula: ").strip()
    toks = tokenize(formula)
    ast = parse(toks)
    kind_before = filter(ast)
    ast2 = apply_modal_dualities(ast)
    ast3 = normalize_formula(ast2)
    kind_after = filter(ast3)
    print("Tokens:", toks)
    print("AST:", ast)
    print("Type before:", kind_before)
    print("AST normalized:", ast3)
    print("Type after:", kind_after)
    print("Tree:")
    print(ast3.to_tree())

if __name__ == "__main__":
    main()
