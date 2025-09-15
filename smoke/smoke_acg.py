from preprocessing import tokenize, parse, apply_modal_dualities, normalize_formula
from acg import build_acg_final, compute_acg_size
from acg import cgs1

def main():
    formula = input("ATL formula: ").strip()
    ast = normalize_formula(apply_modal_dualities(parse(tokenize(formula))))
    acg = build_acg_final(ast, cgs1, materialize_alphabet=True)
    print("AST:", ast)
    print("ACG size:", compute_acg_size(acg))
    print("ACG:")
    print(acg)

if __name__ == "__main__":
    main()
