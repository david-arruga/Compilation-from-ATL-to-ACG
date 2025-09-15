from preprocessing import tokenize, parse, apply_modal_dualities, normalize_formula
from acg import build_acg_final
from acceptance_game import build_game, pretty_node
from acg import cgs1

def main():
    formula = input("ATL formula: ").strip()
    ast = normalize_formula(apply_modal_dualities(parse(tokenize(formula))))
    acg = build_acg_final(ast, cgs1, materialize_alphabet=True)
    V, E, S1, S2, B, initial = build_game(acg, cgs1)
    print("Initial:", pretty_node(initial))
    print("|V|:", len(V))
    print("|E|:", len(E))
    print("|S1|:", len(S1))
    print("|S2|:", len(S2))
    print("|B|:", len(B))

if __name__ == "__main__":
    main()
