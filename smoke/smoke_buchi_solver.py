from preprocessing import tokenize, parse, apply_modal_dualities, normalize_formula
from acg import build_acg_final, compute_acg_size
from acceptance_game import build_game, pretty_node
from buchi_solver import solve_buchi_game
from acg import cgs1

def main():
    formula = input("ATL formula: ").strip()
    ast = normalize_formula(apply_modal_dualities(parse(tokenize(formula))))
    acg = build_acg_final(ast, cgs1, materialize_alphabet=True)
    V, E, S1, S2, B, initial = build_game(acg, cgs1)
    Sj, W_total = solve_buchi_game(V, E, S1, S2, B)
    winner = "player 0" if initial in Sj else "player 1"
    satisfiable = "YES" if winner == "player 0" else "NO"
    print("AST:", ast)
    print("AST tree:")
    print(ast.to_tree())
    print("ACG size:", compute_acg_size(acg))
    print("ACG:")
    print(acg)
    print("Acceptance game initial:", pretty_node(initial))
    print("|V|:", len(V))
    print("|E|:", len(E))
    print("|S1|:", len(S1))
    print("|S2|:", len(S2))
    print("|B|:", len(B))
    print("Winner:", winner)
    print("Satisfiable in cgs1:", satisfiable)

if __name__ == "__main__":
    main()
