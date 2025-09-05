from preprocessing import tokenize, parse, apply_modal_dualities, normalize_formula
from acg import build_acg_final, compute_acg_size
from acceptance_game import build_game, pretty_node
from buchi_solver import solve_buchi_game
from acceptance_game.examples import cgs1, cgs2, cgs3, cgs4
import argparse

def pick_cgs(idx: int):
    if idx == 1:
        return cgs1
    if idx == 2:
        return cgs2
    if idx == 3:
        return cgs3
    if idx == 4:
        return cgs4
    raise ValueError("cgs must be 1..4")

def run_once(formula: str, cgs_idx: int, materialize_alphabet: bool):
    ast = parse(tokenize(formula))
    ast = apply_modal_dualities(ast)
    ast = normalize_formula(ast)
    cgs = pick_cgs(cgs_idx)
    acg = build_acg_final(ast, cgs, materialize_alphabet=materialize_alphabet)
    V, E, S1, S2, B, initial = build_game(acg, cgs)
    Sj, W_total = solve_buchi_game(V, E, S1, S2, B)
    winner = "player 0" if initial in Sj else "player 1"
    satisfiable = "YES" if winner == "player 0" else "NO"
    print("Formula:", formula)
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
    print("Satisfiable:", satisfiable)

def main():
    p = argparse.ArgumentParser(prog="main", description="End-to-end ATL→ACG→Acceptance→Büchi")
    p.add_argument("--formula", default="<Reactor> globally (safe or operational)")
    p.add_argument("--cgs", type=int, default=1, choices=[1,2,3,4])
    p.add_argument("--alphabet", action="store_true")
    args = p.parse_args()
    run_once(args.formula, args.cgs, args.alphabet)

if __name__ == "__main__":
    main()