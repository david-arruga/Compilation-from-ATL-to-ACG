from .attractor import attractor_1, attractor_2

def avoid_set_classical(Sj, Bj, S1, S2, E):
    A1j = attractor_1(E, S1, S2, Bj)
    notA1j = Sj - A1j
    Wj1 = attractor_2(E, S1, S2, notA1j)
    return Wj1

def solve_buchi_game(S, E, S1, S2, B):
    Sj = set(S)
    W_total = set()
    j = 0
    contador = 0
    numerador = 0
    while True:
        contador = contador + 1
        if contador > 100:
            numerador = numerador + 1
            print(f"\n Iteracion número {numerador*contador}")
            contador = 0
        Bj = B & Sj
        Wj1 = avoid_set_classical(Sj, Bj, S1, S2, E)
        if not Wj1:
            break
        Sj = Sj - Wj1
        W_total |= Wj1
        j += 1
    return Sj, W_total