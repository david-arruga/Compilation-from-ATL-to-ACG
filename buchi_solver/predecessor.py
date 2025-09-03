def predecessor_1(E, S1, S2, X):
    predecessor = set()
    for (src, dst) in E.keys():
        if dst in X:
            if src in S1:
                predecessor.add(src)
            elif src in S2:
                successors = {d for (s, d) in E if s == src}
                if successors <= X:
                    predecessor.add(src)
    print(f"\Big predecessor : {predecessor.__str__}")
    return predecessor

def predecessor_2(E, S1, S2, X):
    predecessor = set()
    for (src, dst) in E.keys():
        if dst in X:
            if src in S2:
                predecessor.add(src)
            elif src in S1:
                successors = {d for (s, d) in E if s == src}
                if successors <= X:
                    predecessor.add(src)
    return predecessor