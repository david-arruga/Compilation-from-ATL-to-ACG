from .predecessor import predecessor_1, predecessor_2

def attractor_1(E, S1, S2, target):
    attractor = set(target)
    changed = True
    while changed:
        changed = False
        pred = predecessor_1(E, S1, S2, attractor)
        new = pred - attractor
        if new:
            attractor.update(new)
            changed = True
    return attractor

def attractor_2(E, S1, S2, target):
    attractor = set(target)
    changed = True
    while changed:
        changed = False
        pred = predecessor_2(E, S1, S2, attractor)
        new = pred - attractor
        if new:
            attractor.update(new)
            changed = True
    return attractor