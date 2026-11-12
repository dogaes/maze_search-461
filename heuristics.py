import math


def manhattan(a, b):
    """Sum of row and column distance. Exact lower bound for 4-way moves."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    """Straight-line distance. Never overestimates, but is a weaker guess."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a, b):
    """Max of row/column distance. Treats diagonal steps as cost 1."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def octile(a, b):
    """Diagonal steps cost sqrt(2), straight steps cost 1. Exact for 8-way moves."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "octile": octile,
}


def is_admissible(name, diagonal):
    """True if the heuristic never overestimates the real cost for this movement mode."""
    if name == "manhattan" and diagonal:
        return False   # counts a diagonal (cost 1.41) as 2
    return True


if __name__ == "__main__":
    a, b = (0, 0), (3, 4)
    print("From (0,0) to (3,4) on an empty grid")
    print("  true cost 4-way: 7      true cost 8-way: 5.243")
    for name, h in HEURISTICS.items():
        print(f"  {name:10s} h = {h(a, b):.3f}   "
              f"admissible 4-way: {is_admissible(name, False)}, "
              f"8-way: {is_admissible(name, True)}")

