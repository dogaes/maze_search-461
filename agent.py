import math
from grid_gen import OPEN_CELL, OBSTACLE_CELL

MOVES_4 = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

MOVES_DIAGONAL = {
    "up-left": (-1, -1),
    "up-right": (-1, 1),
    "down-left": (1, -1),
    "down-right": (1, 1),
}

DIAGONAL_COST = math.sqrt(2)  # a diagonal step covers sqrt(2) units of distance


def is_open(grid, r, c):
    n = grid.shape[0]
    return 0 <= r < n and 0 <= c < n and grid[r, c] == OPEN_CELL


def get_neighbors(grid, cell, diagonal=False):
    r, c = cell
    neighbors = []
    for dr, dc in MOVES_4.values():
        if is_open(grid, r + dr, c + dc):
            neighbors.append((r + dr, c + dc))

    if diagonal:
        for dr, dc in MOVES_DIAGONAL.values():
            nr, nc = r + dr, c + dc
            if (is_open(grid, nr, nc)
                    and is_open(grid, r + dr, c)      # vertical side cell
                    and is_open(grid, r, c + dc)):    # horizontal side cell
                neighbors.append((nr, nc))
    return neighbors


def step_cost(a, b):
    if a[0] != b[0] and a[1] != b[1]:
        return DIAGONAL_COST
    return 1.0


if __name__ == "__main__":
    from grid_gen import make_environment, grid_to_text
    grid, start, goal = make_environment(n=6, obstacle_ratio=0.3, seed=12345)
    print(grid_to_text(grid, start, goal))
    print("4-way from start:", get_neighbors(grid, start))
    print("8-way from start:", get_neighbors(grid, start, diagonal=True))