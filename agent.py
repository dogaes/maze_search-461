from grid_gen import OPEN_CELL, OBSTACLE_CELL

MOVES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}

def get_neighbors(grid, cell):
    n = grid.shape[0]
    r, c = cell
    neighbors = []
    for dr, dc in MOVES.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == OPEN_CELL:
            neighbors.append((nr, nc))
    return neighbors


if __name__ == "__main__":
    from grid_gen import make_environment, grid_to_text
    grid, start, goal = make_environment(n=6, obstacle_ratio=0.3)
    print(grid_to_text(grid, start, goal))
    print("Start:", start, "-> can move to:", get_neighbors(grid, start))
    print("Corner (0,0) neighbors:", get_neighbors(grid, (0, 0)))