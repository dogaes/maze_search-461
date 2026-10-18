import numpy as np

OPEN_CELL = 0
OBSTACLE_CELL = 1

def generate_grid(n=15, obstacle_ratio=0.25, seed=None):
    if n <= 0:
        raise ValueError("Grid size n must be a positive integer.")
    if not 0.0 <= obstacle_ratio < 1.0:
        raise ValueError("Obstacle ratio must be between 0.0 and 1.0 (exclusive).")
    
    if seed is not None:
        np.random.seed(seed)
    
    # create a grid with random obstacles
    grid = (np.random.rand(n, n) < obstacle_ratio).astype(int)
    return grid

def get_open_cells(grid):
    return [tuple(int(x) for x in cell) for cell in np.argwhere(grid == OPEN_CELL)]

def place_start_goal(grid, start=None, goal=None):
    n = grid.shape[0]
    
    for name, cell in (("start", start), ("goal", goal)):
        if cell is not None:
            r, c = cell
            if not (0 <= r < n and 0 <= c < n):
                raise ValueError(f"{name} {cell} is outside the grid bounds.")
            grid[r, c] = OPEN_CELL

    if start is not None and start == goal:
        raise ValueError("Start and goal coordinates cannot be the same.")
    
    open_cells = get_open_cells(grid)
    if len(open_cells) < 2:
        raise ValueError("Not enough open cells to place start and goal.")
    
    if start is None and goal is None:
        i, j = np.random.choice(len(open_cells), size=2, replace=False)
        start, goal = open_cells[i], open_cells[j]
    elif start is None:
        candidates = [c for c in open_cells if c != tuple(goal)]
        start = candidates[np.random.randint(len(candidates))]
    elif goal is None:
        candidates = [c for c in open_cells if c != tuple(start)]
        goal = candidates[np.random.randint(len(candidates))]

    return tuple(start), tuple(goal)

def make_environment(n=15, obstacle_ratio=0.25, start=None, goal=None, seed=None):
    grid = generate_grid(n, obstacle_ratio, seed)
    start, goal = place_start_goal(grid, start, goal)
    return grid, start, goal

def grid_to_text(grid, start=None, goal=None):
    lines = []
    for r in range(grid.shape[0]):
        row = []
        for c in range(grid.shape[1]):
            if (r, c) == start:
                row.append("S")
            elif (r, c) == goal:
                row.append("G")
            elif grid[r, c] == OBSTACLE_CELL:
                row.append("#")
            else:
                row.append(".")
        lines.append(" ".join(row))
    return "\n".join(lines)
