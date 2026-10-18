import matplotlib.pyplot as plt

from grid_gen import make_environment
from search import search
from grid_view import draw_grid
from tree_view import draw_tree

# default settings (the parameter window will replace these later)
N = 15              # grid size (N x N)
RATIO = 0.25        # fraction of cells that are obstacles
METHOD = "BFS"      # "BFS" or "DFS"
SEED = None         # set a number to repeat the same maze
DELAY = 0.10        # seconds between animation frames


def run_search(n=N, ratio=RATIO, method=METHOD, seed=SEED,
               start=None, goal=None, delay=DELAY, local=False):
    """Generate a maze and animate the search in two windows."""
    grid, start, goal = make_environment(n, ratio, start, goal, seed)
    print(f"Grid {n}x{n}, obstacles {ratio:.0%}, method {method}")
    print(f"Start: {start}   Goal: {goal}")

    plt.ion()  # interactive mode so both windows can redraw live

    # window 1: 2D grid
    fig_grid, ax_grid = plt.subplots(figsize=(6, 6))
    fig_grid.canvas.manager.set_window_title("2D Grid View")

    # window 2: search tree
    fig_tree, ax_tree = plt.subplots(figsize=(9, 6))
    fig_tree.canvas.manager.set_window_title("Search Tree View")

    # show the empty maze first so the viewer sees the starting state
    draw_grid(ax_grid, grid, start, goal, None, method)
    plt.pause(1.0)

    # animate: each step from the search generator redraws both windows
    for step in search(grid, start, goal, method):
        draw_grid(ax_grid, grid, start, goal, step, method)
        draw_tree(ax_tree, start, goal, step, method, local)
        plt.pause(delay)

    # step now holds the final step: report the result
    if step["path"]:
        print(f"Path found! Length: {len(step['path']) - 1} moves")
        print(f"States searched: {len(step['order'])}")
        print(f"Path: {step['path']}")
    else:
        print(f"No path exists. States searched: {len(step['order'])}")

    plt.ioff()
    plt.show()  # keep both windows open until the user closes them


if __name__ == "__main__":
    from params_gui import ask_params

    # show the form, run the search, and repeat until the user closes the form
    while True:
        params = ask_params()
        if params is None:
            break
        try:
            run_search(**params)
        except ValueError as e:
            print(f"Error: {e}")