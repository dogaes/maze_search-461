import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

COLORS = ListedColormap([
    "white",      # 0: open cell
    "black",      # 1: obstacle
    "lightgreen",  # 2: visited cell
    "lightgray",     # 3: frontier cell
    "yellow",      # 4: final path
    "green",        # 5: current agent position
    "lightblue",       # 6: start
    "pink",     # 7: goal
])

HERE = os.path.dirname(os.path.abspath(__file__))


def load_icon(filename):
    """Load an image from the project folder, or return None if it's missing."""
    path = os.path.join(HERE, filename)
    try:
        return plt.imread(path)
    except (FileNotFoundError, OSError):
        return None


START_ICON = load_icon("start.png")
GOAL_ICON = load_icon("goal.png")


def place_icon(ax, icon, cell, fallback_text, text_color):
    """Draw an image filling one grid cell, or a text label if there's no image."""
    r, c = cell
    if icon is not None:
        # extent = (left, right, bottom, top) so the image fills exactly one cell
        ax.imshow(icon, extent=(c - 0.5, c + 0.5, r + 0.5, r - 0.5), zorder=3)
    else:
        ax.text(c, r, fallback_text, ha="center", va="center",
                color=text_color, fontsize=12, fontweight="bold")

def draw_grid(ax, grid, start, goal, step=None, method=""):
    codes = grid.copy()
    if step is not None:
        for cell in step["order"]: codes[cell] = 2
        for cell in step ["frontier"]: codes[cell] = 3
        if step["path"]:
            for cell in step["path"]: codes[cell] = 4
        if step["current"] is not None:
            codes[step["current"]] = 5
    codes[start] = 6
    codes[goal] = 7

    ax.clear()
    ax.imshow(codes, cmap=COLORS, vmin=-0.5, vmax=7.5)

    # put "S" and "G" text labels on the start and goal
    place_icon(ax, START_ICON, start, "S", "white")
    place_icon(ax, GOAL_ICON, goal, "G", "black")

    # extra imshow calls can shift the view, so lock it back to the grid
    n = grid.shape[0]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    
    # add a title with the current step number and the number of states searched
    if step is None:
        ax.set_title("Maze")
    elif step["done"] and step["path"]:
        ax.set_title(f"{method} done: path length {len(step['path']) - 1}, "
                     f"states searched {len(step['order'])}")
    elif step["done"]:
        ax.set_title(f"{method}: no path found ({len(step['order'])} states searched)")
    else:
        ax.set_title(f"{method}: states searched {len(step['order'])}")
