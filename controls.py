import random

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

from grid_gen import make_environment, STUDENT_SEED
from grid_view import draw_grid
from tree_view import draw_tree
from search import search, path_cost, METHODS, INFORMED
from heuristics import HEURISTICS, is_admissible
import profiler


def parse_cell(text):
    """'3,4' -> (3, 4); blank -> None (random placement)."""
    text = text.strip()
    if not text:
        return None
    r, c = text.split(",")
    return (int(r), int(c))


class SearchController:
    """Owns the maze, the running search generator and all three windows."""

    def __init__(self):
        self.grid = self.start = self.goal = None
        self.seed = STUDENT_SEED
        self.gen = None          # the running search generator
        self.step = None         # the latest step dict
        self.running = False

        # --- the two drawing windows from Part A ---
        self.fig_grid, self.ax_grid = plt.subplots(figsize=(6, 6))
        self.fig_tree, self.ax_tree = plt.subplots(figsize=(9, 6))
        self.fig_grid.canvas.manager.set_window_title("2D Grid View")
        self.fig_tree.canvas.manager.set_window_title("Search Tree View")

        # --- the control panel window ---
        self.fig = plt.figure(figsize=(5.2, 9.2))
        self.fig.canvas.manager.set_window_title("Control Panel")
        self._build_controls()

        # Timer drives the animation: one search step per tick
        self.timer = self.fig.canvas.new_timer(interval=self._interval())
        self.timer.add_callback(self.advance)

        self.new_maze()

    # ------------------------------------------------------------------ UI
    def _build_controls(self):
        f = self.fig
        f.text(0.5, 0.975, "Search Controls", ha="center", fontsize=13, fontweight="bold")

        # Selectors
        f.text(0.05, 0.94, "Search method", fontweight="bold")
        self.rb_method = RadioButtons(f.add_axes([0.05, 0.78, 0.42, 0.155]),
                                      METHODS, active=METHODS.index("ASTAR"))
        f.text(0.53, 0.94, "Heuristic", fontweight="bold")
        heuristics = list(HEURISTICS)
        self.rb_heur = RadioButtons(f.add_axes([0.53, 0.81, 0.42, 0.125]), heuristics)
        f.text(0.53, 0.785, "Movement", fontweight="bold")
        self.rb_move = RadioButtons(f.add_axes([0.53, 0.705, 0.42, 0.075]),
                                    ["4-way", "8-way"])
        f.text(0.05, 0.755, "Tree view", fontweight="bold")
        self.cb_local = CheckButtons(f.add_axes([0.05, 0.705, 0.42, 0.045]),
                                     ["local view"], [False])

        # Text boxes for the maze parameters
        self.tb_n = TextBox(f.add_axes([0.40, 0.645, 0.25, 0.04]), "Grid size N  ", "15")
        self.tb_dens = TextBox(f.add_axes([0.40, 0.595, 0.25, 0.04]), "Obstacle density  ", "0.20")
        self.tb_seed = TextBox(f.add_axes([0.40, 0.545, 0.25, 0.04]), "Seed (blank=random)  ",
                               str(STUDENT_SEED))
        self.tb_start = TextBox(f.add_axes([0.40, 0.495, 0.25, 0.04]), "Start r,c (opt.)  ", "")
        self.tb_goal = TextBox(f.add_axes([0.40, 0.445, 0.25, 0.04]), "Goal r,c (opt.)  ", "")
        f.text(0.69, 0.555, f"ID seed {STUDENT_SEED}", fontsize=8, color="gray")

        # Speed slider (search steps per second)
        self.sl_speed = Slider(f.add_axes([0.32, 0.39, 0.5, 0.03]), "Speed (steps/s)",
                               1, 60, valinit=10, valstep=1)

        # Buttons
        self.bt_run = Button(f.add_axes([0.05, 0.315, 0.28, 0.05]), "Run")
        self.bt_step = Button(f.add_axes([0.36, 0.315, 0.28, 0.05]), "Step")
        self.bt_restart = Button(f.add_axes([0.67, 0.315, 0.28, 0.05]), "Restart")
        self.bt_single = Button(f.add_axes([0.05, 0.245, 0.43, 0.05]), "Single-run report")
        self.bt_batch = Button(f.add_axes([0.52, 0.245, 0.43, 0.05]), "Batch compare")

        # Status / live metrics area
        self.status = f.text(0.05, 0.215, "", fontsize=9, va="top", family="monospace")

        # Wire up callbacks
        self.bt_run.on_clicked(self.toggle_run)
        self.bt_step.on_clicked(lambda _: (self.stop(), self.advance()))
        self.bt_restart.on_clicked(lambda _: self.new_maze())
        self.bt_single.on_clicked(self.single_report)
        self.bt_batch.on_clicked(self.batch_compare)
        self.sl_speed.on_changed(self.change_speed)
        # Changing method / heuristic / movement re-runs on the SAME maze
        self.rb_method.on_clicked(lambda _: self.reset_search())
        self.rb_heur.on_clicked(lambda _: self.reset_search())
        self.rb_move.on_clicked(lambda _: self.reset_search())
        self.cb_local.on_clicked(lambda _: self.redraw())

    # ------------------------------------------------------- current settings
    @property
    def method(self):
        return self.rb_method.value_selected

    @property
    def heuristic(self):
        return self.rb_heur.value_selected

    @property
    def diagonal(self):
        return self.rb_move.value_selected == "8-way"

    @property
    def local(self):
        return self.cb_local.get_status()[0]

    def _interval(self):
        return int(1000 / self.sl_speed.val)

    # --------------------------------------------------------------- actions
    def new_maze(self):
        """Restart: build a new maze from the text-box settings."""
        self.stop()
        try:
            n = int(self.tb_n.text)
            density = float(self.tb_dens.text)
            seed_text = self.tb_seed.text.strip()
            self.seed = int(seed_text) if seed_text else random.randint(1, 99999)
            start, goal = parse_cell(self.tb_start.text), parse_cell(self.tb_goal.text)
            self.grid, self.start, self.goal = make_environment(
                n, density, start, goal, self.seed)
        except ValueError as e:
            self.set_status(f"Input error: {e}", "red")
            return
        if not seed_text:
            self.tb_seed.set_val(str(self.seed))   # show the random seed that was used
        self.fig_grid.canvas.manager.set_window_title(f"2D Grid View  (seed {self.seed})")
        self.reset_search()

    def reset_search(self):
        """Start the selected method from scratch on the current maze."""
        self.stop()
        self.gen = search(self.grid, self.start, self.goal, self.method,
                          self.diagonal, self.heuristic)
        self.step = None
        # Grey out the heuristic choices when the method doesn't use one
        heur_color = "black" if self.method in INFORMED else "lightgray"
        for label in self.rb_heur.labels:
            label.set_color(heur_color)
        self.redraw()

    def advance(self, *_):
        """Take one search step (one node expansion) and redraw."""
        if self.gen is None or (self.step is not None and self.step["done"]):
            self.stop()
            return
        try:
            self.step = next(self.gen)
        except StopIteration:
            self.stop()
            return
        if self.step["done"]:
            self.stop()
        self.redraw()

    def toggle_run(self, _=None):
        if self.running:
            self.stop()
        else:
            if self.step is not None and self.step["done"]:
                self.reset_search()            # finished: Run again starts over
            self.running = True
            self.bt_run.label.set_text("Stop")
            self.timer.start()
            self.fig.canvas.draw_idle()

    def stop(self):
        if self.running:
            self.timer.stop()
        self.running = False
        if hasattr(self, "bt_run"):
            self.bt_run.label.set_text("Run")
            self.fig.canvas.draw_idle()

    def change_speed(self, _):
        self.timer.interval = self._interval()
        if self.running:                       # some backends need a restart
            self.timer.stop()
            self.timer.start()

    # -------------------------------------------------------------- drawing
    def method_label(self):
        if self.method in INFORMED:
            return f"{self.method} ({self.heuristic}, {self.rb_move.value_selected})"
        return f"{self.method} ({self.rb_move.value_selected})"

    def redraw(self):
        label = self.method_label()
        draw_grid(self.ax_grid, self.grid, self.start, self.goal, self.step, label)
        self.ax_grid.set_xlabel(f"seed {self.seed}   start {self.start}   goal {self.goal}")

        if self.step is None:
            self.ax_tree.clear()
            self.ax_tree.axis("off")
            self.ax_tree.set_title("Search tree - press Run or Step")
        else:
            draw_tree(self.ax_tree, self.start, self.goal, self.step, label, self.local)

        self.fig_grid.canvas.draw_idle()
        self.fig_tree.canvas.draw_idle()
        self.update_status()

    def update_status(self):
        lines = [f"Seed: {self.seed}   Maze: {self.grid.shape[0]}x{self.grid.shape[0]}",
                 f"Method: {self.method_label()}"]
        if self.method in INFORMED and not is_admissible(self.heuristic, self.diagonal):
            lines.append("! heuristic NOT admissible with diagonals")
        if self.step is not None:
            expanded = self.step.get("total_expanded", len(self.step["order"]))
            lines.append(f"Expanded: {expanded}   Frontier: {len(self.step['frontier'])}")
            if "limit" in self.step:
                lines.append(f"ID-DFS depth limit: {self.step['limit']}")
            if self.step["done"]:
                if self.step["path"]:
                    p = self.step["path"]
                    lines.append(f"DONE  moves {len(p) - 1}, cost {path_cost(p):.2f}")
                else:
                    lines.append("DONE  no path exists")
        self.set_status("\n".join(lines))

    def set_status(self, text, color="black"):
        self.status.set_text(text)
        self.status.set_color(color)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ profiling
    def single_report(self, _=None):
        """Single-run mode: profile the selected method on the current maze."""
        self.stop()
        optimal = profiler.optimal_cost_for(self.grid, self.start, self.goal, self.diagonal)
        metrics = profiler.profile_run(self.grid, self.start, self.goal, self.method,
                                       self.diagonal, self.heuristic, optimal)
        print("Single run:", metrics)
        fig = profiler.show_single_report(metrics, self.seed)
        fig.show()

    def batch_compare(self, _=None):
        """Batch mode: run the experiment script (experiments.json)."""
        self.stop()
        self.set_status("Running batch... progress in the terminal.\n"
                        "Windows will not respond until it finishes.", "blue")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        _, log_path = profiler.run_batch(show=True, block=False)
        self.set_status(f"Batch done. Log saved to:\nresults/{log_path.split('results')[-1][1:]}",
                        "darkgreen")


if __name__ == "__main__":
    controller = SearchController()
    plt.show()