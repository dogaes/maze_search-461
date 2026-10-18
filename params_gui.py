import json
import os
import tkinter as tk
from tkinter import ttk, messagebox

SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_params.json")

DEFAULTS = {
    "n": "15", "ratio": "0.25", "method": "BFS", "seed": "",
    "start_r": "", "start_c": "", "goal_r": "", "goal_c": "",
    "delay": "0.05", "local": "False",
}


def load_saved():
    try:
        with open(SAVE_FILE) as f:
            return {**DEFAULTS, **json.load(f)}
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULTS)


def save_params(raw):
    with open(SAVE_FILE, "w") as f:
        json.dump(raw, f, indent=2)


def parse_cell(r_text, c_text, name):
    if r_text.strip() == "" and c_text.strip() == "":
        return None
    if r_text.strip() == "" or c_text.strip() == "":
        raise ValueError(f"{name}: fill in both row and column, or leave both blank.")
    return (int(r_text), int(c_text))


def ask_params():
    saved = load_saved()
    result = {}

    root = tk.Tk()
    root.title("Search Parameters")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=12)
    frame.grid()

    entries = {}

    def add_row(row, label, key, width=8):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
        e = ttk.Entry(frame, width=width)
        e.insert(0, saved[key])
        e.grid(row=row, column=1, sticky="w")
        entries[key] = e

    add_row(0, "Grid size N (N x N):", "n")
    add_row(1, "Obstacle ratio (0 - 0.9):", "ratio")

    ttk.Label(frame, text="Search method:").grid(row=2, column=0, sticky="w", pady=3)
    method_var = tk.StringVar(value=saved["method"])
    ttk.Combobox(frame, textvariable=method_var, values=["BFS", "DFS"],
                 width=6, state="readonly").grid(row=2, column=1, sticky="w")

    add_row(3, "Seed (blank = random):", "seed")

    # start and goal: row + column boxes side by side
    for row, name, rk, ck in ((4, "Start (row, col):", "start_r", "start_c"),
                              (5, "Goal (row, col):", "goal_r", "goal_c")):
        ttk.Label(frame, text=name).grid(row=row, column=0, sticky="w", pady=3)
        box = ttk.Frame(frame)
        box.grid(row=row, column=1, sticky="w")
        for i, key in enumerate((rk, ck)):
            e = ttk.Entry(box, width=4)
            e.insert(0, saved[key])
            e.grid(row=0, column=i, padx=(0, 4))
            entries[key] = e

    ttk.Label(frame, text="(leave start/goal blank for random)",
              foreground="gray").grid(row=6, column=0, columnspan=2, sticky="w")

    add_row(7, "Animation delay (sec):", "delay")
    local_var = tk.BooleanVar(value=bool(saved["local"]))
    ttk.Checkbutton(frame, text="Local tree view (best for big grids)",
                    variable=local_var).grid(row=8, column=0, columnspan=2, sticky="w", pady=3)

    def on_run():
        raw = {k: e.get() for k, e in entries.items()}
        raw["method"] = method_var.get()
        raw["local"] = local_var.get()
        try:
            n = int(raw["n"])
            ratio = float(raw["ratio"])
            if n < 2:
                raise ValueError("Grid size must be at least 2.")
            if not 0 <= ratio < 1:
                raise ValueError("Obstacle ratio must be between 0 and 1.")
            seed = int(raw["seed"]) if raw["seed"].strip() else None
            start = parse_cell(raw["start_r"], raw["start_c"], "Start")
            goal = parse_cell(raw["goal_r"], raw["goal_c"], "Goal")
            delay = float(raw["delay"])
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return

        save_params(raw)
        result.update(n=n, ratio=ratio, method=raw["method"], seed=seed,
                      start=start, goal=goal, delay=delay, local=raw["local"])
        root.destroy()

    ttk.Button(frame, text="Run Search", command=on_run).grid(
        row=9, column=0, columnspan=2, pady=(10, 0))

    root.mainloop()          # waits here until Run is clicked or the window closes
    return result or None
