import csv
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime

import matplotlib.pyplot as plt

from grid_gen import make_environment, STUDENT_SEED
from search import search, path_cost, METHODS, INFORMED

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

def _trace(grid, start, goal, method, diagonal, heuristic):
    """Run a search in light mode and collect the search-trace counters."""
    generated = 0
    max_children = 0
    peak_footprint = 0
    step = None
    for step in search(grid, start, goal, method, diagonal, heuristic, light=True):
        generated += step["n_children"]
        max_children = max(max_children, step["n_children"])
        peak_footprint = max(peak_footprint,
                             step["frontier_size"] + step["explored_size"])
    expanded = step["expanded"]
    return {
        "path": step["path"],
        "expanded": expanded,
        "generated": generated,
        "branching_avg": generated / expanded if expanded else 0.0,
        "branching_max": max_children,
        "peak_footprint": peak_footprint,
    }


def profile_run(grid, start, goal, method, diagonal=False, heuristic="manhattan",
                optimal_cost=None):
    # 1) Timing run: high-resolution wall clock, nothing else switched on
    t0 = time.perf_counter_ns()
    for _ in search(grid, start, goal, method, diagonal, heuristic, light=True):
        pass
    time_ms = (time.perf_counter_ns() - t0) / 1e6

    # 2) Memory run: peak bytes allocated by Python during the search
    tracemalloc.start()
    trace = _trace(grid, start, goal, method, diagonal, heuristic)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    path = trace["path"]
    cost = path_cost(path) if path else None
    optimality = (optimal_cost / cost) if (path and optimal_cost) else None

    return {
        "method": method,
        "heuristic": heuristic if method in INFORMED else "-",
        "movement": "8-way" if diagonal else "4-way",
        "found": path is not None,
        "path_moves": len(path) - 1 if path else None,     # solution depth
        "path_cost": round(cost, 3) if cost is not None else None,
        "optimality": round(optimality, 3) if optimality is not None else None,
        "time_ms": time_ms,
        "mem_peak_kb": peak_bytes / 1024,
        "peak_footprint": trace["peak_footprint"],
        "expanded": trace["expanded"],
        "generated": trace["generated"],
        "branching_avg": round(trace["branching_avg"], 3),
        "branching_max": trace["branching_max"],
    }


def optimal_cost_for(grid, start, goal, diagonal):
    """True cheapest path cost, from A* with an admissible heuristic (None if unsolvable)."""
    heuristic = "octile" if diagonal else "manhattan"
    for step in search(grid, start, goal, "ASTAR", diagonal, heuristic, light=True):
        pass
    return path_cost(step["path"]) if step["path"] else None


def show_single_report(metrics, seed=None):
    """Open a window showing one run's metrics as a table (single-run mode)."""
    labels = [
        ("Method", metrics["method"]),
        ("Heuristic", metrics["heuristic"]),
        ("Movement", metrics["movement"]),
        ("Seed", seed if seed is not None else "-"),
        ("Path found", "yes" if metrics["found"] else "no"),
        ("Path length (moves / depth)", metrics["path_moves"]),
        ("Path cost", metrics["path_cost"]),
        ("Optimality (optimal / found)", metrics["optimality"]),
        ("Runtime (ms)", f"{metrics['time_ms']:.3f}"),
        ("Process memory peak (KB)", f"{metrics['mem_peak_kb']:.1f}"),
        ("Footprint: peak frontier+explored", metrics["peak_footprint"]),
        ("Nodes expanded", metrics["expanded"]),
        ("Nodes generated", metrics["generated"]),
        ("Branching factor avg / max",
         f"{metrics['branching_avg']} / {metrics['branching_max']}"),
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.canvas.manager.set_window_title("Single Run Metrics")
    ax.axis("off")
    table = ax.table(cellText=[[k, "-" if v is None else v] for k, v in labels],
                     colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title("Single run report", fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Batch compare
# --------------------------------------------------------------------------

DEFAULT_EXPERIMENT = {
    "description": "Default protocol: 4 sizes x 3 densities = 12 search spaces",
    "base_seed": STUDENT_SEED,
    "sizes": [10, 15, 20, 30],
    "densities": [0.08, 0.16, 0.24],
    "movements": ["4-way", "8-way"],
    "heuristics": ["manhattan", "euclidean", "chebyshev", "octile"],
    "methods": METHODS,
    "repeats": 5,
    "placement": "corners",
}


def load_experiment(path=None):
    """Load an experiment script (JSON). Missing keys fall back to the defaults."""
    if path is None:
        path = os.path.join(HERE, "experiments.json")
    if not os.path.exists(path):
        print(f"No experiment file at {path}; using built-in defaults.")
        return dict(DEFAULT_EXPERIMENT)
    with open(path) as f:
        return {**DEFAULT_EXPERIMENT, **json.load(f)}


def build_spaces(exp):
    """
    Generate the search spaces: one per (size, density), each with its own
    reproducible seed. Start/goal go in opposite corners (or random) so every
    space has a long, comparable search. Unsolvable mazes are skipped by
    moving to the next seed, and the seed actually used is recorded.
    """
    spaces = []
    index = 0
    for n in exp["sizes"]:
        for density in exp["densities"]:
            seed = exp["base_seed"] + index * 1000
            for attempt in range(50):
                if exp["placement"] == "corners":
                    start_goal = ((0, 0), (n - 1, n - 1))
                else:
                    start_goal = (None, None)
                grid, start, goal = make_environment(n, density, *start_goal,
                                                     seed=seed + attempt)
                if optimal_cost_for(grid, start, goal, False) is not None:
                    break
            spaces.append({"id": index + 1, "n": n, "density": density,
                           "seed": seed + attempt, "grid": grid,
                           "start": start, "goal": goal})
            index += 1
    return spaces


def _method_variants(exp):
    """Every (method, heuristic) pair to test; uninformed methods run once."""
    variants = []
    for m in exp["methods"]:
        if m in INFORMED:
            variants += [(m, h) for h in exp["heuristics"]]
        else:
            variants.append((m, "-"))
    return variants


def run_batch(exp_path=None, show=True, block=True):
    """
    Run the full experiment protocol and report it.

    Returns (summary_rows, log_path).
    """
    exp = load_experiment(exp_path)
    spaces = build_spaces(exp)
    variants = _method_variants(exp)
    movements = [m == "8-way" for m in exp["movements"]]
    total = len(spaces) * len(movements) * len(variants)
    print(f"Batch: {len(spaces)} spaces x {len(movements)} movements x "
          f"{len(variants)} method variants x {exp['repeats']} repeats")

    raw_rows = []      # one row per individual run
    summary = []       # one row per (space, movement, method, heuristic)
    done = 0
    for space in spaces:
        for diagonal in movements:
            opt = optimal_cost_for(space["grid"], space["start"], space["goal"], diagonal)
            for method, heur in variants:
                runs = []
                for rep in range(exp["repeats"]):
                    m = profile_run(space["grid"], space["start"], space["goal"],
                                    method, diagonal,
                                    heur if heur != "-" else "manhattan", opt)
                    m.update(space=space["id"], n=space["n"], density=space["density"],
                             seed=space["seed"], repeat=rep + 1)
                    runs.append(m)
                    raw_rows.append(m)
                summary.append(_summarize(runs))
                done += 1
                print(f"\r  progress {done}/{total}", end="", flush=True)
    print()

    log_path = write_logs(exp, spaces, raw_rows, summary)
    print(f"Results written to {log_path}")
    if show:
        show_batch_window(summary, exp, block)
    return summary, log_path


def _summarize(runs):
    """Mean and spread (best / worst) of the repeated runs of one configuration."""
    first = runs[0]
    times = [r["time_ms"] for r in runs]
    mems = [r["mem_peak_kb"] for r in runs]
    row = {k: first[k] for k in ("space", "n", "density", "seed", "method",
                                 "heuristic", "movement", "found", "path_moves",
                                 "path_cost", "optimality", "expanded", "generated",
                                 "branching_avg", "branching_max", "peak_footprint")}
    row.update(
        time_mean=statistics.mean(times),
        time_std=statistics.stdev(times) if len(times) > 1 else 0.0,
        time_best=min(times), time_worst=max(times),
        mem_mean=statistics.mean(mems),
        mem_best=min(mems), mem_worst=max(mems),
    )
    return row


def write_logs(exp, spaces, raw_rows, summary):
    """Write raw CSV, summary CSV and an annotated, human-readable .log file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(RESULTS_DIR, f"batch_{stamp}")

    for path, rows in ((base + "_raw.csv", raw_rows), (base + "_summary.csv", summary)):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    log_path = base + ".log"
    with open(log_path, "w") as f:
        f.write("SEARCH METHOD BATCH COMPARISON\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"Machine: {platform.platform()} | Python {platform.python_version()}\n")
        f.write(f"Protocol: {exp['description']}\n")
        f.write(f"Base seed: {exp['base_seed']}   Repeats per config: {exp['repeats']}\n")
        f.write(f"Start/goal placement: {exp['placement']}\n")
        f.write(f"Movements: {exp['movements']}   Heuristics: {exp['heuristics']}\n\n")
        f.write("Notes:\n")
        f.write("  time   = wall clock in milliseconds (time.perf_counter_ns), mean ± std, "
                "best/worst over repeats\n")
        f.write("  mem    = process memory peak during search in KB (tracemalloc), "
                "measured in a separate run\n")
        f.write("  footpr = algorithmic footprint, peak (frontier + explored) node count\n")
        f.write("  b avg/max = observed branching factor (children added per expansion)\n")
        f.write("  optimal = optimal path cost / found path cost (1.000 = optimal)\n\n")

        f.write("SEARCH SPACES\n")
        for s in spaces:
            f.write(f"  #{s['id']:2d}: {s['n']}x{s['n']}, density {s['density']:.0%}, "
                    f"seed {s['seed']}, start {s['start']}, goal {s['goal']}\n")
        f.write("\n")

        header = (f"{'sp':>3} {'size':>5} {'dens':>5} {'move':>6} {'method':>7} "
                  f"{'heur':>10} {'moves':>6} {'cost':>8} {'optimal':>8} "
                  f"{'expand':>7} {'gener':>7} {'b avg/max':>10} {'footpr':>7} "
                  f"{'time ms (mean±std) [best..worst]':>36} {'mem KB':>9}\n")
        f.write(header)
        f.write("-" * len(header) + "\n")
        for r in summary:
            time_txt = (f"{r['time_mean']:.3f}±{r['time_std']:.3f} "
                        f"[{r['time_best']:.3f}..{r['time_worst']:.3f}]")
            f.write(f"{r['space']:>3} {r['n']:>3}x{r['n']:<2}{r['density']:>5.0%} "
                    f"{r['movement']:>6} {r['method']:>7} {r['heuristic']:>10} "
                    f"{_fmt(r['path_moves']):>6} {_fmt(r['path_cost']):>8} "
                    f"{_fmt(r['optimality']):>8} {r['expanded']:>7} {r['generated']:>7} "
                    f"{r['branching_avg']:>5.2f}/{r['branching_max']:<4} "
                    f"{r['peak_footprint']:>7} {time_txt:>36} {r['mem_mean']:>9.1f}\n")

        f.write("\nAVERAGES PER METHOD (across all spaces)\n")
        for (method, heur, move), rows in _group(summary).items():
            f.write(f"  {move:6s} {method:7s} {heur:10s}  "
                    f"time {statistics.mean(r['time_mean'] for r in rows):9.3f} ms  "
                    f"expanded {statistics.mean(r['expanded'] for r in rows):9.1f}  "
                    f"footprint {statistics.mean(r['peak_footprint'] for r in rows):8.1f}  "
                    f"mem {statistics.mean(r['mem_mean'] for r in rows):8.1f} KB  "
                    f"optimality {_mean_opt(rows)}\n")
    return log_path


def _fmt(value):
    return "-" if value is None else value


def _group(summary):
    """Group summary rows by (method, heuristic, movement), keeping test order."""
    groups = {}
    for r in summary:
        groups.setdefault((r["method"], r["heuristic"], r["movement"]), []).append(r)
    return groups


def _mean_opt(rows):
    vals = [r["optimality"] for r in rows if r["optimality"] is not None]
    return f"{statistics.mean(vals):.3f}" if vals else "-"


def show_batch_window(summary, exp, block=True):
    """
    Open a results window: comparison table + three charts.
    block=False when called from the control panel (its window loop is already running).
    """
    groups = _group(summary)
    names = [f"{m}{'' if h == '-' else '/' + h[:5]}\n{mv}" for (m, h, mv) in groups]
    mean = lambda rows, key: statistics.mean(r[key] for r in rows)

    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title("Batch Comparison Results")
    fig.suptitle(f"Batch comparison - {exp['description']} "
                 f"({exp['repeats']} repeats, base seed {exp['base_seed']})",
                 fontweight="bold")

    # Table: averages per method/heuristic/movement
    ax_t = fig.add_subplot(2, 2, 1)
    ax_t.axis("off")
    cells = []
    for (m, h, mv), rows in groups.items():
        cells.append([m, h, mv,
                      f"{mean(rows, 'time_mean'):.3f}",
                      f"{mean(rows, 'mem_mean'):.0f}",
                      f"{mean(rows, 'expanded'):.0f}",
                      f"{mean(rows, 'peak_footprint'):.0f}",
                      _mean_opt(rows)])
    table = ax_t.table(cellText=cells,
                       colLabels=["Method", "Heur.", "Move", "Time ms", "Mem KB",
                                  "Expanded", "Footprint", "Optimal"],
                       cellLoc="center", bbox=[0, 0, 1, 1])   # fit inside the panel
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    ax_t.set_title("Averages across all search spaces")

    x = range(len(names))
    # Chart: mean runtime with best..worst spread
    ax_time = fig.add_subplot(2, 2, 2)
    means = [mean(rows, "time_mean") for rows in groups.values()]
    lows = [m - min(r["time_best"] for r in rows) for m, rows in zip(means, groups.values())]
    highs = [max(r["time_worst"] for r in rows) - m for m, rows in zip(means, groups.values())]
    ax_time.bar(x, means, yerr=[lows, highs], capsize=2, color="steelblue")
    ax_time.set_yscale("log")
    ax_time.set_ylabel("ms (log scale)")
    ax_time.set_title("Runtime: mean, whiskers = best..worst")
    ax_time.set_xticks(list(x), names, rotation=90, fontsize=6)

    # Chart: search effort
    ax_eff = fig.add_subplot(2, 2, 3)
    ax_eff.bar(x, [mean(rows, "expanded") for rows in groups.values()], color="seagreen")
    ax_eff.set_yscale("log")
    ax_eff.set_ylabel("nodes expanded (log scale)")
    ax_eff.set_title("Search effort")
    ax_eff.set_xticks(list(x), names, rotation=90, fontsize=6)

    # Chart: runtime vs grid size (one line per method, A* with octile/manhattan)
    ax_sc = fig.add_subplot(2, 2, 4)
    for (m, h, mv), rows in groups.items():
        if h not in ("-", "octile") or mv != exp["movements"][-1]:
            continue
        sizes = sorted({r["n"] for r in rows})
        ys = [statistics.mean(r["time_mean"] for r in rows if r["n"] == s) for s in sizes]
        ax_sc.plot(sizes, ys, marker="o", label=f"{m}{'' if h == '-' else '/' + h}")
    ax_sc.set_yscale("log")
    ax_sc.set_xlabel("grid size N (N x N)")
    ax_sc.set_ylabel("ms (log scale)")
    ax_sc.set_title(f"Scaling: runtime vs size ({exp['movements'][-1]})")
    ax_sc.legend(fontsize=7)

    fig.tight_layout()
    if block:
        plt.show()
    else:
        fig.show()
    return fig


if __name__ == "__main__":
    run_batch(sys.argv[1] if len(sys.argv) > 1 else None)