import heapq
import itertools
from collections import deque

from agent import get_neighbors, step_cost
from heuristics import HEURISTICS

METHODS = ["BFS", "DFS", "IDDFS", "GREEDY", "ASTAR"]
INFORMED = ("GREEDY", "ASTAR")          # methods that use a heuristic


def reconstruct_path(parent, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def path_cost(path):
    """Total cost of a path: 1 per straight step, sqrt(2) per diagonal step."""
    return sum(step_cost(a, b) for a, b in zip(path, path[1:]))


def _snapshot(current, children, frontier, parent, order, done, path,
              light=False, explored_size=0):
    if light:
        return {
            "current": current,
            "n_children": len(children),
            "frontier_size": frontier if isinstance(frontier, int) else len(frontier),
            "explored_size": explored_size,
            "expanded": len(order),
            "done": done,
            "path": path,
        }
    return {
        "current": current,
        "children": children,
        "frontier": list(frontier),
        "visited": set(parent),
        "parent": dict(parent),
        "order": list(order),
        "done": done,
        "path": path,
    }


def search(grid, start, goal, method="bfs", diagonal=False,
           heuristic="manhattan", light=False):
    method = method.upper()
    if method in ("ASTAR", "A*"):
        yield from best_first_search(grid, start, goal, True, diagonal, heuristic, light)
        return
    if method == "GREEDY":
        yield from best_first_search(grid, start, goal, False, diagonal, heuristic, light)
        return
    if method in ("IDDFS", "ID-DFS"):
        yield from iddfs(grid, start, goal, diagonal, light)
        return

    # ---- BFS / DFS (Part A code; only the yields and the diagonal option changed) ----
    frontier = deque([start])
    visited = {start}
    parent = {start: None}
    order = []

    while frontier:
        if method == "BFS":
            current = frontier.popleft()   # queue - oldest first - level by level
        else:
            current = frontier.pop()       # stack - newest first - deep
        order.append(current)

        if current == goal:
            yield _snapshot(current, [], frontier, parent, order, True,
                            reconstruct_path(parent, start, goal),
                            light, explored_size=len(order))
            return

        children = []
        for neighbor in get_neighbors(grid, current, diagonal):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                children.append(neighbor)
                frontier.append(neighbor)
        yield _snapshot(current, children, frontier, parent, order, False, None,
                        light, explored_size=len(order))

    yield _snapshot(None, [], [], parent, order, True, None,
                    light, explored_size=len(order))


def best_first_search(grid, start, goal, use_g, diagonal, heuristic, light=False):
    h = HEURISTICS[heuristic]
    tie = itertools.count()                 # last tie-breaker: earlier push wins
    g = {start: 0.0}                        # best known cost from start
    parent = {start: None}
    # Heap entries: (priority, h, tie, cell). On equal priority the cell with the
    # smaller h (closer to the goal) is expanded first, which saves a lot of work.
    frontier = [(h(start, goal), h(start, goal), next(tie), start)]
    closed = set()                          # cells already expanded
    order = []

    def frontier_view():
        # Light mode only needs the size; visual mode needs the actual cells
        if light:
            return len(frontier)
        return list(dict.fromkeys(c for _, _, _, c in frontier if c not in closed))

    while frontier:
        _, _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue                        # stale copy: a cheaper one was already used
        closed.add(current)
        order.append(current)

        if current == goal:
            yield _snapshot(current, [], frontier_view(), parent, order, True,
                            reconstruct_path(parent, start, goal),
                            light, explored_size=len(closed))
            return

        children = []
        for nb in get_neighbors(grid, current, diagonal):
            if nb in closed:
                continue
            new_g = g[current] + step_cost(current, nb)
            if use_g:
                better = nb not in g or new_g < g[nb]     # A*: keep cheapest route
            else:
                better = nb not in parent                 # Greedy: first route found
            if better:
                g[nb] = new_g
                parent[nb] = current
                children.append(nb)
                h_nb = h(nb, goal)
                priority = new_g + h_nb if use_g else h_nb
                heapq.heappush(frontier, (priority, h_nb, next(tie), nb))

        yield _snapshot(current, children, frontier_view(), parent, order, False, None,
                        light, explored_size=len(closed))

    yield _snapshot(None, [], [], parent, order, True, None,
                    light, explored_size=len(closed))


def iddfs(grid, start, goal, diagonal, light=False):
    limit = 0
    total_expanded = 0

    while True:
        parent = {start: None}
        best_depth = {start: 0}
        stack = [(start, 0)]
        order = []            # expansions in THIS round (what the grid view paints)
        cutoff = False        # True if some branch was stopped by the limit

        def make_step(current, children, done, path):
            frontier = len(stack) if light else [c for c, _ in stack]
            step = _snapshot(current, children, frontier, parent, order, done, path,
                             light, explored_size=len(best_depth))
            step["limit"] = limit
            step["total_expanded"] = total_expanded
            if light:
                step["expanded"] = total_expanded   # count every re-expansion
            return step

        while stack:
            current, depth = stack.pop()
            if depth > best_depth[current]:
                continue      # stale entry: reached later by a shorter route
            order.append(current)
            total_expanded += 1

            if current == goal:
                yield make_step(current, [], True, reconstruct_path(parent, start, goal))
                return

            children = []
            if depth == limit:
                cutoff = True                  # deeper cells exist; try next round
            else:
                for nb in get_neighbors(grid, current, diagonal):
                    if depth + 1 < best_depth.get(nb, float("inf")):
                        best_depth[nb] = depth + 1
                        parent[nb] = current
                        children.append(nb)
                        stack.append((nb, depth + 1))

            yield make_step(current, children, False, None)

        if not cutoff:
            # Nothing was cut off, so every reachable cell was seen: no path exists
            yield make_step(None, [], True, None)
            return
        limit += 1


if __name__ == "__main__":
    from grid_gen import make_environment, grid_to_text, STUDENT_SEED
    grid, start, goal = make_environment(n=15, obstacle_ratio=0.2, seed=STUDENT_SEED)
    print(grid_to_text(grid, start, goal))

    for diagonal in (False, True):
        print(f"\n--- {'8-way' if diagonal else '4-way'} movement ---")
        for method, heur in [("BFS", "manhattan"), ("DFS", "manhattan"),
                             ("IDDFS", "manhattan"), ("GREEDY", "manhattan"),
                             ("ASTAR", "manhattan"), ("ASTAR", "octile")]:
            for step in search(grid, start, goal, method, diagonal, heur):
                pass
            if step["path"]:
                expanded = step.get("total_expanded", len(step["order"]))
                print(f"{method:6s} {heur:9s}: cost {path_cost(step['path']):6.2f}, "
                      f"moves {len(step['path']) - 1:3d}, expanded {expanded}")
            else:
                print(f"{method:6s} {heur:9s}: no path")