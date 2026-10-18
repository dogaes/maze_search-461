from collections import deque
from agent import get_neighbors


def reconstruct_path(parent, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def search(grid, start, goal, method="bfs"):
    frontier = deque([start])
    visited = {start}
    parent = {start: None}
    order = []

    while frontier:
        if method.upper() == "BFS":
            current = frontier.popleft()   # queue: oldest first → level by level
        else:
            current = frontier.pop()       # stack: newest first → dive deep
        order.append(current)

        if current == goal:
            yield {
                "current": current,
                "children": [],
                "frontier": list(frontier),
                "visited": set(visited),
                "parent": dict(parent),
                "order": list(order),
                "done": True,
                "path": reconstruct_path(parent, start, goal),
            }
            return
        
        children = []
        for neighbor in get_neighbors(grid, current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                children.append(neighbor)
                frontier.append(neighbor)
        yield {
            "current": current,
            "children": children,
            "frontier": list(frontier),
            "visited": set(visited),
            "parent": dict(parent),
            "order": list(order),
            "done": False,
            "path": None,
        }
    
    yield {
        "current": None,
        "children": [],
        "frontier": [],
        "visited": set(visited),
        "parent": dict(parent),
        "order": list(order),
        "done": True,
        "path": None,
    }

