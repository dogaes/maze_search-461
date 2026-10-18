import networkx as nx

def build_children(parent):
    children = {}
    for child, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(child)
    return children


def tree_layout(children, root):
    pos = {}
    next_x = [0]               # list so the inner function can change it

    def place(node, depth):
        kids = children.get(node, [])
        if not kids:                  # leaf: next free slot
            x = next_x[0]
            next_x[0] += 1
        else:                       # parent: average of children
            xs = [place(k, depth + 1) for k in kids]
            x = sum(xs) / len(xs)
        pos[node] = (x, -depth)
        return x

    place(root, 0)
    return pos

def local_subtree(children, parent, focus, up=2, down=2):
    # climb up to the grandparent (or stop early at the real root)
    root = focus
    levels_up = 0
    while levels_up < up and parent.get(root) is not None:
        root = parent[root]
        levels_up += 1

    # walk down from the view root, keeping only nodes within the depth limit
    max_depth = levels_up + down
    sub = {}
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if depth < max_depth and node in children:
            sub[node] = children[node]
            for kid in children[node]:
                stack.append((kid, depth + 1))
    return root, sub

def draw_tree(ax, start, goal, step, method="", local=False):
    children = build_children(step["parent"])
    if local and not step["done"] and step["current"] is not None:
        root, children = local_subtree(children, step["parent"], step["current"])
        view = "local view"
    else:
        root = start
        view = "full tree"

    pos = tree_layout(children, root)

    G = nx.DiGraph()
    G.add_node(root)

    for p, kids in children.items():
        for kid in kids:
            G.add_edge(p, kid)

    # same color scheme as grid_view
    path = set(step["path"] or [])
    frontier = set(step["frontier"])

    def color_for(node):
        if node == start:
            return "lightblue"
        if node == goal:
            return "pink"
        if node in path:
            return "yellow"
        if node == step["current"]:
            return "green"
        if node in frontier:
            return "lightgray"
        return "lightgreen"            # expanded

    colors = [color_for(node) for node in G.nodes()]

    # labels only when the tree is small enough to read
    show_labels = len(G) <= 40
    labels = {node: f"{node[0]},{node[1]}" for node in G.nodes()}

    ax.clear()
    nx.draw(G, pos, ax=ax,
            node_color=colors,
            node_size=300 if show_labels else 40,
            labels=labels if show_labels else None,
            with_labels=show_labels,
            font_size=7,
            edgecolors="gray",
            arrows=False)
    ax.set_title(f"{method} search tree ({view}): {len(G)} nodes shown")
