import sys
import heapq
from collections import deque
import math

import networkx as nx
import matplotlib.pyplot as plt

# read input file
def parse_input_file(filename):
    nodes = {}
    edges = {}
    origin = None
    destinations = set()
    
    with open(filename, 'r') as file:
        section = None
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("Nodes:"):
                section = "nodes"
                continue
            elif line.startswith("Edges:"):
                section = "edges"
                continue
            elif line.startswith("Origin:"):
                section = "origin"
                continue
            elif line.startswith("Destinations:"):
                section = "destinations"
                continue
            
            if section == "nodes":
                node_id, coords = line.split(":")
                x, y = map(float, coords.strip("() ").split(","))
                nodes[node_id.strip()] = (x, y)
            elif section == "edges":
                edge, cost = line.split(":")
                node1, node2 = edge.strip("() ").split(",")
                if node1 not in edges:
                    edges[node1] = []
                edges[node1].append((node2, float(cost.strip())))
            elif section == "origin":
                origin = line.strip()
            elif section == "destinations":
                destinations = set(line.replace(";", "").split())
    
    return nodes, edges, origin, destinations

# placeholder for the methods

# uninformed methods

def bfs(edges, origin, destinations):
    queue = deque([(origin, [origin], 0)])
    visited = set()

    while queue:
        node, path, cost = queue.popleft()

        if node in destinations:
            return path, cost

        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in edges.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor], cost + edge_cost))

    return None, None

def dfs(edges, origin, destinations):
    stack = [(origin, [origin], 0)]  # (current_node, path, total_cost)
    visited = set()

    while stack:
        node, path, cost = stack.pop()

        if node in destinations:
            return path, cost

        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in edges.get(node, []):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], cost + edge_cost))

    return None, None

def ucs(edges, origin, destinations): # Uniform cost search
    pq = [(0, origin, [origin])]  # (cumulative_cost, current_node, path)
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node in destinations:
            return path, cost

        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in edges.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

    return None, None

#informed methods

def heuristic(node, goal, nodes):
    x1, y1 = nodes[node]
    x2, y2 = nodes[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def a_star(nodes, edges, origin, destinations):
    pq = [(0, 0, origin, [origin])]
    best_costs = {origin: 0}
    goal = min(destinations, key=lambda d: heuristic(origin, d, nodes))

    visited = set()

    while pq:
        f_score, cost, node, path = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node in destinations:
            return path, cost

        for neighbor, edge_cost in edges.get(node, []):
            new_cost = cost + edge_cost
            if neighbor not in best_costs or new_cost < best_costs[neighbor]:
                best_costs[neighbor] = new_cost
                f_score = new_cost + heuristic(neighbor, goal, nodes)
                heapq.heappush(pq, (f_score, new_cost, neighbor, path + [neighbor]))

    return None, None

def greedy_best_first_search(nodes, edges, origin, destinations):
    # Priority queue: (heuristic_cost, current_node, path)
    pq = [(heuristic(origin, min(destinations, key=lambda d: heuristic(origin, d, nodes)), nodes), origin, [origin], 0)]
    print(f"goal: {min(destinations, key=lambda d: heuristic(origin, d, nodes))}")
    print(f"Initial queue: {pq}")
    visited = set()
    goal = min(destinations, key=lambda d: heuristic(origin, d, nodes))  # Closest destination based on heuristic
    print(f"Goal: {goal}")

    while pq:
        # Get the node with the smallest heuristic value
        _, node, path, cost = heapq.heappop(pq)
        print(f"{_**2}, {node}, {path}, {cost}")

        # If the node is a destination, return the path and total cost
        if node in destinations:
            return path, cost

        if node not in visited:
            visited.add(node)

            # Add neighbors to the priority queue
            for neighbor, edge_cost in edges.get(node, []):
                if neighbor not in visited:
                    h_cost = heuristic(neighbor, goal, nodes)
                    heapq.heappush(pq, (h_cost, neighbor, path + [neighbor], cost + edge_cost))

    return None, None

def hsm_search(nodes, edges, origin, destinations):
    open_set = []
    h = min(heuristic(origin, goal, nodes) for goal in destinations)
    heapq.heappush(open_set, (h, origin, 0, 0, [origin]))  # (f, node, moves, cost_so_far, path)
    visited = set()

    while open_set:
        f, current, moves, cost_so_far, path = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current in destinations:
            return path, cost_so_far

        for neighbor, edge_cost in edges.get(current, []):
            if neighbor not in visited:
                h_new = min(heuristic(neighbor, goal, nodes) for goal in destinations)
                f_new = (moves + 1) + h_new
                heapq.heappush(open_set, (f_new, neighbor, moves + 1, cost_so_far + edge_cost, path + [neighbor]))

    return None, None

# end of methods section

# start of visualized methods section

def bfs_with_visualization(nodes, edges, origin, destinations):
    visited = set()
    queue = deque([(origin, [origin], 0)])
    edge_list = [(u, v, cost) for u in edges for v, cost in edges[u]]

    G = nx.DiGraph()
    for node, (x, y) in nodes.items():
        G.add_node(node, pos=(x, y))
    for u, v, cost in edge_list:
        G.add_edge(u, v, weight=cost)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 7))
    used_edges = []

    while queue:
        node, path, cost = queue.popleft()

        if node in visited:
            continue

        visited.add(node)

        if node in destinations:
            animate_graph(G, pos, edge_labels, origin, destinations,
                              visited, [n for n, _, _ in queue], used_edges,
                              title="BFS - Final Node Reached", ax=ax)
            final_path = path
            final_cost = cost
            break

        for neighbor, edge_cost in edges.get(node, []):
            used_edges.append((node, neighbor))
            animate_graph(G, pos, edge_labels, origin, destinations,
                              visited, [n for n, _, _ in queue], used_edges,
                              title=f"BFS - Evaluating edge {node} → {neighbor}", ax=ax)
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor], cost + edge_cost))

    if node in destinations:
        animate_final_path(G, pos, edge_labels, final_path, ax, "BFS")

    plt.ioff()
    plt.show()

    return (final_path, final_cost) if node in destinations else (None, None)

def dfs_with_visualization(nodes, edges, origin, destinations):
    visited = set()
    stack = [(origin, [origin], 0)]
    edge_list = [(u, v, cost) for u in edges for v, cost in edges[u]]

    G = nx.DiGraph()
    for node, (x, y) in nodes.items():
        G.add_node(node, pos=(x, y))
    for u, v, cost in edge_list:
        G.add_edge(u, v, weight=cost)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 7))
    used_edges = []

    while stack:
        node, path, cost = stack.pop()

        if node in visited:
            continue

        visited.add(node)

        if node in destinations:
            animate_graph(G, pos, edge_labels, origin, destinations,
                              visited, [n for n, _, _ in stack], used_edges,
                              title="DFS - Final Node Reached", ax=ax)
            final_path = path
            final_cost = cost
            break

        for neighbor, edge_cost in edges.get(node, []):
            used_edges.append((node, neighbor))
            animate_graph(G, pos, edge_labels, origin, destinations,
                              visited, [n for n, _, _ in stack], used_edges,
                              title=f"DFS - Checking {node} → {neighbor}", ax=ax)
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor], cost + edge_cost))

    if node in destinations:
        animate_final_path(G, pos, edge_labels, final_path, ax, "DFS")

    plt.ioff()
    plt.show()
    
    return (final_path, final_cost) if node in destinations else (None, None)

def ucs_with_visualization(nodes, edges, origin, destinations):
    visited = set()
    pq = [(0, origin, [origin])]
    edge_list = [(u, v, cost) for u in edges for v, cost in edges[u]]

    G = nx.DiGraph()
    for node, (x, y) in nodes.items():
        G.add_node(node, pos=(x, y))
    for u, v, cost in edge_list:
        G.add_edge(u, v, weight=cost)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 7))
    used_edges = []

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)

        if node in destinations:
            animate_graph(G, pos, edge_labels, origin, destinations,
                          visited, [n for _, n, _ in pq], used_edges,
                          title="UCS - Final Node Reached", ax=ax)
            final_path = path
            final_cost = cost
            break

        for neighbor, edge_cost in edges.get(node, []):
            used_edges.append((node, neighbor))
            animate_graph(G, pos, edge_labels, origin, destinations,
                          visited, [n for _, n, _ in pq], used_edges,
                          title=f"UCS - Checking {node} → {neighbor}", ax=ax)
            if neighbor not in visited:
                heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

    if node in destinations:
        animate_final_path(G, pos, edge_labels, final_path, ax, "UCS")

    plt.ioff()
    plt.show()
    return (final_path, final_cost) if node in destinations else (None, None)

def a_star_with_visualization(nodes, edges, origin, destinations):
    pq = [(0, 0, origin, [origin])]
    best_costs = {origin: 0}
    goal = min(destinations, key=lambda d: heuristic(origin, d, nodes))

    visited = set()
    used_edges = []

    edge_list = [(u, v, cost) for u in edges for v, cost in edges[u]]

    G = nx.DiGraph()
    for node, (x, y) in nodes.items():
        G.add_node(node, pos=(x, y))
    for u, v, cost in edge_list:
        G.add_edge(u, v, weight=cost)
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 7))

    while pq:
        f_score, cost, node, path = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)

        if node in destinations:
            animate_graph(G, pos, edge_labels, origin, destinations,
                              visited, [n for _, _, n, _ in pq], used_edges,
                              title="A* - Found Path", ax=ax)
            final_path = path
            final_cost = cost
            break  # the final path was found and displayed, breaking while loop to highlight it and end alg

        for neighbor, edge_cost in edges.get(node, []):
            new_cost = cost + edge_cost
            used_edges.append((node, neighbor))

            animate_graph(G, pos, edge_labels, origin, destinations,
                              visited, [n for _, _, n, _ in pq], used_edges,
                              title=f"A* - Evaluating edge {node} → {neighbor}", ax=ax)

            if neighbor not in best_costs or new_cost < best_costs[neighbor]:
                best_costs[neighbor] = new_cost
                f_score = new_cost + heuristic(neighbor, goal, nodes)
                heapq.heappush(pq, (f_score, new_cost, neighbor, path + [neighbor]))

    if node in destinations:
        animate_final_path(G, pos, edge_labels, final_path, ax, "A*")

    plt.ioff()
    plt.show()
    
    return (final_path, final_cost) if node in destinations else (None, None)

def greedy_best_first_search_with_visualization(nodes, edges, origin, destinations):
    visited = set()
    goal = min(destinations, key=lambda d: heuristic(origin, d, nodes))
    pq = [(heuristic(origin, goal, nodes), origin, [origin], 0)]
    edge_list = [(u, v, cost) for u in edges for v, cost in edges[u]]

    G = nx.DiGraph()
    for node, (x, y) in nodes.items():
        G.add_node(node, pos=(x, y))
    for u, v, cost in edge_list:
        G.add_edge(u, v, weight=cost)

    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 7))
    used_edges = []

    while pq:
        _, node, path, cost = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)

        if node in destinations:
            animate_graph(G, pos, edge_labels, origin, destinations,
                              visited, [n for _, n, _, _ in pq], used_edges,
                              title="GBFS - Final Node Reached", ax=ax)
            final_path = path
            final_cost = cost
            break

        for neighbor, edge_cost in edges.get(node, []):
            used_edges.append((node, neighbor))
            animate_graph(G, pos, edge_labels, origin, destinations,
                              visited, [n for _, n, _, _ in pq], used_edges,
                              title=f"GBFS - Checking {node} → {neighbor}", ax=ax)
            if neighbor not in visited:
                h_cost = heuristic(neighbor, goal, nodes)
                heapq.heappush(pq, (h_cost, neighbor, path + [neighbor], cost + edge_cost))

    if node in destinations:
        animate_final_path(G, pos, edge_labels, final_path, ax, "GBFS")

    plt.ioff()
    plt.show()

    return (final_path, final_cost) if node in destinations else (None, None)

def hsm_search_with_visualization(nodes, edges, origin, destinations):
    open_set = []
    h = min(heuristic(origin, goal, nodes) for goal in destinations)
    heapq.heappush(open_set, (h, origin, 0, 0, [origin]))
    visited = set()
    used_edges = []

    # Setup visualization
    edge_list = [(u, v, cost) for u in edges for v, cost in edges[u]]
    G = nx.DiGraph()
    for node, (x, y) in nodes.items():
        G.add_node(node, pos=(x, y))
    for u, v, cost in edge_list:
        G.add_edge(u, v, weight=cost)
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 7))

    while open_set:
        f, current, moves, cost_so_far, path = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current in destinations:
            animate_graph(G, pos, edge_labels, origin, destinations,
                          visited, [n for _, n, _, _, _ in open_set], used_edges,
                          title="HSM - Final Node Reached", ax=ax)
            final_path = path
            final_cost = cost_so_far
            break

        for neighbor, edge_cost in edges.get(current, []):
            if neighbor not in visited:
                used_edges.append((current, neighbor))
                animate_graph(G, pos, edge_labels, origin, destinations,
                              visited, [n for _, n, _, _, _ in open_set], used_edges,
                              title=f"HSM - Checking {current} → {neighbor}", ax=ax)
                h_new = min(heuristic(neighbor, goal, nodes) for goal in destinations)
                f_new = (moves + 1) + h_new
                heapq.heappush(open_set, (f_new, neighbor, moves + 1, cost_so_far + edge_cost, path + [neighbor]))

    if current in destinations:
        animate_final_path(G, pos, edge_labels, final_path, ax, "HSM")

    plt.ioff()
    plt.show()

    return (final_path, final_cost) if current in destinations else (None, None)


# end of visualized methods section

# start of visualization section

def animate_graph(G, pos, edge_labels, origin, destinations,
                      visited, frontier, active_edges, title, ax):
    ax.clear()

    # Draw base graph
    nx.draw(G, pos, with_labels=True, node_color='lightgray', node_size=800,
            edge_color='gray', font_size=12, ax=ax,
            arrows=True, connectionstyle="arc3,rad=0.1")

    # Draw actively evaluated edges
    nx.draw_networkx_edges(G, pos, edgelist=active_edges, edge_color="green", width=2.5, ax=ax,
                           connectionstyle="arc3,rad=0.1")

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,
                                 label_pos=0.4, rotate=False,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8), ax=ax)

    # Highlight nodes in pathfinding
    nx.draw_networkx_nodes(G, pos, nodelist=frontier, node_color="yellow", node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=visited, node_color="green", node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[origin], node_color="red", node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=destinations, node_color="lime", node_size=800, ax=ax)

    plt.title(title)
    plt.pause(0.5)

def animate_final_path(G, pos, edge_labels, path, ax, method):
    ax.clear()

    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    edge_colors = ['blue' if (u, v) in path_edges else 'gray' for u, v in G.edges()]

    # Draw base graph
    nx.draw(G, pos, with_labels=True, node_color='lightgray',
            edge_color=edge_colors, edgecolors="black",
            node_size=800, font_size=12, ax=ax,
            arrows=True, connectionstyle="arc3,rad=0.1")

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,
                                 label_pos=0.4, rotate=False,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8), ax=ax)

    # Highlight nodes in path
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="cyan", node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_color="red", node_size=800, ax=ax)  # origin
    nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_color="lime", node_size=800, ax=ax)  # destination

    plt.title(f"{method} - Final Path")
    plt.pause(2)

# end of visualization section

# main function
def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return
    
    filename = sys.argv[1]
    method = sys.argv[2].strip().lower()
    nodes, edges, origin, destinations = parse_input_file(filename)
    
    if method == 'bfs':
        path, cost = bfs(edges, origin, destinations)
    elif method == 'bfs_v':
        path, cost = bfs_with_visualization(nodes, edges, origin, destinations)
    elif method == 'dfs':
        path, cost = dfs(edges, origin, destinations)
    elif method == 'dfs_v':
        path, cost = dfs_with_visualization(nodes, edges, origin, destinations)
    elif method == 'ucs':
        path, cost = ucs(edges, origin, destinations)
    elif method == 'ucs_v':
        path, cost = ucs_with_visualization(nodes, edges, origin, destinations)
    elif method == 'a*':
        path, cost = a_star(nodes, edges, origin, destinations)
    elif method == 'a*_v':
        path, cost = a_star_with_visualization(nodes, edges, origin, destinations)
    elif method == 'gbfs':
        path, cost = greedy_best_first_search(nodes, edges, origin, destinations)
    elif method == 'gbfs_v':
        path, cost = greedy_best_first_search_with_visualization(nodes, edges, origin, destinations)
    elif method == 'hsm':
        path, cost = hsm_search(nodes, edges, origin, destinations)
    elif method == 'hsm_v':
        path, cost = hsm_search_with_visualization(nodes, edges, origin, destinations)
    else:
        print(f"Unknown method: {method}")
        return
    
    if path:
        print(f"Filename: {filename} Method: {method.capitalize()}")
        print(f"Destination: {path[-1]}, Path Length: {len(path)}")
        print(f"Path: {' -> '.join(path)}")
    else:
        print(f"No path found using {method}")

if __name__ == "__main__":
    main()
