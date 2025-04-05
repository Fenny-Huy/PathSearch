import sys
import heapq
from collections import deque
import math

import networkx as nx
import matplotlib.pyplot as plt
import time

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

def bfs(edges, origin, destinations):
    queue = deque([(origin, [origin], 0)])  # (current_node, path, total_cost)
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


def bfs_with_visualization(nodes, edges, origin, destinations):
    # Create a graph using networkx
    G = nx.Graph()
    for node, coords in nodes.items():
        G.add_node(node, pos=coords)
    for node, neighbors in edges.items():
        for neighbor, cost in neighbors:
            G.add_edge(node, neighbor, weight=cost)

    # Get positions for nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Initialize visualization
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax)
    plt.title("BFS Visualization")
    plt.show()

    # BFS algorithm with visualization
    queue = deque([(origin, [origin], 0)])
    visited = set()

    while queue:
        # Draw the current state of the graph
        current_nodes = [node for node, _, _ in queue]
        nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=current_nodes, node_color='yellow', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=list(visited), node_color='green', ax=ax)
        plt.pause(1)  # Pause to visualize the current state

        node, path, cost = queue.popleft()
        if node in destinations:
            print(f"Destination {node} reached!")
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red', ax=ax)
            plt.pause(1)
            return path, cost

        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in edges.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor], cost + edge_cost))

    print("No path found.")
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


def heuristic(node, goal, nodes):
    x1, y1 = nodes[node]
    x2, y2 = nodes[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def heuristic_a_star(node, goal, nodes):
    x1, y1 = nodes[node]
    x2, y2 = nodes[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def heuristic_gbfs(node, goal, nodes):
    x1, y1 = nodes[node]
    x2, y2 = nodes[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def heuristic_informed_custom(node, goal, nodes):
    x1, y1 = nodes[node]
    x2, y2 = nodes[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def a_star(nodes, edges, origin, destinations):
    pq = [(0, origin, [origin], 0)]
    visited = set()
    goal = min(destinations, key=lambda d: heuristic(origin, d, nodes))
    
    while pq:
        _, node, path, cost = heapq.heappop(pq)
        if node in destinations:
            return path, cost
        
        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in edges.get(node, []):
                new_cost = cost + edge_cost
                f_score = new_cost + heuristic(neighbor, goal, nodes)
                heapq.heappush(pq, (f_score, neighbor, path + [neighbor], new_cost))
    
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

def bidirectional_a_star(nodes, edges, origin, destinations):
    goal = min(destinations, key=lambda d: heuristic_informed_custom(origin, d, nodes))
    forward_pq = [(0, origin, [origin], 0)]  # (f_score, current_node, path, total_cost)
    backward_pq = [(0, goal, [goal], 0)]    # (f_score, current_node, path, total_cost)
    forward_visited = {}
    backward_visited = {}

    while forward_pq and backward_pq:
        # Forward search
        _, node, path, cost = heapq.heappop(forward_pq)
        if node in backward_visited:
            return path + backward_visited[node][::-1][1:], cost + backward_visited[node][1]
        if node not in forward_visited:
            forward_visited[node] = (path, cost)
            for neighbor, edge_cost in edges.get(node, []):
                new_cost = cost + edge_cost
                f_score = new_cost + heuristic_informed_custom(neighbor, goal, nodes)
                heapq.heappush(forward_pq, (f_score, neighbor, path + [neighbor], new_cost))

        # Backward search
        _, node, path, cost = heapq.heappop(backward_pq)
        if node in forward_visited:
            return forward_visited[node][0] + path[::-1][1:], forward_visited[node][1] + cost
        if node not in backward_visited:
            backward_visited[node] = (path, cost)
            for neighbor, edge_cost in edges.get(node, []):
                new_cost = cost + edge_cost
                f_score = new_cost + heuristic_informed_custom(neighbor, origin, nodes)
                heapq.heappush(backward_pq, (f_score, neighbor, path + [neighbor], new_cost))

    return None, None


# end of methods section




# main function
def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return
    
    filename = sys.argv[1]
    method = sys.argv[2].lower()
    nodes, edges, origin, destinations = parse_input_file(filename)
    
    if method == 'bfs':
        path, cost = bfs(edges, origin, destinations)
    elif method == 'dfs':
        path, cost = dfs(edges, origin, destinations)
    elif method == 'a*':
        path, cost = a_star(nodes, edges, origin, destinations)
    elif method == 'gbfs':
        path, cost = greedy_best_first_search(nodes, edges, origin, destinations)
    elif method == 'bfs_v':
        path, cost = bfs_with_visualization(nodes, edges, origin, destinations)
    elif method == 'bidirectional_a*':
        path, cost = bidirectional_a_star(nodes, edges, origin, destinations)
    else:
        print(f"Unknown method: {method}")
        return
    
    if path:
        print(f"{path}")
        print(f"{filename} {method}")
        print(f"{path[-1]} {len(path)}")
        print(" -> ".join(path))
        print(f"Cost: {cost}")
    else:
        print(f"No path found using {method}")

if __name__ == "__main__":
    main()
