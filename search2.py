import sys
import heapq
import math

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

def heuristic(node1, node2, nodes):
    x1, y1 = nodes[node1]
    x2, y2 = nodes[node2]
    return math.hypot(x2 - x1, y2 - y1)

def a_star_search(nodes, edges, origin, destinations):
    open_set = []
    heapq.heappush(open_set, (0, origin, 0, [origin]))  # (f, current, g, path)
    visited = set()
    
    while open_set:
        f, current, g, path = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current in destinations:
            return current, len(visited), path
        
        for neighbor, cost in edges.get(current, []):
            if neighbor not in visited:
                g_new = g + cost
                h = min(heuristic(neighbor, goal, nodes) for goal in destinations)
                f_new = g_new + h
                heapq.heappush(open_set, (f_new, neighbor, g_new, path + [neighbor]))
    
    return None, len(visited), []

def gbfs_search(nodes, edges, origin, destinations):
    open_set = []
    h = min(heuristic(origin, goal, nodes) for goal in destinations)
    print(f"min heuristic between destinations: {h**2}")
    heapq.heappush(open_set, (h, origin, [origin]))
    visited = set()
    
    while open_set:
        h, current, path = heapq.heappop(open_set)
        print(f"current node: {current}, heuristic2: {h**2}, path: {path}")
        
        if current in visited:
            continue
        visited.add(current)
        
        if current in destinations:
            return current, len(visited), path
        
        for neighbor, _ in edges.get(current, []):
            if neighbor not in visited:
                h_new = min(heuristic(neighbor, goal, nodes) for goal in destinations)
                heapq.heappush(open_set, (h_new, neighbor, path + [neighbor]))
    
    return None, len(visited), []

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <input_file> <method>")
        return
    
    filename = sys.argv[1]
    method = sys.argv[2].lower()

    nodes, edges, origin, destinations = parse_input_file(filename)

    if method == "a*":
        goal, num_nodes, path = a_star_search(nodes, edges, origin, destinations)
    elif method == "gbfs":
        goal, num_nodes, path = gbfs_search(nodes, edges, origin, destinations)
    else:
        print(f"Unknown method '{method}'. Use 'a*' or 'gbfs'.")
        return

    print(f"{filename} {method}")
    if goal:
        print(f"{goal} {num_nodes}")
        print(" ".join(path))
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
