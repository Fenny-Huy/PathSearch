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


def a_star_search_cost(nodes, edges, origin, destinations):
    open_set = []
    heapq.heappush(open_set, (0, origin, 0, [origin]))  # (f, current, g, path)
    visited = set()
    
    while open_set:
        f, current, g, path = heapq.heappop(open_set)
        print(f"current node: {current}, f: {f}, g: {g}, path: {path}")
        
        if current in visited:
            continue
        visited.add(current)
        
        if current in destinations:
            return current, len(visited), path, g  # Return the total cost (g) along with the path
        
        for neighbor, cost in edges.get(current, []):
            if neighbor not in visited:
                g_new = g + cost
                h = min(heuristic(neighbor, goal, nodes) for goal in destinations)
                f_new = g_new + h
                heapq.heappush(open_set, (f_new, neighbor, g_new, path + [neighbor]))
    
    return None, len(visited), [], 0  # Return 0 cost if no path is found


def gbfs_search(nodes, edges, origin, destinations):
    open_set = []
    h = min(heuristic(origin, goal, nodes) for goal in destinations)
    print(f"min heuristic between destinations: {round(h**2)}")
    heapq.heappush(open_set, (h, origin, [origin]))
    visited = set()
    
    while open_set:
        h, current, path = heapq.heappop(open_set)
        print(f"current node: {current}, heuristic2: {round(h**2)}, path: {path}")
        
        if current in visited:
            continue
        visited.add(current)
        
        if current in destinations:
            return current, len(visited), path
        
        for neighbor, _ in edges.get(current, []):
            if neighbor not in visited:
                for goal in destinations:
                    h_test = heuristic(neighbor, goal, nodes)
                    print(f"neighbor: {neighbor}, goal: {goal}, heuristic2: {round(h_test**2)}, something: {_}")
                h_new = min(heuristic(neighbor, goal, nodes) for goal in destinations)
                heapq.heappush(open_set, (h_new, neighbor, path + [neighbor]))
    
    return None, len(visited), []

def gbfs_search_cost(nodes, edges, origin, destinations):
    open_set = []
    h = min(heuristic(origin, goal, nodes) for goal in destinations)
    print(f"min heuristic between destinations: {round(h**2)}")
    heapq.heappush(open_set, (h, origin, 0, [origin]))  # (heuristic, current_node, total_cost, path)
    visited = set()
    
    while open_set:
        h, current, cost, path = heapq.heappop(open_set)
        print(f"current node: {current}, heuristic2: {round(h**2)}, cost: {cost}, path: {path}")
        
        if current in visited:
            continue
        visited.add(current)
        
        if current in destinations:
            return current, len(visited), path, cost  # Return the total cost along with the path
        
        for neighbor, edge_cost in edges.get(current, []):
            if neighbor not in visited:
                for goal in destinations:
                    h_test = heuristic(neighbor, goal, nodes)
                    print(f"neighbor: {neighbor}, goal: {goal}, heuristic2: {round(h_test**2)}, edge cost: {edge_cost}")
                h_new = min(heuristic(neighbor, goal, nodes) for goal in destinations)
                heapq.heappush(open_set, (h_new, neighbor, cost + edge_cost, path + [neighbor]))
        

        print(f"open_set: {open_set}")
    
    return None, len(visited), [], 0  # Return 0 cost if no path is found


def hsm_search(nodes, edges, origin, destinations):
    open_set = []
    h = min(heuristic(origin, goal, nodes) for goal in destinations)
    heapq.heappush(open_set, (h, origin, 0, [origin]))  # (f = moves + h, node, moves, path)
    visited = set()

    while open_set:
        f, current, moves, path = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)

        if current in destinations:
            return current, len(visited), path
        
        for neighbor, _ in edges.get(current, []):
            if neighbor not in visited:
                h_new = min(heuristic(neighbor, goal, nodes) for goal in destinations)
                f_new = (moves + 1) + h_new
                heapq.heappush(open_set, (f_new, neighbor, moves + 1, path + [neighbor]))
    
    return None, len(visited), []

def hms_search(nodes, edges, origin, destinations):
    """
    Custom search: Heuristic Moves Search (HMS)
    This method finds the path with the fewest moves by treating every move as a cost of 1,
    and uses a heuristic to guide the search towards the nearest destination.
    Evaluation function: f = (number of moves so far) + h, where h is the heuristic estimate.
    """
    open_set = []
    initial_h = min(heuristic(origin, goal, nodes) for goal in destinations)
    heapq.heappush(open_set, (initial_h, origin, 0, [origin]))  # (f, node, moves, path)
    visited = set()
    
    while open_set:
        f, current, moves, path = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)
        
        if current in destinations:
            return current, len(visited), path
        
        for neighbor, _ in edges.get(current, []):
            if neighbor not in visited:
                h_new = min(heuristic(neighbor, goal, nodes) for goal in destinations)
                f_new = (moves + 1) + h_new
                heapq.heappush(open_set, (f_new, neighbor, moves + 1, path + [neighbor]))
    
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
    elif method == "gbfs_cost":
        goal, num_nodes, path, cost = gbfs_search_cost(nodes, edges, origin, destinations)
    elif method == "a*_cost":
        goal, num_nodes, path, cost = a_star_search_cost(nodes, edges, origin, destinations)
    elif method == "hsm":
        goal, num_nodes, path = hsm_search(nodes, edges, origin, destinations)
    elif method == "hms":
        goal, num_nodes, path = hms_search(nodes, edges, origin, destinations)
    else:
        print(f"Unknown method '{method}'. Use 'a*' or 'gbfs'.")
        return

    print(f"{filename} {method}")
    if goal:
        print(f"{goal} {num_nodes}")
        print(" ".join(path))
        print(f"Total cost: {cost}") if method == "gbfs_cost" or method == "a*_cost" else None
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
