import sys
import heapq
import math

def parse_input_file(filename):
    raw_nodes = {}
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
                raw_nodes[node_id.strip()] = (x, y)
            elif section == "edges":
                edge, cost = line.split(":")
                node1, node2 = edge.strip("() ").split(",")
                cost = float(cost.strip())
                if node1 not in edges:
                    edges[node1] = []
                edges[node1].append((node2, cost))
            elif section == "origin":
                origin = line.strip()
            elif section == "destinations":
                destinations = set(line.replace(";", "").split())

    # Deduplicate nodes: keep the one with lowest ID for each coordinate
    coord_to_node = {}
    node_map = {}  # maps old_node_id -> new_node_id

    for node_id in sorted(raw_nodes.keys(), key=int):
        coord = raw_nodes[node_id]
        if coord not in coord_to_node:
            coord_to_node[coord] = node_id
        node_map[node_id] = coord_to_node[coord]

    # Build final cleaned nodes dict
    nodes = {new_id: coord for coord, new_id in coord_to_node.items()}

    # Update edges using mapped node IDs, and keep lowest cost per (src, dest)
    edge_map = {}
    for src, neighbors in edges.items():
        src_new = node_map[src]
        for dest, cost in neighbors:
            dest_new = node_map[dest]
            key = (src_new, dest_new)
            if key not in edge_map or cost < edge_map[key]:
                edge_map[key] = cost

    # Rebuild new_edges dict from edge_map
    new_edges = {}
    for (src, dest), cost in edge_map.items():
        if src not in new_edges:
            new_edges[src] = []
        new_edges[src].append((dest, cost))

    # Update origin and destinations
    origin = node_map[origin]
    destinations = {node_map[d] for d in destinations}

    return nodes, new_edges, origin, destinations


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
        print(f"Current node: {current}, f: {f}, moves: {moves}, path: {path}")
        
        if current in visited:
            continue
        visited.add(current)

        if current in destinations:
            return current, len(visited), path
        
        for neighbor, _ in edges.get(current, []):
            if neighbor not in visited:
                #print(f"before add, open_set: {open_set}")
                for goal in destinations:
                    h_test = heuristic(neighbor, goal, nodes)
                    #print(f"neighbor: {neighbor}, goal: {goal}, heuristic2: {round(h_test**2)}, something: {_}")
                h_new = min(heuristic(neighbor, goal, nodes) for goal in destinations)
                f_new = (moves + 1) + h_new
                #print(f"f_new: {f_new}, moves: {moves + 1}, h_new: {h_new}")
                heapq.heappush(open_set, (f_new, neighbor, moves + 1, path + [neighbor]))
                #print(f"after add, open_set: {open_set}")
                #print(f"")
     
    
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
