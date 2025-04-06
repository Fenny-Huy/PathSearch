import sys

def parse_file(filename):
    
    graph = {}
    origin = None
    destinations = set()
    section = None

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
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
            # Expected format: "1: (4,1)"
            parts = line.split(":")
            node_id = int(parts[0].strip())
            # Coordinates are not used for DFS in this example.
            # Initialize graph for this node.
            if node_id not in graph:
                graph[node_id] = []
        elif section == "edges":
            # Expected format: "(2,1): 4"
            parts = line.split(":")
            edge_part = parts[0].strip()  # e.g., "(2,1)"
            cost = int(parts[1].strip())  # cost is available but not used in DFS
            # Remove parentheses and split the two nodes
            edge_part = edge_part.strip("()")
            from_node, to_node = map(int, edge_part.split(","))
            # For DFS, we only care about connectivity.
            if from_node in graph:
                graph[from_node].append(to_node)
            else:
                graph[from_node] = [to_node]
            # (If the graph is not fully defined in the Nodes section, ensure to add the target node too)
            if to_node not in graph:
                graph[to_node] = []
        elif section == "origin":
            origin = int(line.strip())
        elif section == "destinations":
            # Expected format: "5; 4" or similar
            parts = line.split(";")
            for part in parts:
                destinations.add(int(part.strip()))

    return graph, origin, destinations

def dfs(graph, current, destinations, path, visited, nodes_created):
    """
    Recursively performs depth-first search.
    
    Parameters:
    - graph: the dictionary representing the graph.
    - current: the current node.
    - destinations: a set of goal nodes.
    - path: the current path taken (list of nodes).
    - visited: a set of visited nodes to avoid cycles.
    - nodes_created: a list with one element used as a mutable counter for expanded nodes.
    
    Returns:
    - A list representing the path to a destination if found; otherwise, None.
    """
    nodes_created[0] += 1  # Count each call as a node expansion
    if current in destinations:
        return path

    # Expand neighbors in ascending order to satisfy tie-breaking conditions.
    for neighbor in sorted(graph[current]):
        if neighbor not in visited:
            visited.add(neighbor)
            result = dfs(graph, neighbor, destinations, path + [neighbor], visited, nodes_created)
            if result is not None:
                return result  # Found a path to a destination
            # Backtrack: remove neighbor from visited for alternative paths.
            visited.remove(neighbor)
    return None

def main():
    # Ensure proper usage:
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)
    
    filename = sys.argv[1]
    method = sys.argv[2].upper()

    # Only DFS is handled in this code snippet.
    if method != "DFS":
        print("This code example only implements DFS.")
        sys.exit(1)

    # Parse the file to build the graph, get the origin, and destinations.
    graph, origin, destinations = parse_file(filename)

    # Initialize visited set with the origin
    visited = set([origin])
    nodes_created = [0]  # Using a list as a mutable counter

    # Run DFS
    path = dfs(graph, origin, destinations, [origin], visited, nodes_created)

    if path is not None:
        goal = path[-1]
        # Output format: 
        # filename method
        # goal number_of_nodes
        # path (as a sequence of node IDs)
        print(f"{filename} {method}")
        print(f"{goal} {nodes_created[0]}")
        print(" ".join(map(str, path)))
    else:
        print("No path found.")

if __name__ == '__main__':
    main()
