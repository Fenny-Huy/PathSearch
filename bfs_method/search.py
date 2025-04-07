import sys
from collections import deque

# Placeholder: replace this with file reading and parsing
def read_graph_from_file(filename):
    nodes = {
        1: (4, 1),
        2: (2, 2),
        3: (4, 4),
        4: (6, 3),
        5: (5, 6),
        6: (7, 5)
    }

    edges = {
        2: [(1, 4), (3, 4)],
        3: [(1, 5), (2, 5), (5, 6), (6, 7)],
        1: [(3, 5), (4, 6)],
        4: [(1, 6), (3, 5), (5, 7)],
        5: [(3, 6), (4, 8)],
        6: [(3, 7)]
    }

    origin = 2
    destinations = [5, 4]

    return nodes, edges, origin, destinations

def bfs(graph, origin, destinations):
    queue = deque()
    visited = set()
    parent = {}
    queue.append(origin)
    visited.add(origin)
    num_nodes = 1

    while queue:
        current = queue.popleft()
        if current in destinations:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = parent.get(current)
            path.reverse()
            return path[-1], num_nodes, path
        
        for neighbor, _ in sorted(graph.get(current, []), key=lambda x: x[0]):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
                num_nodes += 1

    return None, num_nodes, []

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return

    filename = sys.argv[1]
    method = sys.argv[2].lower()

    if method != "bfs":
        print(f"Search method '{method}' not supported in this implementation.")
        return

    nodes, edges, origin, destinations = read_graph_from_file(filename)
    goal, num_nodes, path = bfs(edges, origin, destinations)

    print(f"{filename} {method}")
    if goal is not None:
        print(f"{goal} {num_nodes}")
        print(" ".join(map(str, path)))
    else:
        print("No path found.")

if __name__ == "__main__":
    main()