import sys
import heapq
from collections import deque
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
                queue.append((neighbor, path + [neighbor], cost + edge_cost))
    
    return None, None

def dfs(edges, origin, destinations):
    stack = [(origin, [origin], 0)]
    visited = set()
    
    while stack:
        node, path, cost = stack.pop()
        if node in destinations:
            return path, cost
        
        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in edges.get(node, []):
                stack.append((neighbor, path + [neighbor], cost + edge_cost))
    
    return None, None

def heuristic(node, goal, nodes):
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
    else:
        print(f"Unknown method: {method}")
        return
    
    if path:
        print(f"{filename} {method}")
        print(f"{path[-1]} {len(path)}")
        print(" -> ".join(path))
    else:
        print(f"No path found using {method}")

if __name__ == "__main__":
    main()
