def dfs(edges, origin, destinations):
    
    # Initialize the stack with a tuple: (current_node, path, total_cost)
    stack = [(origin, [origin], 0)]
    visited = set()

    while stack:
        node, path, cost = stack.pop()

        if node in destinations:
            return path, cost

        if node not in visited:
            visited.add(node)
            # Get the list of neighbors (if any) and sort them in descending order (by integer value)
            # so that when popped, the smallest (ascending order) is expanded first.
            for neighbor, edge_cost in sorted(edges.get(node, []), key=lambda x: int(x[0]), reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], cost + edge_cost))

    return None, None
"""
    Guys this would just be the summary for anyone going through this code 

    Performs Depth-First Search (DFS) on the graph.

    Parameters:
    - edges: A dictionary where each key is a node ID (as a string) and the value is a list 
             of tuples (neighbor, cost). The neighbor is also a string.
    - origin: The starting node ID (as a string).
    - destinations: A set of goal node IDs (as strings).

    Returns:
    - A tuple (path, total_cost) if a destination is reached; otherwise, (None, None).

    Note:
    - When multiple neighbors are available, the neighbors are sorted in ascending numerical 
      order. (To achieve this with a stack, we sort in descending order before pushing so that 
      the smallest node is popped first.)
"""