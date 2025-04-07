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

class SearchAlgorithm:
    def __init__(self, nodes, edges, origin, destinations):
        self.nodes = nodes
        self.edges = edges
        self.origin = origin
        self.destinations = destinations
        self.visited = set()
        self.frontier = None

    def initialize(self):
        """Initialize the frontier and other algorithm-specific data structures."""
        raise NotImplementedError("Subclasses must implement this method.")

    def expand_node(self):
        """Expand a node from the frontier."""
        raise NotImplementedError("Subclasses must implement this method.")

    def add_to_frontier(self, neighbor, path, cost):
        """Add a neighbor to the frontier."""
        raise NotImplementedError("Subclasses must implement this method.")

    def search(self):
        """Perform the search."""
        self.initialize()
        while self.frontier:
            cost, node, path = self.expand_node()

            if node in self.visited:
                continue

            self.visited.add(node)

            if node in self.destinations:
                return path, cost

            for neighbor, edge_cost in self.edges.get(node, []):
                if neighbor not in self.visited:
                    self.add_to_frontier(neighbor, path, cost + edge_cost)

        return None, None
    
    def search_with_visualizer(self, visualizer):

        self.initialize()

        
        # Record the initial state.
        visualizer.add_state(
            visited=self.visited,
            frontier=[n for _, n, _ in self.frontier],
            evaluated_edges=visualizer.evaluated_edges,
            current_node=None,
            current_path=None,
            title="Initial State"
        )

        while self.frontier:
            cost, node, path = self.expand_node()
            if node in self.visited:
                continue
            self.visited.add(node)

            # Record the state after expanding a node.
            visualizer.add_state(
                visited=self.visited,
                frontier=[n for _, n, _ in self.frontier],
                evaluated_edges=visualizer.evaluated_edges,
                current_node=node,
                current_path=path,
                title=f"Expanded {node}"
            )

            if node in self.destinations:
                visualizer.add_state(
                    visited=self.visited,
                    frontier=[n for _, n, _ in self.frontier],
                    evaluated_edges=visualizer.evaluated_edges,
                    current_node=node,
                    current_path=path,
                    title="Destination Reached"
                )
                return path, cost
            
            for neighbor, edge_cost in self.edges.get(node, []):
                if neighbor not in self.visited:
                    # Add the evaluated edge (if not already recorded).
                    if (node, neighbor) not in visualizer.evaluated_edges:
                        visualizer.evaluated_edges.append((node, neighbor))
                    candidate_path = path + [neighbor]
                    visualizer.add_state(
                        visited=self.visited,
                        frontier=[n for _, n, _ in self.frontier],
                        evaluated_edges=visualizer.evaluated_edges,
                        current_node=node,
                        current_path=candidate_path,
                        title=f"Evaluating {node} → {neighbor}"
                    )
                    self.add_to_frontier(neighbor, path, cost + edge_cost)

        return None, None


# Uninformed search algorithms

class BFS(SearchAlgorithm):
    def initialize(self):
        self.frontier = deque([(0, self.origin, [self.origin])])
    def expand_node(self):
        return self.frontier.popleft()
    def add_to_frontier(self, neighbor, path, cost):
        self.frontier.append((cost, neighbor, path + [neighbor]))

class DFS(SearchAlgorithm):
    def initialize(self):
        self.frontier = [(0, self.origin, [self.origin])]
    def expand_node(self):
        return self.frontier.pop()
    def add_to_frontier(self, neighbor, path, cost):
        self.frontier.append((cost, neighbor, path + [neighbor]))

class UCS(SearchAlgorithm):
    def initialize(self):
        self.frontier = [(0, self.origin, [self.origin])]
    def expand_node(self):
        return heapq.heappop(self.frontier)
    def add_to_frontier(self, neighbor, path, cost):
        heapq.heappush(self.frontier, (cost, neighbor, path + [neighbor]))

# Informed search algorithms

class AStar(SearchAlgorithm):
    def initialize(self):
        self.goal = min(self.destinations, key=lambda d: self.heuristic(self.origin, d))
        self.frontier = [(0, 0, self.origin, [self.origin])]  # (f_score, g_cost, node, path)
    def expand_node(self):
        f_score, g_cost, node, path = heapq.heappop(self.frontier)
        return g_cost, node, path
    def add_to_frontier(self, neighbor, path, cost):
        h_cost = self.heuristic(neighbor, self.goal)
        f_score = cost + h_cost
        heapq.heappush(self.frontier, (f_score, cost, neighbor, path + [neighbor]))
    def heuristic(self, node, goal):
        x1, y1 = self.nodes[node]
        x2, y2 = self.nodes[goal]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class GBFS(SearchAlgorithm):
    def initialize(self):
        self.goal = min(self.destinations, key=lambda d: self.heuristic(self.origin, d))
        self.frontier = [(self.heuristic(self.origin, self.goal), self.origin, [self.origin])]
    def expand_node(self):
        h_cost, node, path = heapq.heappop(self.frontier)
        return 0, node, path
    def add_to_frontier(self, neighbor, path, cost):
        h_cost = self.heuristic(neighbor, self.goal)
        heapq.heappush(self.frontier, (h_cost, neighbor, path + [neighbor]))
    def heuristic(self, node, goal):
        x1, y1 = self.nodes[node]
        x2, y2 = self.nodes[goal]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class HSM(SearchAlgorithm):
    def initialize(self):
        self.goal = min(self.destinations, key=lambda d: self.heuristic(self.origin, d))
        self.frontier = [(0, 0, self.origin, [self.origin])]
    def expand_node(self):
        f_score, moves, node, path = heapq.heappop(self.frontier)
        return moves, node, path
    def add_to_frontier(self, neighbor, path, moves):
        h_cost = self.heuristic(neighbor, self.goal)
        f_score = (moves + 1) + h_cost
        heapq.heappush(self.frontier, (f_score, moves + 1, neighbor, path + [neighbor]))
    def heuristic(self, node, goal):
        x1, y1 = self.nodes[node]
        x2, y2 = self.nodes[goal]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

"""
# Visualized version that wraps any SearchAlgorithm subclass.
class VisualizedSearchAlgorithm:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        # Set up the base graph visualization.
        self.G, self.pos, self.edge_labels, self.fig, self.ax = setup_visualization(
            algorithm.nodes, algorithm.edges
        )
        # Initialize a list to track evaluated (permanently colored green) edges.
        self.evaluated_edges = []
        # Provide the visualizer to the wrapped algorithm for access.
        self.algorithm.visualize = self

    def visualize_step(self, title, current_node=None, current_path=None):
        # Extract node IDs from the frontier.
        frontier_nodes = [str(t[2]) if len(t) == 4 else str(t[1]) for t in self.algorithm.frontier]
        visualize_graph(
            self.G,
            self.pos,
            self.edge_labels,
            [str(node) for node in self.algorithm.visited],
            frontier_nodes,
            self.evaluated_edges,  # permanently evaluated edges (green)
            str(self.algorithm.origin),
            [str(node) for node in self.algorithm.destinations],
            title,
            self.ax,
            current_node=current_node,
            current_path=current_path
        )

    def search(self, visualizer):
        # Record the initial state.
        visualizer.add_state(
            visited=self.visited,
            frontier=[n for _, n, _ in self.frontier],
            evaluated_edges=visualizer.evaluated_edges,
            current_node=None,
            current_path=None,
            title="Initial State"
        )
        self.initialize()
        while self.frontier:
            cost, node, path = self.expand_node()
            if node in self.visited:
                continue
            self.visited.add(node)
            # Record the state after expanding a node.
            visualizer.add_state(
                visited=self.visited,
                frontier=[n for _, n, _ in self.frontier],
                evaluated_edges=visualizer.evaluated_edges,
                current_node=node,
                current_path=path,
                title=f"Expanded {node}"
            )
            if node in self.destinations:
                visualizer.add_state(
                    visited=self.visited,
                    frontier=[n for _, n, _ in self.frontier],
                    evaluated_edges=visualizer.evaluated_edges,
                    current_node=node,
                    current_path=path,
                    title="Destination Reached"
                )
                return path, cost
            for neighbor, edge_cost in self.edges.get(node, []):
                if neighbor not in self.visited:
                    # Add the evaluated edge (if not already recorded).
                    if (node, neighbor) not in visualizer.evaluated_edges:
                        visualizer.evaluated_edges.append((node, neighbor))
                    candidate_path = path + [neighbor]
                    visualizer.add_state(
                        visited=self.visited,
                        frontier=[n for _, n, _ in self.frontier],
                        evaluated_edges=visualizer.evaluated_edges,
                        current_node=node,
                        current_path=candidate_path,
                        title=f"Evaluating {node} → {neighbor}"
                    )
                    self.add_to_frontier(neighbor, path, cost + edge_cost)
        return None, None

def setup_visualization(nodes, edges):
    G = nx.DiGraph()
    for node, (x, y) in nodes.items():
        G.add_node(node, pos=(x, y))
    for u, v, cost in [(u, v, cost) for u in edges for v, cost in edges[u]]:
        if u not in nodes or v not in nodes:
            print(f"Warning: Node '{u}' or '{v}' is missing from the nodes dictionary.")
            continue
        G.add_edge(u, v, weight=cost)
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 7))
    return G, pos, edge_labels, fig, ax

def visualize_graph(G, pos, edge_labels, visited, frontier, evaluated_edges, origin, destinations, title, ax, current_node=None, current_path=None):
    ax.clear()
    # Convert IDs to strings.
    visited = [str(node) for node in visited]
    frontier = [str(node) for node in frontier]
    origin = str(origin)
    destinations = [str(node) for node in destinations]
    
    # Draw the base graph.
    nx.draw(G, pos, with_labels=True, node_color='lightgray', node_size=800,
            edge_color='gray', font_size=12, ax=ax, arrows=True,
            connectionstyle="arc3,rad=0.1")
    
    # Draw evaluated edges (permanent, in green).
    if evaluated_edges:
        nx.draw_networkx_edges(G, pos, edgelist=evaluated_edges, edge_color="green",
                               width=2.5, ax=ax, connectionstyle="arc3,rad=0.1")
    
    # Draw edge labels.
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,
                                 label_pos=0.4, rotate=False,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                                 ax=ax)
    
    # Highlight nodes: frontier in yellow, visited in green, origin in red, destinations in lime.
    nx.draw_networkx_nodes(G, pos, nodelist=frontier, node_color="yellow", node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=visited, node_color="green", node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[origin], node_color="red", node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=destinations, node_color="lime", node_size=800, ax=ax)
    
    # Highlight the currently processed node (in orange) if provided.
    if current_node is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[str(current_node)], node_color="orange", node_size=800, ax=ax)
    
    # Highlight the candidate path (the entire path being evaluated) in orange.
    if current_path is not None and len(current_path) > 1:
        candidate_edges = [(current_path[i], current_path[i+1]) for i in range(len(current_path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=candidate_edges, edge_color="orange",
                               width=3.0, ax=ax, connectionstyle="arc3,rad=0.1")
    
    plt.title(title)
    plt.pause(0.5)

def visualize_final_path(G, pos, edge_labels, path, ax, method):
    ax.clear()
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    edge_colors = ['blue' if (u, v) in path_edges else 'gray' for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color=edge_colors,
            edgecolors="black", node_size=800, font_size=12, ax=ax, arrows=True,
            connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,
                                 label_pos=0.4, rotate=False,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                                 ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="cyan", node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_color="red", node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_color="lime", node_size=800, ax=ax)
    plt.title(f"{method} - Final Path")
    plt.pause(2)
"""
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

class PathFindingVisualizer:
    def __init__(self, nodes, edges, origin, destinations):
        """
        Initialize the visualizer with the graph information.
        """
        self.nodes = nodes
        self.edges = edges
        self.origin = origin
        self.destinations = destinations
        # Build the graph
        self.G, self.pos, self.edge_labels = self.setup_graph()
        # List of state dictionaries to store snapshots of the visualization.
        self.states = []
        self.current_index = 0
        # Initialize evaluated_edges so that it can be used in search_with_visualizer.
        self.evaluated_edges = []
        
        # Set up figure, axes and controls.
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        plt.subplots_adjust(bottom=0.2)
        self.setup_controls()
    
    def setup_graph(self):
        """
        Create a NetworkX graph and extract node positions and edge labels.
        """
        G = nx.DiGraph()
        for node, (x, y) in self.nodes.items():
            G.add_node(node, pos=(x, y))
        for u, neighbors in self.edges.items():
            for v, cost in neighbors:
                if u in self.nodes and v in self.nodes:
                    G.add_edge(u, v, weight=cost)
        pos = nx.get_node_attributes(G, 'pos')
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        return G, pos, edge_labels

    def add_state(self, visited, frontier, evaluated_edges, current_node=None, current_path=None, title=""):
        """
        Record a state of the visualization. All parameters are lists or single values:
         - visited: list or set of visited node IDs.
         - frontier: list or set of node IDs currently in the frontier.
         - evaluated_edges: list of (u, v) tuples for edges that have been permanently marked.
         - current_node: the node currently being expanded (optional).
         - current_path: the candidate path (list of nodes) being evaluated (optional).
         - title: text to display as the title.
        """
        state = {
            "visited": list(visited),
            "frontier": list(frontier),
            "evaluated_edges": list(evaluated_edges),
            "current_node": current_node,
            "current_path": current_path[:] if current_path is not None else None,
            "title": title,
            "origin": self.origin,
            "destinations": list(self.destinations),
            "G": self.G,
            "pos": self.pos,
            "edge_labels": self.edge_labels
        }
        self.states.append(state)
    
    def draw_state(self, state):
        """
        Draw a single state. This function handles drawing the base graph,
        the evaluated edges (in green), the candidate/current path (in orange), 
        and all the node groups with custom colors. It also adds a legend.
        """
        self.ax.clear()
        # Draw base graph.
        nx.draw(self.G, self.pos, with_labels=True, node_color='lightgray', node_size=800,
                edge_color='gray', font_size=12, ax=self.ax, arrows=True,
                connectionstyle="arc3,rad=0.1")
        
        # Draw evaluated edges (permanent, in green).
        if state["evaluated_edges"]:
            nx.draw_networkx_edges(self.G, self.pos, edgelist=state["evaluated_edges"],
                                   edge_color="green", width=2.5, ax=self.ax,
                                   connectionstyle="arc3,rad=0.1")
        
        # Draw edge labels.
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=state["edge_labels"],
                                     font_size=10, label_pos=0.4, rotate=False,
                                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                                     ax=self.ax)
        
        # Draw nodes: frontier in yellow, visited in green.
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[str(n) for n in state["frontier"]],
                               node_color="yellow", node_size=800, ax=self.ax)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[str(n) for n in state["visited"]],
                               node_color="green", node_size=800, ax=self.ax)
        # Draw origin (red) and destinations (lime).
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[state["origin"]],
                               node_color="red", node_size=800, ax=self.ax)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[str(n) for n in state["destinations"]],
                               node_color="lime", node_size=800, ax=self.ax)
        
        # Highlight the current node (if provided) in orange.
        if state["current_node"] is not None:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[str(state["current_node"])],
                                   node_color="orange", node_size=800, ax=self.ax)
        
        # Highlight the candidate/current path if provided.
        if state["current_path"] is not None and len(state["current_path"]) > 1:
            candidate_edges = [(state["current_path"][i], state["current_path"][i+1])
                               for i in range(len(state["current_path"]) - 1)]
            nx.draw_networkx_edges(self.G, self.pos, edgelist=candidate_edges,
                                   edge_color="orange", width=3.0, ax=self.ax,
                                   connectionstyle="arc3,rad=0.1")
        
        self.ax.set_title(state["title"])
        
        # Add a legend.
        legend_patches = [
            mpatches.Patch(color='red', label='Origin'),
            mpatches.Patch(color='lime', label='Destination'),
            mpatches.Patch(color='green', label='Visited / Evaluated'),
            mpatches.Patch(color='yellow', label='Frontier'),
            mpatches.Patch(color='orange', label='Current Node / Path')
        ]
        self.ax.legend(handles=legend_patches, loc='upper left')
        self.fig.canvas.draw_idle()
    
    def show_state(self, index):
        """
        Display the state at the given index.
        """
        if index < 0 or index >= len(self.states):
            return
        self.current_index = index
        self.draw_state(self.states[index])
    
    def next_state(self, event):
        """
        Callback for the "Next" button.
        """
        if self.current_index < len(self.states) - 1:
            self.current_index += 1
            self.draw_state(self.states[self.current_index])
    
    def prev_state(self, event):
        """
        Callback for the "Previous" button.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.draw_state(self.states[self.current_index])
    
    def setup_controls(self):
        """
        Create Next and Previous buttons on the figure.
        """
        axprev = plt.axes([0.3, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
        bprev = Button(axprev, 'Previous')
        bnext = Button(axnext, 'Next')
        bprev.on_clicked(self.prev_state)
        bnext.on_clicked(self.next_state)
    
    def start(self):
        """
        If states have been added, show the first one and then start the event loop.
        """
        if self.states:
            self.show_state(0)
        plt.show()

methods = {
    'bfs': {'class': BFS, 'informed': False},
    'dfs': {'class': DFS, 'informed': False},
    'ucs': {'class': UCS, 'informed': False},
    'a*': {'class': AStar, 'informed': True},
    'gbfs': {'class': GBFS, 'informed': True},
    'hsm': {'class': HSM, 'informed': True},
}

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python search.py <filename> <method> [visualize (optional boolean)]")
        print("Methods: bfs, dfs, ucs, a*, gbfs, hsm")
        print("Example: python search.py input.txt gbfs true")
        return

    filename = sys.argv[1]
    method = sys.argv[2].lower()
    visualize = len(sys.argv) == 4 and sys.argv[3].lower() in {"true", "yes", "1"}

    # Parse the input file.
    nodes, edges, origin, destinations = parse_input_file(filename)

    # Map methods to classes.
    algorithms = {
        "bfs": BFS,
        "dfs": DFS,
        "ucs": UCS,
        "a*": AStar,
        "gbfs": GBFS,
        "hsm": HSM,
    }

    if method not in algorithms:
        print(f"Unknown method: {method}")
        return

    algorithm_class = algorithms[method]
    algorithm = algorithm_class(nodes, edges, origin, destinations)

    if visualize:
        # Instead of wrapping with VisualizedSearchAlgorithm,
        # create a PathFindingVisualizer and pass it to a modified search method.
        visualizer = PathFindingVisualizer(nodes, edges, origin, destinations)
        # Use the modified search loop that records states.
        path, cost = algorithm.search_with_visualizer(visualizer)
        # Launch the interactive viewer.
        visualizer.start()
    else:
        path, cost = algorithm.search()

    if path:
        print(f"Filename: {filename} Method: {method.upper()}")
        print(f"Destination: {path[-1]}, Path Length: {len(path)}")
        print(f"Path: {' -> '.join(path)}")
    else:
        print(f"No path found using {method}")

if __name__ == "__main__":
    main()

