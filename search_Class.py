import sys
import heapq
from collections import deque
import math

import networkx as nx
import matplotlib
matplotlib.use("TkAgg") # used in troubleshooting to ensure interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

# File Parsing and Graph Building
def parse_input_file(filename):
    nodes = {}
    edges = {}
    origin = None
    destinations = set()
    section = None

    with open(filename, 'r') as f:
        for line in f:
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
                if ':' in line:
                    parts = line.split(':', 1)
                    node_id = parts[0].strip()
                    coord_str = parts[1].strip().strip("() ")
                    x_str, y_str = coord_str.split(',')
                    nodes[node_id] = (float(x_str), float(y_str))
            elif section == "edges":
                if ':' in line:
                    parts = line.split(':', 1)
                    edge_str = parts[0].strip().strip("() ")
                    cost_str = parts[1].strip()
                    u_str, v_str = edge_str.split(',')
                    if u_str not in edges:
                        edges[u_str] = []
                    edges[u_str].append((v_str, float(cost_str)))
            elif section == "origin":
                origin = line.strip()
            elif section == "destinations":
                dests = line.replace(';', ' ').split()
                for d in dests:
                    destinations.add(d.strip())
    return nodes, edges, origin, list(destinations)

def build_graph(nodes, edges):
    G = nx.DiGraph()
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)
    for u, neighbors in edges.items():
        for v, cost in neighbors:
            G.add_edge(u, v, weight=cost)
    pos = nx.get_node_attributes(G, 'pos')
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    return G, pos, edge_labels

# Visualization Class
class PathFindingVisualizer:
    def __init__(self, nodes, edges, origin, destinations, method):
        # init the visualizer with the graph info and an empty list of states
        self.nodes = nodes
        self.edges = edges
        self.origin = origin
        self.destinations = destinations
        self.G, self.pos, self.edge_labels = build_graph(nodes, edges)
        self.states = []
        self.current_index = 0
        self.evaluated_edges = []
        self.method = method.upper()
        
        # set up figure and controls
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        plt.subplots_adjust(bottom=0.2)
        self.ax.set_title("Pathfinding Visualization")
        self.setup_controls()
        self.fig.canvas.manager.set_window_title(f"Pathfinding Algorithm Visualizer: {self.method}")
    
    def setup_controls(self):
        axprev = self.fig.add_axes([0.1, 0.05, 0.1, 0.075])
        axnext = self.fig.add_axes([0.8, 0.05, 0.1, 0.075])

        self.bprev = Button(axprev, 'Previous')
        self.bnext = Button(axnext, 'Next')

        self.bprev.on_clicked(self.prev_state)
        self.bnext.on_clicked(self.next_state)
    
    def add_state(self, visited, frontier, current_node=None, current_path=None, title=""):
        # states allow for traversal through the algoritm, which helps
        # troubleshooting and learning the pathfinding methods
        state = {
            "visited": list(visited),
            "frontier": list(frontier),
            "evaluated_edges": list(self.evaluated_edges),
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
        # clear and redraw the current state/step in the pathfinding alg
        # laying method calls is important for graph accuracy
        self.ax.clear()

        # draw base graph
        nx.draw(self.G, self.pos, with_labels=True, node_color='lightgray', node_size=800,
                edge_color='gray', font_size=12, ax=self.ax, arrows=True,
                connectionstyle="arc3,rad=0.1")
        
        # draw evaluated edges
        if state["evaluated_edges"]: 
            nx.draw_networkx_edges(self.G, self.pos, edgelist=state["evaluated_edges"],
                                   edge_color="green", width=2.5, ax=self.ax,
                                   connectionstyle="arc3,rad=0.1")
            
        # draw edge labels
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=state["edge_labels"],
                                     font_size=10, label_pos=0.4, rotate=False, ax=self.ax,
                                     bbox=dict(facecolor='white', edgecolor='none', alpha=0))
        
        # draw frontier and visited nodes
        if state["frontier"]:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=state["frontier"],
                                   node_color="yellow", node_size=800, ax=self.ax)
        if state["visited"]:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=state["visited"],
                                   node_color="green", node_size=800, ax=self.ax)
            
        # darw origin and destinations
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=[state["origin"]],
                               node_color="red", node_size=800, ax=self.ax)
        nx.draw_networkx_nodes(self.G, self.pos, nodelist=state["destinations"],
                               node_color="lime", node_size=800, ax=self.ax)
        
        # highlight current node
        if state["current_node"]:
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[state["current_node"]],
                                   node_color="orange", node_size=800, ax=self.ax)
            
        # check if final state
        if state.get("final", False) and state["current_path"] and len(state["current_path"]) > 1:
            candidate_edges = [(state["current_path"][i], state["current_path"][i+1])
                       for i in range(len(state["current_path"]) - 1)]
            
            # draw final state with visual adjustments for clarity
            nx.draw_networkx_edges(self.G, self.pos, edgelist=candidate_edges, edge_color="blue", 
                       width=4.0, ax=self.ax, connectionstyle="arc3,rad=0.1")
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=state["current_path"],
                       node_color="cyan", linewidths=2, edgecolors="blue", node_size=800, ax=self.ax)
            
            # highlight only the used destination
            used_destination = state["current_path"][-1]
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[used_destination],
                       node_color="lime", linewidths=2, edgecolors="blue", node_size=800, ax=self.ax)
            
            # below was done in the edge case that multiple origins existed, and
            # because i figured it was more likely to help than hurt in all/most cases
            # highlight the the used origin
            used_origin = state["current_path"][0]
            nx.draw_networkx_nodes(self.G, self.pos, nodelist=[used_origin],
                       node_color="red", linewidths=2, edgecolors="blue", node_size=800, ax=self.ax)
            
        else:
            # if not final state, draw the current path in orange
            if state["current_path"] and len(state["current_path"]) > 1:
                candidate_edges = [(state["current_path"][i], state["current_path"][i+1])
                                   for i in range(len(state["current_path"]) - 1)]
                nx.draw_networkx_edges(self.G, self.pos, edgelist=candidate_edges,
                                       edge_color="orange", width=3.0, ax=self.ax,
                                       connectionstyle="arc3,rad=0.1")

        # draw legend in top left corner
        legend_patches = [
            mpatches.Patch(color='red', label='Origin'),
            mpatches.Patch(color='lime', label='Destination'),
            mpatches.Patch(color='green', label='Visited / Evaluated'),
            mpatches.Patch(color='yellow', label='Frontier'),
            mpatches.Patch(color='orange', label='Current Node / Path'),
            mpatches.Patch(color='blue', label='Final Path used'),
            mpatches.Patch(color='cyan', label='Nodes in final Path'),
        ]
        self.ax.legend(handles=legend_patches, loc='upper left')

        self.ax.set_title(f"{self.method}: {state["title"]}")
        plt.draw()
    
    def show_state(self, index):
        # shows states after validating index
        if 0 <= index < len(self.states):
            self.current_index = index
            self.draw_state(self.states[index])
    
    def next_state(self, event):
        # button callback to move to the next state with basic validation
        if self.current_index < len(self.states) - 1:
            self.current_index += 1
            self.show_state(self.current_index)
    
    def prev_state(self, event):
        # button callback to move to the previous state with basic validation
        if self.current_index > 0:
            self.current_index -= 1
            self.show_state(self.current_index)
    
    def start(self):
        # run gui and show initial state of graph, with you guessed it, a little more basic validation
        if self.states:
            self.show_state(0)
        else:
            print("No states to visualize.")
            return

        plt.show()

# Search Algorithm Base and Implementations
class SearchAlgorithm:
    # basic init, does what it says on the tin
    def __init__(self, nodes, edges, origin, destinations):
        self.nodes = nodes
        self.edges = edges
        self.origin = origin
        self.destinations = destinations
        self.visited = set()
        self.frontier = None

    # below methods are abstract and must be implemented in subclasses
    # i have no idea how to implement this normally in python but the internet is magic
    # i really do learn into OOP practices a lot but moving from kotlin to python is a bit of a challenge
    # if this is not the right way to do this, please let me know and i will fix it - Dylan Morrison
    def initialize(self):
        raise NotImplementedError("Subclasses must implement initialize().")

    def expand_node(self):
        raise NotImplementedError("Subclasses must implement expand_node().")

    def add_to_frontier(self, neighbor, path, cost):
        raise NotImplementedError("Subclasses must implement add_to_frontier().")

    def extract_node(self, t):
        # For BFS/DFS/UCS the tuple is (cost, node, path), else (f_score, g_cost, node, path).
        return t[2] if len(t) == 4 else t[1]

    # through the magic of OOP, this method is used in the search methods to initialize the search
    # and then run the search until a destination is found or the frontier is empty
    # it was modelled to be abstract enough that scalability would be improved
    def search(self):
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

        # if no path to destination, return nothing
        # feels almost like a null return from kotlin :D
        return None, None

    def search_with_visualizer(self, visualizer):
        self.initialize()
        visualizer.add_state(self.visited, [self.extract_node(t) for t in self.frontier],
                             current_node=None, current_path=None, title="Initial State")                    
        
        while self.frontier:
            cost, node, path = self.expand_node()

            if node in self.visited:
                continue

            self.visited.add(node)
            visualizer.add_state(self.visited, [self.extract_node(t) for t in self.frontier],
                                 current_node=node, current_path=path, title=f"Expanded {node}")
            
            if node in self.destinations:
                # within the destination check, i kept the basic add_state as i felt the title reprenting 
                # the processing of the destination aided in showing the pathfinding algorithm in action,
                # and jumping to the final state in some algorithms was a bit jarring (especially in UCS)
                visualizer.add_state(self.visited, [self.extract_node(t) for t in self.frontier],
                                     current_node=node, current_path=path, title="Destination Reached")
                
                # manually creating the final state with all details for accuracy of the state fields
                final_state = {
                    "visited": list(self.visited),
                    "frontier": [self.extract_node(t) for t in self.frontier],
                    "evaluated_edges": list(visualizer.evaluated_edges),
                    "current_node": node,
                    "current_path": path,
                    "title": "Final Path Highlighted",
                    "origin": self.origin,
                    "destinations": list(self.destinations),
                    "G": visualizer.G,
                    "pos": visualizer.pos,
                    "edge_labels": visualizer.edge_labels,
                    "final": True   # mark as final for unique styling 
                }
                visualizer.states.append(final_state)
                return path, cost
            
            for neighbor, edge_cost in self.edges.get(node, []):
                if neighbor not in self.visited:
                    if (node, neighbor) not in visualizer.evaluated_edges:
                        visualizer.evaluated_edges.append((node, neighbor))

                    candidate_path = path + [neighbor]
                    visualizer.add_state(self.visited, [self.extract_node(t) for t in self.frontier],
                                          current_node=node, current_path=candidate_path,
                                          title=f"Evaluating {node} → {neighbor}")
                    self.add_to_frontier(neighbor, path, cost + edge_cost)

        return None, None

# child class to allow inheritance of the heuristic method where needed
class InformedSearchAlgorithm(SearchAlgorithm):
    def heuristic(self, node, goal):
        x1, y1 = self.nodes[node]
        x2, y2 = self.nodes[goal]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Uninformed Algorithms
class BFS(SearchAlgorithm): # expand nodes by level first

    def initialize(self):
        # init frontier as queue [cost, node, path]
        self.frontier = deque([(0, self.origin, [self.origin])])

    def expand_node(self):
        # Remove and return the first node from the frontier (FIFO).
        return self.frontier.popleft()
    
    def add_to_frontier(self, neighbor, path, cost):
        # Add a neighbor to the end of the frontier.
        self.frontier.append((cost, neighbor, path + [neighbor]))

class DFS(SearchAlgorithm): # expand nodes by depth first

    def initialize(self):
        # init frontier as stack [cost, node, path]
        self.frontier = [(0, self.origin, [self.origin])]

    def expand_node(self):
        # Remove and return the last node from the frontier (LIFO).
        return self.frontier.pop()
    
    def add_to_frontier(self, neighbor, path, cost):
        # Add a neighbor to the top of the frontier.
        self.frontier.append((cost, neighbor, path + [neighbor]))

class UCS(SearchAlgorithm): # expand nodes by lowest cost first

    def initialize(self):
        # init frontier as priority queue [cost, node, path]
        self.frontier = [(0, self.origin, [self.origin])]

    def expand_node(self):
        # Remove and return the node with the lowest cost from the frontier.
        return heapq.heappop(self.frontier)
    
    def add_to_frontier(self, neighbor, path, cost):
        # Add a neighbor to the frontier, maintaining the priority order by cost.
        heapq.heappush(self.frontier, (cost, neighbor, path + [neighbor]))

# Informed Algorithms
class AStar(InformedSearchAlgorithm): # prioritise by cost to current node (by edge weighting) and heuristic estimate to goal

    def initialize(self):
        # init frontier as priority queue [f_score, g_cost, node, path]
        # g_cost is the cost to reach the node
        # h_cost is the heuristic estimate to the goal
        # f_score = g_cost + h_cost
        self.goal = min(self.destinations, key=lambda d: self.heuristic(self.origin, d))
        self.frontier = [(0, 0, self.origin, [self.origin])]

    def expand_node(self):
        # Remove and return the node with the lowest f_score from the frontier.
        f_score, g_cost, node, path = heapq.heappop(self.frontier)
        return g_cost, node, path
    
    def add_to_frontier(self, neighbor, path, cost):
        # Add a neighbor to the frontier, maintaining the priority order by f_score.
        h_cost = self.heuristic(neighbor, self.goal)
        f_score = cost + h_cost
        heapq.heappush(self.frontier, (f_score, cost, neighbor, path + [neighbor]))

class GBFS(InformedSearchAlgorithm): # prioritise by heuristic estimate to goal only -
    # does not always find the optimal path, but can be faster in large graphs

    def initialize(self):
        # init frontier as priority queue [h_cost, node, path]
        # h_cost is the heuristic estimate to the goal
        self.goal = min(self.destinations, key=lambda d: self.heuristic(self.origin, d))
        self.frontier = [(self.heuristic(self.origin, self.goal), self.origin, [self.origin])]

    def expand_node(self):
        # Remove and return the node with the lowest h_cost from the frontier.
        h_cost, node, path = heapq.heappop(self.frontier)
        return 0, node, path
    
    def add_to_frontier(self, neighbor, path, cost):
        # Add a neighbor to the frontier, maintaining the priority order by h_cost.
        h_cost = self.heuristic(neighbor, self.goal)
        heapq.heappush(self.frontier, (h_cost, neighbor, path + [neighbor]))

class HSM(InformedSearchAlgorithm): # prioritise by heuristic estimate to goal and cost to current node (by move count)

    def initialize(self):
        # init frontier as priority queue [f_score, g_cost, node, path]
        # g_cost is the number of moves to reach the node
        # h_cost is the heuristic estimate to the goal
        # f_score = g_cost + h_cost
        self.goal = min(self.destinations, key=lambda d: self.heuristic(self.origin, d))
        self.frontier = [(0, 0, self.origin, [self.origin])]

    def expand_node(self):
        # Remove and return the node with the lowest f_score from the frontier.
        f_score, moves, node, path = heapq.heappop(self.frontier)
        return moves, node, path
    
    def add_to_frontier(self, neighbor, path, moves):
        # Add a neighbor to the frontier, maintaining the priority order by f_score.
        h_cost = self.heuristic(neighbor, self.goal)
        f_score = (moves + 1) + h_cost
        heapq.heappush(self.frontier, (f_score, moves + 1, neighbor, path + [neighbor]))

# Main Function
def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python search.py <filename> <method> [visualize (optional boolean)]")
        print("Methods: bfs, dfs, ucs, a*, gbfs, hsm")
        print("Example: python search.py input.txt gbfs true")
        return

    filename = sys.argv[1]
    method = sys.argv[2].lower()
    visualize = len(sys.argv) == 4 and sys.argv[3].lower() in {"true", "yes", "1"}
    nodes, edges, origin, destinations = parse_input_file(filename)

    # pair classes and methods
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
        # create visualizer instance and pass it to the search method, then start gui
        visualizer = PathFindingVisualizer(nodes, edges, origin, destinations, method)
        path, cost = algorithm.search_with_visualizer(visualizer)
    else: # run method without visualizations
        path, cost = algorithm.search()

    if path:
        print(f"Filename: {filename} Method: {method.upper()}")
        print(f"Destination: {path[-1]}, Path Length: {len(path)}")
        print(" -> ".join(path))

        # delayed start of visualizer to allow for output to be compared with gui
        # comes with the benefit of confirming that the pathfinding method worked, again
        if visualize:
            visualizer.start()
    else:
        print(f"No path found using {method}")

if __name__ == "__main__":
    main()
