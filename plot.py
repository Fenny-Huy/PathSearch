import matplotlib.pyplot as plt
import networkx as nx
import sys

def parse_input(file_path):
    """Parses the input file to extract nodes and directed edges."""
    nodes = {}
    edges = []
    origin = None
    destinations = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    section = None

    for line in lines:
        line = line.strip()
        if line == "Nodes:":
            section = "nodes"
        elif line == "Edges:":
            section = "edges"
        elif line == "Origin:":
            section = "origin"
        elif line == "Destinations:":
            section = "destinations"
        elif section == "nodes":
            node_id, coords = line.split(":")
            x, y = map(int, coords.strip(" ()").split(","))
            nodes[int(node_id)] = (x, y)
        elif section == "edges":
            edge_info, weight = line.split(":")
            node1, node2 = map(int, edge_info.strip(" ()").split(","))
            edges.append((node1, node2, int(weight)))  # Directed edge
        elif section == "origin":
            origin = int(line.strip())
        elif section == "destinations":
            destinations = list(map(int, line.split(";")))

    return nodes, edges, origin, destinations

def plot_graph(file_path):
    """Plots the directed graph based on the input file."""
    nodes, edges, origin, destinations = parse_input(file_path)

    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node, (x, y) in nodes.items():
        G.add_node(node, pos=(x, y))

    # Add directed edges with weights
    for node1, node2, weight in edges:
        G.add_edge(node1, node2, weight=weight)

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Draw graph with arrows
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=800, font_size=12, arrows=True, connectionstyle="arc3,rad=0.1", edgecolors="black")

    # Draw edge labels (weights) with improved visibility
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=10, label_pos=0.3, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )

    # Highlight origin and destination nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[origin], node_color="red", node_size=900, label="Origin")
    nx.draw_networkx_nodes(G, pos, nodelist=destinations, node_color="green", node_size=900, label="Destinations")

    plt.title("Directed Graph Visualization")
    plt.legend(["Nodes", "Edges", "Origin (Red)", "Destinations (Green)"])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    plot_graph(input_file)