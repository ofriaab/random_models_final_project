import os
import matplotlib.pyplot as plt



class Node:
    def __init__(self, data):
        self.data = data
        self.out_neighbors = []
        self.in_neighbors = []
        self.left = []  # New attribute for the left side of the bipartite graph
        self.right = []  # New attribute for the right side of the bipartite graph
        self.matched = False  # New attribute to keep track of matching status



def parse_graph_from_file(filename):
    nodes = {}
    total_nodes = 0
    with open(filename, 'r') as file:
        for line in file:
            dest, src = map(int, line.split())
            if src not in nodes:
                nodes[src] = Node(src)
                total_nodes += 1
            if dest not in nodes:
                nodes[dest] = Node(dest)
                total_nodes += 1
            nodes[src].out_neighbors.append(nodes[dest])
            nodes[dest].in_neighbors.append(nodes[src])
    # Initialize left and right attributes for each node
    for node in nodes.values():
        node.left = [node]  # Each node in the left side has itself
        node.right = []  # Right side initially empty
    return nodes, total_nodes


def build_bipartite_graph(nodes):
    for node in nodes.values():
        for out_neighbor in node.out_neighbors:
            # Add edges from left to right
            for left_node in node.left:
                for right_node in out_neighbor.left:
                    left_node.right.append(right_node)


def find_maximal_matching(nodes):
    matching = []
    for node in nodes.values():
        for right_node in node.right:
            if not right_node.matched:
                node.matched = right_node
                right_node.matched = node
                matching.append((node, right_node))
                break
    return matching


def calculate_width(nodes, total_nodes, matching_size):
    width = (total_nodes - matching_size)/total_nodes
    return width


# Example usage
def process_dag_file(filename):
    # Step 1: Parse graph from file
    nodes, total_nodes = parse_graph_from_file(filename)

    # Step 2: Build bipartite graph
    build_bipartite_graph(nodes)

    # Step 3: Find maximal matching
    matching = find_maximal_matching(nodes)

    # Step 4: Calculate width
    width = calculate_width(nodes, total_nodes, len(matching))

    return width

def plot_width_distribution(widths):
    width_counts = {width: widths.count(width) for width in set(widths)}
    sorted_widths = sorted(width_counts.keys())
    frequencies = [width_counts[width] for width in sorted_widths]

    plt.bar(sorted_widths, frequencies, color='skyblue', edgecolor='skyblue')
    plt.title('Width Distribution')
    plt.xlabel('Width')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def process_files_in_directory(directory):
    widths = []
    for filename in os.listdir(directory):
        if filename.endswith(".deps"):
            filepath = os.path.join(directory, filename)
            width = process_dag_file(filepath)
            widths.append(width)


    # Calculate statistics
    total_widths = sum(widths)
    num_files = len(widths)
    expected_value = total_widths / num_files
    variance = sum((width - expected_value) ** 2 for width in widths) / num_files
    max_width = max(widths)
    min_width = min(widths)

    # Plot width distribution
    plot_width_distribution(widths)

    return expected_value, variance, max_width, min_width

deps_directory = 'deps'
expected_value, variance, max_width, min_width = process_files_in_directory(deps_directory)
print("Expected value of width:", expected_value)
print("Variance of width:", variance)
print("Maximum width:", max_width)
print("Minimum width:", min_width)
