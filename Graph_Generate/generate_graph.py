import networkx as nx
import matplotlib.pyplot as plt
import random

def initialize_nodes(n, runtime_range=(1, 100)):
    nodes = [{'id': i, 'runtime': random.randint(*runtime_range), 'degree': 0, 'weight': 1, 'position': i, 'predecessors': []} for i in range(n)]
    return nodes

def select_node(nodes):
    total_weight = sum(node['weight'] for node in nodes)
    rnd = random.uniform(0, total_weight)
    for node in nodes:
        rnd -= node['weight']
        if rnd <= 0:
            return node

def select_neighbor(nodes, u, r):
    potential_neighbors = [node for node in nodes if node['position'] > u['position'] and node['position'] < u['position']+r]
    total_weight = sum(node['weight'] for node in potential_neighbors)
    rnd = random.uniform(0, total_weight)
    for node in potential_neighbors:
        rnd -= node['weight']
        if rnd <= 0:
            return node

def add_edge(graph, u, v):
    graph.add_edge(u['id'], v['id'])
    u['degree'] += 1
    v['degree'] += 1
    u['weight'] = u['degree'] + 1
    v['weight'] = v['degree'] + 1
    v['predecessors'].append(u['id'])

def generate_dag(n, r):
    nodes = initialize_nodes(n)
    graph = nx.DiGraph()  # Directed graph
    for node in nodes:
        graph.add_node(node['id'])
    edge_count = 0
    max_edges = 3 * n  # Targeting average degree of 3
    while edge_count < max_edges:
        u = select_node(nodes)
        v = select_neighbor(nodes, u, r)
        if v and not graph.has_edge(u['id'], v['id']):  # Ensure no duplicate edges
            add_edge(graph, u, v)
            edge_count += 1
    return graph, nodes

def draw_graph(graph, nodes):
    pos = nx.spring_layout(graph)  # Layout for visualization
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, labels={node['id']: node['id'] for node in nodes},
            node_size=500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray")
    plt.title("Generated Directed Acyclic Graph")
    plt.show()

# Example usage:
n = 20
r = 5
generated_graph, nodes = generate_dag(n, r)
draw_graph(generated_graph, nodes)

# Display the graph representation:
graph_representation = [(node['id'], node['runtime'], node['predecessors']) for node in nodes]
for node in graph_representation:
    print(node)