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

def select_neighbor(nodes, u):
    potential_neighbors = [node for node in nodes if node['position'] > u['position']]
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

def generate_custom_model(n):
    nodes = initialize_nodes(n)
    graph = nx.DiGraph()  # Directed graph
    for node in nodes:
        graph.add_node(node['id'])
    edge_count = 0
    max_edges = 3 * n  # Targeting average degree of 3
    while edge_count < max_edges:
        u = select_node(nodes)
        v = select_neighbor(nodes, u)
        if v and not graph.has_edge(u['id'], v['id']):  # Ensure no duplicate edges
            add_edge(graph, u, v)
            edge_count += 1
    return graph, nodes

def generate_gnp_model(n, p):
    graph = nx.gnp_random_graph(n, p, directed=True)
    nodes = [{'id': i, 'runtime': random.randint(1, 100), 'predecessors': list(graph.predecessors(i))} for i in graph.nodes()]
    return graph, nodes

def generate_layer_by_layer_model(n, layers):
    nodes_per_layer = n // layers
    nodes = []
    graph = nx.DiGraph()
    for layer in range(layers):
        for i in range(nodes_per_layer):
            node_id = layer * nodes_per_layer + i
            node = {'id': node_id, 'runtime': random.randint(1, 100), 'predecessors': []}
            nodes.append(node)
            graph.add_node(node_id)
            if layer > 0:
                # Connect to nodes in the previous layer
                for j in range(nodes_per_layer):
                    prev_node_id = (layer - 1) * nodes_per_layer + j
                    if random.random() < 0.5:
                        graph.add_edge(prev_node_id, node_id)
                        node['predecessors'].append(prev_node_id)
    return graph, nodes

def generate_graph(model_type, n, **kwargs):
    if model_type == 'custom':
        return generate_custom_model(n)
    elif model_type == 'gnp':
        p = kwargs.get('p', 0.1)
        return generate_gnp_model(n, p)
    elif model_type == 'layered':
        layers = kwargs.get('layers', 5)
        return generate_layer_by_layer_model(n, layers)
    else:
        raise ValueError("Unsupported model type")

def draw_graph(graph, nodes):
    pos = nx.spring_layout(graph)  # Layout for visualization
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, labels={node['id']: node['id'] for node in nodes},
            node_size=500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray")
    plt.title("Generated Graph")
    plt.show()

def main():
    n = 100  # Number of nodes
    model_type = 'custom'  # Change to 'gnp' or 'layered' to generate different models
    if model_type == 'custom':
        graph, nodes = generate_graph(model_type, n)
    elif model_type == 'gnp':
        p = 0.1  # Probability for G(n,p) model
        graph, nodes = generate_graph(model_type, n, p=p)
    elif model_type == 'layered':
        layers = 5  # Number of layers for layer-by-layer model
        graph, nodes = generate_graph(model_type, n, layers=layers)

    draw_graph(graph, nodes)

    # Display the graph representation:
    graph_representation = [(node['id'], node['runtime'], node['predecessors']) for node in nodes]
    for node in graph_representation:
        print(node)

if __name__ == "__main__":
    main()
