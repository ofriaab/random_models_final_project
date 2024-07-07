import networkx as nx
import matplotlib.pyplot as plt
import random

class GraphModel:
    def __init__(self, n, runtime_range=(1, 100)):
        self.n = n
        self.runtime_range = runtime_range
        self.nodes = [{'id': i, 'runtime': random.randint(*self.runtime_range), 'predecessors': [], 'weight': 1, 'out_degree':0, 'position':i } for i in range(n)]
        self.graph = nx.DiGraph()  # Directed graph

    def generate_graph(self):
        raise NotImplementedError("Subclasses should implement this!")

    def draw_graph(self):
        pos = nx.spring_layout(self.graph)  # Layout for visualization
        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, pos, with_labels=True, labels={node['id']: node['id'] for node in self.nodes},
                node_size=500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray")
        plt.title(f"Graph of Type: {self.__class__.__name__}")
        plt.show()

    def get_graph_representation(self):
        return [(node['id'], node['runtime'], node['predecessors']) for node in self.nodes]

class CustomModel(GraphModel):
    def __init__(self, n, runtime_range=(1, 100), R=5, edges_per_node=3):
        super().__init__(n, runtime_range)
        self.R = R
        self.edges_per_node = edges_per_node
        self.generate_graph()

    def select_node(self):
        total_weight = sum(node['weight'] for node in self.nodes)
        rnd = random.uniform(0, total_weight)
        for node in self.nodes:
            rnd -= node['weight']
            if rnd <= 0:
                return node

    def select_neighbor(self, u):
        potential_neighbors = [node for node in self.nodes if node['id'] > u['id'] and node['position'] < u['position']+self.R]
        total_weight = sum(node['weight'] for node in potential_neighbors)
        rnd = random.uniform(0, total_weight)
        for node in potential_neighbors:
            rnd -= node['weight']
            if rnd <= 0:
                return node

    def add_edge(self, u, v):
        self.graph.add_edge(u['id'], v['id'])
        u['out_degree'] += 1
        u['weight'] = u['weight'] + 1
        v['weight'] = v['weight'] + 1
        v['predecessors'].append(u['id'])

    def generate_graph(self):
        for node in self.nodes:
            self.graph.add_node(node['id'])
        edge_count = 0
        max_edges = self.edges_per_node * self.n  # Targeting average degree of 3
        while edge_count < max_edges:
            u = self.select_node()
            v = self.select_neighbor(u)
            if v and not self.graph.has_edge(u['id'], v['id']):  # Ensure no duplicate edges
                self.add_edge(u, v)
                edge_count += 1

class GnpModel(GraphModel):
    def __init__(self, n, p, runtime_range=(1, 100)):
        super().__init__(n, runtime_range)
        self.p = p
        self.generate_graph()

    def generate_graph(self):
        self.graph = nx.gnp_random_graph(self.n, self.p, directed=True)
        for node in self.graph.nodes():
            self.nodes[node]['predecessors'] = list(self.graph.predecessors(node))

class LayeredModel(GraphModel):
    def __init__(self, n, layers, runtime_range=(1, 100)):
        super().__init__(n, runtime_range)
        self.layers = layers
        self.generate_graph()

    def generate_graph(self):
        nodes_per_layer = self.n // self.layers
        for layer in range(self.layers):
            for i in range(nodes_per_layer):
                node_id = layer * nodes_per_layer + i
                self.nodes[node_id]['predecessors'] = []
                self.graph.add_node(node_id)
                if layer > 0:
                    # Connect to nodes in the previous layer
                    for j in range(nodes_per_layer):
                        prev_node_id = (layer - 1) * nodes_per_layer + j
                        if random.random() < 0.5:
                            self.graph.add_edge(prev_node_id, node_id)
                            self.nodes[node_id]['predecessors'].append(prev_node_id)

# Example usage:

def main():
    n = 100  # Number of nodes
    p = 0.1  # Probability for G(n,p) model
    layers = 5  # Number of layers for layer-by-layer model

    # Custom Model
    # custom_model = CustomModel(n, R=5)
    # print(custom_model.get_graph_representation())

    # # G(n,p) Model
    # gnp_model = GnpModel(n, p)
    # gnp_model.draw_graph()
    # print("G(n,p) Model Graph Representation:")
    # print(gnp_model.get_graph_representation())

    # # Layered Model
    # layered_model = LayeredModel(n, layers)
    # layered_model.draw_graph()
    # print("Layered Model Graph Representation:")
    # print(layered_model.get_graph_representation())

if __name__ == "__main__":
    main()
