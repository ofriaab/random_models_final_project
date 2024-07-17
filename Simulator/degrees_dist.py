import graph_models as gm
import matplotlib.pyplot as plt
from collections import Counter
import os

def main():
    n = 7120  # Number of nodes for the Custom model
    R = 5     # Parameter for the Custom model

    # Custom Model
    custom_model = gm.CustomModel(n, R)
    tasks = custom_model.get_graph_representation()

    # Calculate in-degrees and out-degrees
    in_degrees = Counter()
    out_degrees = Counter()

    for task in tasks:
        node_id, _, _, dependencies = task
        out_degrees[node_id] = len(task[3])
        in_degrees[node_id]=task[2]
    # Plot in-degree distribution
    in_degree_values = list(in_degrees.values())
    plt.hist(in_degree_values, bins=50, alpha=0.75, edgecolor='black')
    plt.title('Distribution of In-Degrees for Custom Model')
    plt.xlabel('In-Degree')
    plt.ylabel('Frequency')
    plt.grid(True)
    in_degree_plot_file = os.path.join(os.path.dirname(__file__), 'cst_in_degree_distribution.png')
    plt.savefig(in_degree_plot_file)
    plt.show()
    print(f'In-degree distribution plot saved to {in_degree_plot_file}')

    # Plot out-degree distribution
    out_degree_values = list(out_degrees.values())
    plt.hist(out_degree_values, bins=50, alpha=0.75, edgecolor='black')
    plt.title('Distribution of Out-Degrees for Custom Model')
    plt.xlabel('Out-Degree')
    plt.ylabel('Frequency')
    plt.grid(True)
    out_degree_plot_file = os.path.join(os.path.dirname(__file__), 'cst_out_degree_distribution.png')
    plt.savefig(out_degree_plot_file)
    plt.show()
    print(f'Out-degree distribution plot saved to {out_degree_plot_file}')

if __name__ == "__main__":
    main()
