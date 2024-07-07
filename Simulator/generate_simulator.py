import graph_models as gm
from SchedulerSimulator import SchedulerSimulator
import SchedulingAlgorithm as SA


def main():
    n = 100  # Number of nodes
    p = 0.1  # Probability for G(n,p) model
    layers = 5  # Number of layers for layer-by-layer model

    # Custom Model
    # custom_model = gm.CustomModel(n, R=5)
    # custom_model.draw_graph()
    # print("Custom Model Graph Representation:")
    # tasks=custom_model.get_graph_representation()
    # print(tasks)

    # G(n,p) Model
    # gnp_model = gm.GnpModel(n, p)
    # gnp_model.draw_graph()
    # print("G(n,p) Model Graph Representation:")
    # print(gnp_model.get_graph_representation())

    # # Layered Model
    # layered_model = gm.LayeredModel(n, layers)
    # # layered_model.draw_graph()
    # print("Layered Model Graph Representation:")
    # print(layered_model.get_graph_representation())

if __name__ == "__main__":
    main()

