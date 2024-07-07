import graph_models as gm
from SchedulerSimulator import SchedulerSimulator
import SchedulingAlgorithm as sa


def main():
    n = 50  # Number of nodes
    p = 0.1  # Probability for G(n,p) model
    layers = 5  # Number of layers for layer-by-layer model

    # Custom Model
    # custom_model = gm.CustomModel(n, R=5)
    # # custom_model.draw_graph()
    # print("Custom Model Graph Representation:")
    # tasks=custom_model.get_graph_representation()


    # G(n,p) Model
    gnp_model = gm.GnpModel(n, p)
    gnp_model.draw_graph()
    print("G(n,p) Model Graph Representation:")
    tasks=gnp_model.get_graph_representation()
    print(tasks)

    # Layered Model
    # layered_model = gm.LayeredModel(n, layers)
    # tasks=layered_model.get_graph_representation()
    num_processors=4
    algorithm=sa.MinimalRuntimeAlgorithm()
    simulator = SchedulerSimulator(tasks, num_processors, algorithm)
    simulator.run()
    simulator.save_statistics(file_name='lbl')
if __name__ == "__main__":
    main()

