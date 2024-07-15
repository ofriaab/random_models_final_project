import graph_models as gm
from SchedulerSimulator import SchedulerSimulator
import SchedulingAlgorithm as sa


def main():
    n = 5207  # Number of nodes
    p = 0.1  # Probability for G(n,p) model
    layers = 5  # Number of layers for layer-by-layer model

    # # Custom Model
    # custom_model = gm.CustomModel(n, R=5)
    # tasks=custom_model.get_graph_representation()


    # G(n,p) Model
    gnp_model = gm.GnpModel(n, p)
    tasks=gnp_model.get_graph_representation()


    # Layered Model
    # layered_model = gm.LayeredModel(n, layers)
    # tasks=layered_model.get_graph_representation()


    num_processors=4
    # algorithm = sa.SimpleQueueAlgorithm()
    # algorithm=sa.MinimalRuntimeAlgorithm()
    algorithm=sa.MaxOutdegreeAlgorithm()
    simulator = SchedulerSimulator(tasks, num_processors, algorithm)
    simulator.run()
    simulator.save_statistics(file_name='gnp_moda_5207')
    print(f'number of nodes: {len(tasks)}')
if __name__ == "__main__":
    main()

