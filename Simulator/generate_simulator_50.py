import graph_models as gm
from SchedulerSimulator import SchedulerSimulator
import SchedulingAlgorithm as sa
import pandas as pd
import os
import matplotlib.pyplot as plt

def main():
    n = 7120  # Number of nodes for the Custom model
    R = 5     # Parameter for the Custom model

    # Define the algorithms to be tested
    algorithms = [sa.SimpleQueueAlgorithm(), sa.MinimalRuntimeAlgorithm(), sa.MaxOutdegreeAlgorithm()]
    algorithm_names = ["SimpleQueueAlgorithm", "MinimalRuntimeAlgorithm", "MaxOutdegreeAlgorithm"]

    # Initialize the results list
    results = []
    all_task_durations = []



    num_processors = 4

    # Run the simulations
    for i in range(50):
         # Custom Model
        print(f'iteration : {i}')
        custom_model = gm.CustomModel(n, R)
        tasks = custom_model.get_graph_representation()

        row = {"Iteration": i + 1, "Number of Nodes": len(tasks)}
        task_durations = []
        for algorithm, algo_name in zip(algorithms, algorithm_names):
            simulator = SchedulerSimulator(tasks, num_processors, algorithm)
            simulator.run()
            makespan = simulator.get_makespan()
            row[algo_name] = makespan

            # Store task durations

        task_durations.extend([task[1]for task in tasks])

        results.append(row)
        all_task_durations.extend(task_durations)

    # Save results to an Excel file
    df = pd.DataFrame(results)
    output_file = os.path.join(os.path.dirname(__file__), 'ctm_simulation_results.xlsx')
    df.to_excel(output_file, index=False)
    print(f'Results saved to {output_file}')

    # Plot the distribution of task durations
    plt.hist(all_task_durations, bins=50, alpha=0.75, edgecolor='black')
    plt.title('Distribution of Task Durations Over 50 Iterations')
    plt.xlabel('Task Duration')
    plt.ylabel('Frequency')
    plt.grid(True)
    plot_file = os.path.join(os.path.dirname(__file__), 'ctm_task_duration_distribution.png')
    plt.savefig(plot_file)
    plt.show()
    print(f'Distribution plot saved to {plot_file}')

if __name__ == "__main__":
    main()
