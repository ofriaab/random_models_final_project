from SchedulerSimulator import SchedulerSimulator
from SchedulingAlgorithm import SimpleQueueAlgorithm, MinimalRuntimeAlgorithm, MaxOutdegreeAlgorithm
import os
import gsf_prof
import sys
import pandas as pd


def parse_file(file_name):
    global nodes_dict

    class Node:
        def __init__(self, name, line, start, end, priority, dependencies=None):
            self.name = name
            self.line = line
            self.start = start
            self.end = end
            self.duration = end - start
            self.priority = priority
            self.dependencies = dependencies if dependencies else []
            self.predecessors = []
            self.successors = []
            self.earliest_start = 0
            self.latest_start = float('inf')
            self.slack = 0

    def parse_file(filename):
        global nodes_dict
        nodes_dict = {}
        prof = gsf_prof.GSFProfDecoder(sys.argv[1])
        apps = prof.read()
        for app in apps.values():
            for frame, records in app['frames'].items():
                if 'profiles/gsf.%06d.prof' % frame == filename:
                    def traverse(nodes):
                        for node in nodes:
                            for r in node.ranges:
                                name = r.name
                                line_number = r.line
                                start_time = r.start
                                end_time = r.end
                                priority = r.priority
                                dependencies = [d.name for d in r.dep]
                                new_node = Node(name, line_number, start_time, end_time, priority, dependencies)
                                nodes_dict[name] = new_node
                            traverse(node.children)

                    traverse(records.time_range_tree())
                    break

        return nodes_dict

    def build_tree():
        global nodes_dict
        tasks = []
        for node in nodes_dict.values():
            for dep_name in node.dependencies:
                dep_node = nodes_dict.get(dep_name)
                if dep_node:
                    dep_node.predecessors.append(node)
                    node.successors.append(dep_node)
                    dep_node.parent = node
        for node in nodes_dict.values():
            tasks.append((node.name, node.duration, len(node.predecessors), [dep for dep in node.dependencies]))
        return tasks

    parse_file(file_name)
    tasks = build_tree()
    return tasks


def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(file)
    return file_paths


# Specify the folder path from which you want to get file paths
folder_path = 'C:\\Users\\yairr\\OneDrive\\מסמכים\\FinalProject\\profiles'

# Get all file paths in the specified folder
all_files = get_file_paths(folder_path)

# Initialize results list
results = []

# Process each file using the parse_file function
algorithms = [SimpleQueueAlgorithm(), MinimalRuntimeAlgorithm(), MaxOutdegreeAlgorithm()]
algorithm_names = ["SimpleQueueAlgorithm", "MinimalRuntimeAlgorithm", "MaxOutdegreeAlgorithm"]

for file_path in all_files:
    if file_path.endswith('.prof'):
        print(file_path)
        tasks = parse_file(f'profiles/{file_path}')
        num_processors = 4
        row = {"File": file_path, "Number of Nodes": len(tasks)}
        for algorithm, algo_name in zip(algorithms, algorithm_names):
            simulator = SchedulerSimulator(tasks, num_processors, algorithm)
            simulator.run()
            makespan = simulator.get_makespan()
            row[algo_name] = makespan
        results.append(row)

# Save results to an Excel file
df = pd.DataFrame(results)
output_file = 'C:\\Users\\yairr\\OneDrive\\מסמכים\\FinalProject\\results.xlsx'
df.to_excel(output_file, index=False)
print(f'Results saved to {output_file}')
