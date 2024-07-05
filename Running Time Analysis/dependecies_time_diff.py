import os

import gsf_prof
import sys
import matplotlib.pyplot as plt



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
        for node in nodes_dict.values():
            for dep_name in node.dependencies:
                dep_node = nodes_dict.get(dep_name)
                if dep_node:
                    dep_node.predecessors.append(node)
                    node.successors.append(dep_node)
                    dep_node.parent = node

    def find_time_diff():
        global nodes_dict
        time_diff = []
        for node in nodes_dict.values():
            for dep_name in node.dependencies:
                dep_node = nodes_dict.get(dep_name)
                time_diff.append(dep_node.start-node.end)
        return time_diff


    parse_file(file_name)
    build_tree()
    time_diff=find_time_diff()
    return time_diff




def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:

            file_paths.append(file)
    return file_paths



# Specify the folder path from which you want to get file paths
folder_path = 'C:\\Users\yairr\OneDrive\מסמכים\FinalProject\profiles'

# Get all file paths in the specified folder
all_files = get_file_paths(folder_path)
total_time_diff=[]
# Process each file using the parse_file function
for file_path in all_files:
    print(file_path)
    file_time_diff = parse_file(f'profiles/{file_path}')

    total_time_diff.extend(file_time_diff)


def remove_highest_10_percent(values):
    if not values:
        return values

    sorted_values = sorted(values)
    n = len(sorted_values)
    num_to_remove = int(n * 0.1)

    # Remove the highest 10%
    result = sorted_values[:n - num_to_remove]
    return result

# Example usage

result = remove_highest_10_percent(total_time_diff)


# Plotting
plt.figure(figsize=(10, 6))  # Adjust size if necessary
plt.hist(result, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('t(v) - t(u)')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of t(v) - t(u)')
plt.grid(True)
plt.show()


print('done')