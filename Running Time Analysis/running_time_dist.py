import os
import numpy as np
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
        running_time_lst=[]
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
                                running_time_lst.append(end_time-start_time)
                                nodes_dict[name] = new_node
                            traverse(node.children)

                    traverse(records.time_range_tree())
                    break

        return running_time_lst






    running_time_lst=parse_file(file_name)
    return running_time_lst




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
total_running_time_lst=[]
# Process each file using the parse_file function
for file_path in all_files:
    print(file_path)
    file_running_time_lst = parse_file(f'profiles/{file_path}')

    total_running_time_lst.extend(file_running_time_lst)



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

result = remove_highest_10_percent(total_running_time_lst)
average_running_time = np.mean(result)
print(f'the mean is : {average_running_time}')
# Calculate the histogram data
counts, bin_edges = np.histogram(result, bins=50)

# Create a dictionary of bin edges and counts
hist_dict = {f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}': counts[i] for i in range(len(counts))}

# Print the dictionary
print(hist_dict)


# Plotting
plt.figure(figsize=(10, 6))  # Adjust size if necessary
plt.hist(result, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('running time')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of running time of tasks')
plt.grid(True)
plt.show()


print('done')