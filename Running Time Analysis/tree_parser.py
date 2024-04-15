import os
from openpyxl import Workbook
import gsf_prof
import sys


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

    def find_roots():
        global nodes_dict
        nodes = nodes_dict
        roots = []
        for node in nodes.values():
            if not hasattr(node, 'parent'):
                roots.append(node)
        return roots

    def find_duration_of_file():
        global nodes_dict
        min_time = min([child.start for child in nodes_dict.values()])
        max_time = max([child.end for child in nodes_dict.values()])
        return max_time - min_time

    def calculate_critical_path_duration():
        global nodes_dict
        graph = nodes_dict

        # Step 1: Calculate Earliest Start Times (ES) and Latest Start Times (LS)
        for node in graph.values():
            node.earliest_start = 0
            for pred in node.predecessors:
                node.earliest_start = max(node.earliest_start, pred.earliest_start + pred.duration)

        end_node = max(graph.values(), key=lambda n: n.earliest_start)
        end_node.latest_start = end_node.earliest_start
        for node in reversed(list(graph.values())):
            node.latest_start = end_node.latest_start
            for succ in node.successors:
                node.latest_start = min(node.latest_start, succ.latest_start - node.duration)

        # Step 2: Calculate Slack Time
        for node in graph.values():
            node.slack = node.latest_start - node.earliest_start - node.duration

        # Step 3: Identify Critical Path and Calculate Duration
        critical_path_edges = []
        critical_path_duration = 0

        # Traverse the graph from the end node to find the critical path
        current_node = end_node
        while current_node:
            if not current_node.predecessors:  # Found a starting node
                critical_path_edges.append((current_node.name, current_node.successors[0].name))
                critical_path_duration += current_node.duration
                break
            next_node = max(current_node.predecessors, key=lambda n: n.earliest_start)
            critical_path_edges.append((next_node.name, current_node.name))
            critical_path_duration += current_node.duration
            current_node = next_node

        critical_path_edges.reverse()  # Reverse the edges to get the correct order

        return critical_path_duration, critical_path_edges

    parse_file(file_name)
    build_tree()
    roots = find_roots()
    # for root in roots:
    #     print(root.name)
    # After finding roots in your code
    duration = find_duration_of_file()
    #

    critical_path_duration, critical_path_edges = calculate_critical_path_duration()

    return duration, critical_path_duration, len(critical_path_edges),len(nodes_dict)




def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:

            file_paths.append(file)
    return file_paths


def write_to_excel(files_name,total_duration, critical_path_duration, critical_path_num_vert,nodes_count, output_file):
    wb = Workbook()
    ws = wb.active

    # Add headers
    headers = ['File Name','Total Duration', 'Critical Path Duration', 'Critical Path Num Vert','Total Number Of Nodes']
    ws.append(headers)

    # Add data rows
    for data in zip(files_name,total_duration, critical_path_duration, critical_path_num_vert,nodes_count):
        ws.append(data)

    # Save the workbook
    wb.save(output_file)

durations = []
critical_path_durations = []
critical_path_length = []
total_nodes_count=[]

# Specify the folder path from which you want to get file paths
folder_path = 'C:\\Users\yairr\OneDrive\מסמכים\FinalProject\profiles'

# Get all file paths in the specified folder
all_files = get_file_paths(folder_path)

# Process each file using the parse_file function
for file_path in all_files:
    print(file_path)
    duration,critical_path_duration,length_critical_path,nodes_count = parse_file(f'profiles/{file_path}')
    durations.append(duration)
    critical_path_durations.append(critical_path_duration)
    critical_path_length.append(length_critical_path)
    total_nodes_count.append(nodes_count)

output_file = 'running_time_data.xlsx'

# Write data to Excel file
write_to_excel(all_files,durations, critical_path_durations, critical_path_length,total_nodes_count, output_file)
print('done')