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

    def find_standalone_nodes():
        global nodes_dict
        nodes = nodes_dict
        standalone_nodes = []
        for node in nodes.values():
            if not hasattr(node, 'parent') and len(node.dependencies)==0:
                standalone_nodes.append(node)
        return standalone_nodes

    parse_file(file_name)
    build_tree()
    stanalone_nodes=find_standalone_nodes()
    return len(stanalone_nodes),len(nodes_dict)




def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:

            file_paths.append(file)
    return file_paths



total_nodes_count=[]

# Specify the folder path from which you want to get file paths
folder_path = 'C:\\Users\yairr\OneDrive\מסמכים\FinalProject\profiles'

# Get all file paths in the specified folder
all_files = get_file_paths(folder_path)

# Process each file using the parse_file function
for file_path in all_files:
    print(file_path)
    standalone_nodes_count,nodes_count = parse_file(f'profiles/{file_path}')

    total_nodes_count.append((standalone_nodes_count,nodes_count))
    if standalone_nodes_count!=0:
        print(f'standalone nodes count:{standalone_nodes_count} in {nodes_count} nodes' )
    else:
        print('no standalone nodes')




print('done')