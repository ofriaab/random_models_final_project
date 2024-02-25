from deps_analyzer import parse_graph_from_file
from deps_analyzer import transitive_closure

def main(file_name):
    new_graph = parse_graph_from_file(file_name)
    trans_closure_matrix = transitive_closure(new_graph.neighborhood_matrix)
    

main('/Users/o.abramovich/random_models_final_project-1/Graph Analysis/deps/0c29e58f0523f2c5cb56fa5ed0fb1faaab70f765.deps')
print("done")