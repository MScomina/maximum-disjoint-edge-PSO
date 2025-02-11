'''
    Script meant to be run to test the algorithms on small instances of the MEDP problem.
    Generates a set of small graphs and tests them with the solvers defined in the SOLVERS list.
'''
import os
import rustworkx as rw
import time
import pprint
import contextlib
import random

import gurobipy as gp

from utils.dataset_parser import load_all_graphs
from utils.graph_utils import get_graph_info
from utils.dataset_generator import generate_save_default_graphs, save_graph_draw
from utils.test_utils import test_graph


# List of solvers to be tested.
SOLVERS = [
    "gurobi", 
    "gurobi_efficient", 
    "msga", 
    "lapso"
]

# Number of times to run the test for each graph.
N_TESTS = 3

# Percentage of nodes used for pairs.
PERCENTAGE_NODES_USED = [0.3, 0.5]

# Verbosity of the algorithms. If true, shows all algorithm outputs.
VERBOSE = False

# If true, writes the output to output_test.txt instead of the console.
WRITE_TO_TEXT_FILE = True

# Determines whether the example graph from the paper should be shown.
SHOW_EXAMPLE_GRAPH = True

MAX_ALGORITHM_EXECUTION_TIME_SEC = 60

SEED = 2718281828



def example_graph():
    '''
        This is a recreation of the graph represented in Fig. 1 from the paper.
        
        This serves as an easy example to show the correct functioning of the algorithms. 

        Expected results are a maximum of 2 connected pairs out of 4.
    '''

    graph = rw.PyGraph()
    graph.attrs = {"name": "example_graph"}
    graph.extend_from_edge_list([
        (0, 4), (1, 4), (2, 5), (3, 5), (4, 5),

        # Bridge edges (limits the maximum number of connected pairs to 2)
        (4, 6), (5, 7),

        (6, 7), (6, 8), (6, 9), (7, 10), (7, 11)
    ])

    # Manually define the pairs of nodes to be connected.
    pairs = [(0, 8), (1, 9), (2, 10), (3, 11)]

    print(f"Testing graph {graph.attrs['name']}:")
    print(get_graph_info(graph), "\n")

    pprint.pprint(test_graph(
        graph,
        n_tests=1,
        custom_pairs=pairs
    ))

    if not os.path.exists("./data/generated/test/images"):
        os.makedirs("./data/generated/test/images")
    save_graph_draw(graph, "./data/generated/test/images/example_graph.png")
    
    print("-"*80)
    print("\n")
       

def main():

    random.seed(SEED)

    if not VERBOSE:
        gp.setParam("OutputFlag", 0)

    if SHOW_EXAMPLE_GRAPH:
        example_graph()

    pp = pprint.PrettyPrinter(indent=4)

    if not os.path.exists("./data/generated/test"):
        generate_save_default_graphs(test=True, seed=SEED)

    graphs = load_all_graphs("./data/generated/test")

    start_time = time.time()   

    for graph in graphs:
        test_time = time.time()
        print(f"Testing graph {graph.attrs['name']}:")
        print(get_graph_info(graph), "\n")

        stats = test_graph(
            graph,
            solvers=SOLVERS,
            n_tests=N_TESTS,
            percentage_nodes_used=PERCENTAGE_NODES_USED,
            verbose=VERBOSE,
            max_algorithm_execution_time_sec=MAX_ALGORITHM_EXECUTION_TIME_SEC
        )

        pp.pprint(stats)
        print(f"\nTest took {time.time() - test_time:.3f} seconds.\n")
        print("-"*80)
        print("\n")

    print(f"Total time: {time.time() - start_time:.3f} seconds.")


if __name__ == "__main__":
    if WRITE_TO_TEXT_FILE:
        print("Writing output to output_test.txt...")
        with open("output_test.txt", "w") as f, contextlib.redirect_stdout(f):
            main()
    else:
        main()