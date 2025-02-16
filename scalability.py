'''
    Script meant to be run to test the scalability of the algorithms on large instances of the MEDP problem.
    Generates a set of large graphs and tests them with the solvers defined in the SOLVERS list.
'''
import os
import time
import pprint
import contextlib
import random

from utils.dataset_parser import load_all_graphs
from utils.dataset_generator import generate_save_default_graphs
from utils.test_utils import test_graph


# List of solvers to be tested.
SOLVERS = [
    "gurobi_efficient", 
    "msga", 
    "lapso"
]

# Number of times to run the test for each graph.
N_TESTS = 3

# Percentage of nodes used for pairs.
PERCENTAGE_NODES_USED = [0.66]

# Verbosity of the algorithms. If true, shows all algorithm outputs.
VERBOSE = False

# If true, writes the output to output_scalability.txt instead of the console.
WRITE_TO_TEXT_FILE = True

MAX_ALGORITHM_EXECUTION_TIME_SEC = 1200

SEED = 1618033988


def main():

    random.seed(SEED)

    pp = pprint.PrettyPrinter(indent=4)

    if not os.path.exists("./data/generated/scalability"):
        generate_save_default_graphs(seed=SEED)

    graphs = load_all_graphs("./data/generated/scalability")

    graphs.sort(key=lambda x: x.attrs["name"])

    start_time = time.perf_counter()

    for graph in graphs:
        test_time = time.perf_counter()

        stats = test_graph(
            graph,
            solvers=SOLVERS,
            n_tests=N_TESTS,
            percentage_nodes_used=PERCENTAGE_NODES_USED,
            verbose=VERBOSE,
            max_algorithm_execution_time_sec=MAX_ALGORITHM_EXECUTION_TIME_SEC
        )

        pp.pprint(stats)
        print(f"\nTest took {time.perf_counter() - test_time:.3f} seconds.\n")
        print("-"*80)
        print("\n")

    print(f"Total time: {time.perf_counter() - start_time:.3f} seconds.")


if __name__ == "__main__":
    if WRITE_TO_TEXT_FILE:
        print("Writing output to output_scalability.txt...")
        with open("output_scalability.txt", "w") as f, contextlib.redirect_stdout(f):
            main()
    else:
        main()

