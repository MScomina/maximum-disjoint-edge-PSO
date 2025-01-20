import networkx as nx
import time
import random
import copy
import rustworkx as rw

MAX_TIME = 3600*8
MAX_ITER = 2500

def MSGA_MEDP(graph : rw.PyGraph, commodity_pairs : list[tuple[int, int]], n_iter : int = MAX_ITER, max_time_s : int = MAX_TIME) -> tuple[int, dict]:
    '''
        Multi-Stage Genetic Algorithm for the Minimum Edge Disjoint Paths problem.
    '''
    if type(graph) == rw.PyDiGraph:
        graph = graph.to_undirected()

    best_solution : int = 0
    best_paths : dict[tuple[int, int], list[int]] = {}

    start_time = time.time()
    iter_count = 0

    while time.time() - start_time < max_time_s and iter_count < n_iter:

        current_solution : int = 0
        current_paths : dict[tuple[int, int], list[int]] = {}

        random.shuffle(commodity_pairs)
        current_graph = copy.deepcopy(graph)

        for start, end in commodity_pairs:
            try:
                shortest_path = list(rw.dijkstra_shortest_paths(
                    current_graph,
                    start,
                    end,
                    default_weight=1.0
                )[end])
                for i in range(len(shortest_path) - 1):
                    current_graph.remove_edge(shortest_path[i], shortest_path[i+1])
                current_solution += 1
                current_paths[(start, end)] = shortest_path
            except (rw.NoPathFound, IndexError):
                current_paths[(start, end)] = []
                continue
            
        if current_solution > best_solution:
            best_solution = current_solution
            best_paths = current_paths
        iter_count += 1
    
    return best_solution, best_paths