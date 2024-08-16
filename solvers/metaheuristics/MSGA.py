import networkx as nx
import time
import random

def MSGA_MEDP(graph : nx.Graph, commodity_pairs : list[tuple[int, int]], n_iter : int, max_time_s : int = 3600) -> int:
    '''
        Multi-Stage Genetic Algorithm for the Minimum Edge Disjoint Paths problem.
    '''
    if type(graph) == nx.DiGraph:
        graph = graph.to_undirected()

    best_solution : int = 0
    best_paths : dict[tuple[int, int], list[int]] = {}

    start_time = time.time()
    iter_count = 0

    while time.time() - start_time < max_time_s and iter_count < n_iter:

        current_solution : int = 0
        current_paths : dict[tuple[int, int], list[int]] = {}

        random.shuffle(commodity_pairs)
        current_graph = graph.copy()

        for start, end in commodity_pairs:
            try:
                shortest_path = nx.shortest_path(current_graph, start, end)
                for i in range(len(shortest_path) - 1):
                    current_graph.remove_edge(shortest_path[i], shortest_path[i+1])
                current_solution += 1
                current_paths[(start, end)] = shortest_path
            except nx.NetworkXNoPath:
                continue
            
        if current_solution > best_solution:
            best_solution = current_solution
            best_paths = current_paths
        iter_count += 1
    
    return best_solution, best_paths