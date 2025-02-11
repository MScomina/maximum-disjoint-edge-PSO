import time

import rustworkx as rw

from solvers.gurobi import generate_gurobi_model, generate_gurobi_model_efficient
from solvers.metaheuristics.MSGA import MSGA_MEDP
from solvers.metaheuristics.LaPSO import LaPSO_MEDP

from utils.graph_utils import generate_unique_node_pairs, generate_collated_node_pairs


SOLVERS = [
    "gurobi", 
    "gurobi_efficient", 
    "msga", 
    "lapso"
]

N_TESTS = 3
PERCENTAGE_NODES_USED = [0.3, 0.5]
VERBOSE = False

MAX_ALGORITHM_EXECUTION_TIME_SEC = 60


def test_graph(graph : rw.PyGraph, solvers : list[str] = SOLVERS, n_tests : int = N_TESTS, percentage_nodes_used : list[float] = PERCENTAGE_NODES_USED,
                verbose : bool = VERBOSE, max_algorithm_execution_time_sec : int = MAX_ALGORITHM_EXECUTION_TIME_SEC, custom_pairs : list[tuple[int, int]] | None = None) -> dict:
    '''
        Tests a single graph on all the algorithms N_TESTS times with randomized pairs.
    '''

    metrics = ["avg_time", "avg_connected_pairs", "avg_optimality_gap"]

    statistics = {solver: {metric: 0 for metric in metrics} for solver in solvers}

    algorithm_times = {solver: [] for solver in solvers}
    algorithm_connected_pairs = {solver: [] for solver in solvers}
    algorithm_optimality_gap = {solver: [] for solver in solvers}
    
    for k in range(n_tests):
        for percentage in percentage_nodes_used:
            if percentage < 0.0 or percentage > 0.66:
                raise Exception("Percentage of nodes used for pairs must be between 0 and 0.66.")
            
            if not custom_pairs:
                print(f"Test {k+1}/{n_tests} - ({percentage*100}% nodes used for pairs) - {graph.attrs['name']}...", end=" ")

                if graph.attrs["name"].startswith("collated"):
                    n_clusters, cluster_size = map(int, graph.attrs["name"].split("_")[2:])
                    pairs_intracluser = max(1, int(cluster_size * percentage)//2)
                    pairs_intercluster = min(max(0, int(0.5*(cluster_size-2*pairs_intracluser)/(n_clusters-1))),5)
                    pairs = generate_collated_node_pairs(
                        node_range=(0, graph.num_nodes()),
                        amount_intracluster=pairs_intracluser,
                        amount_intercluster=pairs_intercluster,  
                        n_clusters=n_clusters
                    )
                else:
                    pairs = generate_unique_node_pairs(
                        (0, graph.num_nodes()),
                        int(graph.num_nodes()*percentage)//2
                    )
            else:
                pairs = custom_pairs
            
            for solver in SOLVERS:
                time_start = time.time()
                match solver:
                    case "gurobi":
                        model = generate_gurobi_model(graph, pairs, verbose=verbose, max_seconds=max_algorithm_execution_time_sec)
                        model.optimize()
                        result = (model.ObjVal, model.ObjBound)
                        algorithm_times[solver].append(time.time() - time_start)
                    case "gurobi_efficient":
                        model = generate_gurobi_model_efficient(graph, pairs, verbose=verbose, max_seconds=max_algorithm_execution_time_sec)
                        model.optimize()
                        result = (model.ObjVal, model.ObjBound)
                        algorithm_times[solver].append(time.time() - time_start)
                    case "msga":
                        result, _ = MSGA_MEDP(graph, pairs, verbose=verbose, max_seconds=max_algorithm_execution_time_sec)
                        algorithm_times[solver].append(time.time() - time_start)
                    case "lapso":
                        result, _ = LaPSO_MEDP(graph, pairs, verbose=verbose, max_seconds=max_algorithm_execution_time_sec, print_frequency=1)
                        algorithm_times[solver].append(time.time() - time_start)
                    case _:
                        raise Exception("Solver not supported.")
                
                algorithm_connected_pairs[solver].append(result[0])
                algorithm_optimality_gap[solver].append((result[1]-result[0])/result[1])

            
            if not custom_pairs:
                print("done.")
            
    for solver in solvers:
        statistics[solver]["avg_time"] = sum(algorithm_times[solver])/len(algorithm_times[solver])
        statistics[solver]["avg_connected_pairs"] = sum(algorithm_connected_pairs[solver])/len(algorithm_connected_pairs[solver])
        statistics[solver]["avg_optimality_gap"] = sum(algorithm_optimality_gap[solver])/len(algorithm_optimality_gap[solver])

    print("\n")
    return statistics