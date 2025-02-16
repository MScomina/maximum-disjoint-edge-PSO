import time

import rustworkx as rw

from solvers.gurobi import generate_gurobi_model, generate_gurobi_model_efficient
from solvers.metaheuristics.MSGA import MSGA_MEDP
from solvers.metaheuristics.LaPSO import LaPSO_MEDP

from utils.graph_utils import generate_unique_node_pairs, generate_collated_node_pairs, get_graph_info

import gurobipy as gp


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

    if ("gurobi" in solvers or "gurobi_efficient" in solvers):
        if verbose:
            gp.setParam("OutputFlag", 1)
        else:
            gp.setParam("OutputFlag", 0)

    metrics = ["avg_time", "avg_connected_pairs", "avg_optimality_gap"]

    if n_tests > 1:
        metrics.append("std_time")

    statistics = {percentage: {solver: {metric: 0 for metric in metrics} for solver in solvers} for percentage in percentage_nodes_used}

    algorithm_times = {percentage: {solver: [] for solver in solvers} for percentage in percentage_nodes_used}
    algorithm_connected_pairs = {percentage: {solver: [] for solver in solvers} for percentage in percentage_nodes_used}
    algorithm_optimality_gap = {percentage: {solver: [] for solver in solvers} for percentage in percentage_nodes_used}

    print(f"Testing graph {graph.attrs['name']}:")
    print(get_graph_info(graph), "\n")
    
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
            
            for solver in solvers:
                time_start = time.perf_counter()
                match solver:
                    case "gurobi":
                        model = generate_gurobi_model(graph, pairs, verbose=verbose, max_seconds=max_algorithm_execution_time_sec)
                        model.optimize()
                        result = (model.ObjVal, model.ObjBound)
                        algorithm_times[percentage][solver].append(time.perf_counter() - time_start)
                        model.close()
                    case "gurobi_efficient":
                        model = generate_gurobi_model_efficient(graph, pairs, verbose=verbose, max_seconds=max_algorithm_execution_time_sec)
                        model.optimize()
                        result = (model.ObjVal, model.ObjBound)
                        algorithm_times[percentage][solver].append(time.perf_counter() - time_start)
                        model.close()
                    case "msga":
                        result, _ = MSGA_MEDP(graph, pairs, verbose=verbose, max_seconds=max_algorithm_execution_time_sec)
                        algorithm_times[percentage][solver].append(time.perf_counter() - time_start)
                    case "lapso":
                        result, _ = LaPSO_MEDP(graph, pairs, verbose=verbose, max_seconds=max_algorithm_execution_time_sec, print_frequency=1)
                        algorithm_times[percentage][solver].append(time.perf_counter() - time_start)
                    case _:
                        raise Exception(f"Solver {solver} not supported.")
                
                algorithm_connected_pairs[percentage][solver].append(result[0])
                algorithm_optimality_gap[percentage][solver].append((result[1]-result[0])/result[1])
            
            if not custom_pairs:
                print("done.")

    gp.disposeDefaultEnv()
            
    for percentage in percentage_nodes_used:
        for solver in solvers:
            statistics[percentage][solver]["avg_time"] = sum(algorithm_times[percentage][solver])/len(algorithm_times[percentage][solver])
            statistics[percentage][solver]["avg_connected_pairs"] = sum(algorithm_connected_pairs[percentage][solver])/len(algorithm_connected_pairs[percentage][solver])
            statistics[percentage][solver]["avg_optimality_gap"] = sum(algorithm_optimality_gap[percentage][solver])/len(algorithm_optimality_gap[percentage][solver])
            if "std_time" in metrics:
                statistics[percentage][solver]["std_time"] = (sum([(time - statistics[percentage][solver]["avg_time"])**2 for time in algorithm_times[percentage][solver]])/len(algorithm_times[percentage][solver]))**0.5
            statistics[percentage][solver]["times"] = algorithm_times[percentage][solver]
    print("\n")
    return statistics