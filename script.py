from utils.dataset_parser import load_all_graphs, load_graph
from utils.graph_utils import get_graph_info, draw_graph, generate_unique_node_pairs, generate_collated_node_pairs
from solvers.gurobi import generate_gurobi_model, generate_gurobi_model_efficient
from solvers.metaheuristics.MSGA import MSGA_MEDP
from solvers.metaheuristics.LaPSO import LaPSO_MEDP
import networkx as nx
import random
import numpy as np
import gurobipy as gp

MSGA_ITERATIONS = 1000
SEED = 31415
TEST = "LaPSO"

def test_collated_graph(path : str, normal_pairs : int, collated_pairs : int, test : str = "gurobi"):
    graph = load_graph(path)
    pairs = generate_collated_node_pairs((0, graph.number_of_nodes()-1), normal_pairs, collated_pairs)
    print(get_graph_info(graph))
    print(f"Number of pairs: {len(pairs)}")
    print(pairs)
    print(f"Number of hypothetical maximum paths: {normal_pairs*10 + 45*(1 if collated_pairs > 0 else 0)}")

    #draw_graph(graph)

    match test:
        case "gurobi":
            model = generate_gurobi_model_efficient(graph, pairs, verbose=True)
            model.optimize()
        case "msga":
            result, paths = MSGA_MEDP(graph, pairs, MSGA_ITERATIONS)
        case "LaPSO":
            result, paths = LaPSO_MEDP(graph, pairs)

    print(f"\nConnected pairs ({test}): {result}\n")
    print(paths)

def test_graph(path : str | nx.Graph, n_pairs : int | list[tuple[int, int]], test : str = "gurobi"):
    if type(path) == str:
        graph = load_graph(path)
    else:
        graph = path
    print(get_graph_info(graph))
    if type(n_pairs) == int:
        pairs = generate_unique_node_pairs((0, graph.number_of_nodes()-1), n_pairs)
    else:
        pairs = n_pairs
    print(len(pairs), pairs)

    draw_graph(graph)

    match test:
        case "gurobi":
            model = generate_gurobi_model_efficient(graph, pairs, verbose=True)
            model.optimize()
        case "msga":
            result, paths = MSGA_MEDP(graph, pairs, MSGA_ITERATIONS)
            print(result)
            print(paths)
            print("\n")
        case "LaPSO":
            result, paths = LaPSO_MEDP(graph, pairs)
            print(result)
            print(paths)
            print("\n")

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    #test_collated_graph("data/collated_graph_500.bb", 11, 3, test="msga")
    #test_collated_graph("data/collated_graph_500.bb", 11, 3, test=TEST)
    #test_collated_graph("data/collated_graph_1500.bb", 45, 6, test=TEST)
    barabasi_graph = nx.barabasi_albert_graph(150, 2, seed=SEED)
    barabasi_pairs = generate_unique_node_pairs((0, 150), 60)  #133
    for edge in barabasi_graph.edges():
        if random.random() < 0.1:
            barabasi_graph.remove_edge(*edge)
            if not nx.is_connected(barabasi_graph):
                barabasi_graph.add_edge(*edge)
    #test_graph(barabasi_graph, barabasi_pairs, test="gurobi")
    test_graph(barabasi_graph, barabasi_pairs, test="msga")
    test_graph(barabasi_graph, barabasi_pairs, test=TEST)
    test_graph(barabasi_graph, barabasi_pairs, test="gurobi")