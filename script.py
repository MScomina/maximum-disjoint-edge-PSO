from utils.dataset_parser import load_all_graphs, load_graph
from utils.graph_utils import get_graph_info, draw_graph, generate_unique_node_pairs, generate_collated_node_pairs
from solvers.gurobi import generate_gurobi_model, generate_gurobi_model_efficient
from solvers.metaheuristics.MSGA import MSGA_MEDP
from solvers.metaheuristics.LaPSO import LaPSO_MEDP
import networkx as nx

MSGA_ITERATIONS = 1000
TEST = "LaPSO"

def test_collated_graph(path : str, normal_pairs : int, collated_pairs : int, test : str = "gurobi"):
    graph = load_graph(path)
    pairs = generate_collated_node_pairs((0, graph.number_of_nodes()-1), normal_pairs, collated_pairs)
    print(len(pairs), pairs)

    draw_graph(graph)

    match test:
        case "gurobi":
            model = generate_gurobi_model_efficient(graph, pairs, verbose=True)
            model.optimize()
        case "msga":
            print(MSGA_MEDP(graph, pairs, MSGA_ITERATIONS))
        case "LaPSO":
            print(sum(1 if len(list) > 0 else 0 for list in LaPSO_MEDP(graph, pairs)))

def test_graph(path : str, n_pairs : int, test : str = "gurobi"):
    graph = load_graph(path)
    print(get_graph_info(graph))
    pairs = generate_unique_node_pairs((0, graph.number_of_nodes()-1), n_pairs)
    print(len(pairs), pairs)

    draw_graph(graph)

    match test:
        case "gurobi":
            model = generate_gurobi_model_efficient(graph, pairs, verbose=True)
            model.optimize()
        case "msga":
            print(MSGA_MEDP(graph, pairs, MSGA_ITERATIONS))
        case "LaPSO":
            print(sum(1 if len(list) > 0 else 0 for list in LaPSO_MEDP(graph, pairs)))

if __name__ == "__main__":
    test_collated_graph("data/collated_graph_500.bb", 5, 3, test=TEST)
    test_graph("data/graph_2_degree1.csv", 200, test=TEST)