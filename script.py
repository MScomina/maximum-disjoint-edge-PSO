from utils.dataset_parser import load_all_graphs, load_graph
from utils.graph_utils import get_graph_info, draw_graph, generate_unique_node_pairs, generate_collated_node_pairs
from solvers.gurobi import generate_gurobi_model, generate_gurobi_model_efficient
from solvers.metaheuristics.MSGA import MSGA_MEDP
import networkx as nx

MSGA_ITERATIONS = 1000

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

if __name__ == "__main__":
    test_collated_graph("data/collated_graph_1500.bb", 50, 2, test="msga")
    test_graph("data/graph_2_degree1.csv", 200, test="msga")