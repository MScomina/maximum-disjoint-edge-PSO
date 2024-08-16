import networkx as nx
import os
import pandas as pd
import numpy as np

EDGE_WEIGHTS = 1

#
#   For some reason the paper, despite mentioning "undirected graphs", calculates some metrics based on direction.
#   Because of that, everything is generated as a directed graph and transformed to undirected when required.
#

def load_graph(file_path : str) -> nx.DiGraph | None:
    '''
        Loads a single graph from a specific file path. Supported types: .csv, .bb
    '''
    graph = nx.DiGraph()
    file_extension = os.path.splitext(file_path)[1]
    match file_extension:
        case ".csv":
            df = pd.read_csv(file_path, names=["node_1", "node_2", "weights"])
            for _, row in df.iterrows():
                graph.add_edge(row["node_1"], row["node_2"], weight=EDGE_WEIGHTS)
        case ".bb":
            with open(file_path, "r") as file:
                for line, text in enumerate(file):
                    if line < 2:
                        continue
                    node_1, node_2, _ = text.split()
                    graph.add_edge(node_1, node_2, weight=EDGE_WEIGHTS)
        case _:
            print(f"File extension {file_extension} not supported.")
            return
        
    nodes = graph.nodes()
    if type(next(iter(nodes))) == str:
        old_mapping = sorted([int(node) for node in nodes])
        mapping = {str(old_label): new_label for new_label, old_label in enumerate(old_mapping)}
    else:
        old_mapping = sorted([node for node in nodes])
        mapping = {old_label: new_label for new_label, old_label in enumerate(old_mapping)}
    graph = nx.relabel_nodes(graph, mapping)
    
    return graph

def load_all_graphs(folder_path : str) -> list[nx.DiGraph]:
    '''
        Loads all graphs from a specific folder path.
    '''
    graphs = []
    for file in os.listdir(folder_path):
        graph = load_graph(os.path.join(folder_path, file))
        if graph:
            graphs.append(graph)
    return graphs