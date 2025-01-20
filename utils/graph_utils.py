import networkx as nx
import rustworkx as rw
import matplotlib.pyplot as plt
import random

def convert_rustworkx_to_networkx(graph : rw.PyGraph) -> nx.Graph:
    '''
        Convert a rustworkx PyGraph or PyDiGraph to a networkx graph.
    '''
    out_graph = nx.Graph()
    for node in graph.node_indices():
        out_graph.add_node(node)
    for edge in graph.edge_list():
        out_graph.add_edge(edge[0], edge[1])
    
    return out_graph

def draw_graph(graph : nx.Graph | rw.PyGraph):
    '''
        Draws a graph using networkx and matplotlib.
        Converts rustworkx graphs to networkx graphs if necessary.
    '''
    if type(graph) == rw.PyGraph:
        graph = convert_rustworkx_to_networkx(graph)
    if type(graph) == nx.DiGraph:
        graph = graph.to_undirected()
    nx.draw(graph, with_labels=True, node_size=50)
    plt.show()


def get_graph_info(graph : nx.DiGraph) -> dict[str, int | float | tuple]:
    '''
        Returns information about the graph:
            - Number of nodes ("nodes")
            - Number of edges ("edges")
            - Minimum degree ("min_deg")
            - Maximum degree ("max_deg")
            - Average degree ("avg_deg")
            - Diameter ("diam")
        
        If the graph is a rustworkx graph, it is converted to a networkx graph.
    '''

    #   For some reason the paper only counts outward edges for degrees.
    current_graph = graph
    if type(current_graph) == rw.PyGraph:
        current_graph = convert_rustworkx_to_networkx(current_graph)
    
    if type(current_graph) == rw.PyDiGraph:
        new_graph = nx.DiGraph()
        for node in graph.node_indexes():
            new_graph.add_node(node)
        for edge in graph.edge_list():
            new_graph.add_edge(edge[0], edge[1])
        current_graph = new_graph

    if type(current_graph) == nx.DiGraph:
        degrees = dict(current_graph.out_degree())
    else:
        degrees = dict(current_graph.degree())
    
    return {
        "nodes": current_graph.number_of_nodes(),
        "edges": current_graph.number_of_edges(),
        "min_deg": min(degrees.values()),
        "max_deg": max(degrees.values()),
        "avg_deg": sum(degrees.values()) / current_graph.number_of_nodes(),
        "diam": nx.diameter(current_graph.to_undirected())
    }


def generate_unique_node_pairs(node_range : tuple[int, int], n : int) -> list[tuple[int, int]]:
    '''
        Generates a list of unique node pairs.
    '''
    if n > (node_range[1] - node_range[0] + 1) // 2:
        raise ValueError("Not enough unique nodes to generate the required number of pairs without overlap.")
    non_selected_nodes = list(range(*node_range))
    pairs = []
    for _ in range(n):
        source = random.choice(non_selected_nodes)
        non_selected_nodes.remove(source)
        target = random.choice(non_selected_nodes)
        non_selected_nodes.remove(target)
        pairs.append((source, target))
    return pairs

def generate_collated_node_pairs(node_range : tuple[int, int], n_intracluster : int, n_intercluster : int, n_clusters : int = 10) -> list[tuple[int, int]]:
    '''
        Generates a list of nodes that forces intra-cluster and inter-cluster connections:

        - n_intracluster: Number of intra-cluster connections for each cluster. <br>
        - n_intercluster: Number of inter-cluster connections for each pair of clusters. <br>
        
        Note: It is assumed that the cluster nodes are sorted to be contiguous (i.e. cluster 0 nodes are 0, 1, 2, ..., cluster 1 nodes are n, n+1, n+2, ...).
    '''
    total_nodes = n_intracluster * n_clusters * 2 + (n_intercluster * n_clusters * (n_clusters-1))
    if total_nodes > (node_range[1] - node_range[0] + 1):
        raise ValueError("Not enough unique nodes to generate the required number of pairs without overlap.")
    
    size_cluster = (node_range[1] - node_range[0] + 1) // n_clusters

    pairs : list[tuple[int, int]] = []
    
    for i in range(n_clusters):
        pairs += generate_unique_node_pairs((i*size_cluster, (i+1)*size_cluster-1), n_intracluster)

    remaining_nodes : list[list[int]] = [
        list(range(i * size_cluster, (i + 1) * size_cluster)) for i in range(n_clusters)
    ]

    for source, target in pairs:
        source_cluster = source // size_cluster
        target_cluster = target // size_cluster
        remaining_nodes[source_cluster].remove(source)
        remaining_nodes[target_cluster].remove(target)

    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            for _ in range(n_intercluster):
                source = random.choice(remaining_nodes[i])
                target = random.choice(remaining_nodes[j])
                pairs.append((source, target))
                remaining_nodes[i].remove(source)
                remaining_nodes[j].remove(target)
    
    return pairs