import networkx as nx
import rustworkx as rw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
import warnings
import numpy as np

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
        Graph drawing utility.
    '''
    if type(graph) == rw.PyGraph:
        graph = convert_rustworkx_to_networkx(graph)
    if type(graph) == nx.DiGraph:
        graph = graph.to_undirected()
    nx.draw(graph, with_labels=True, node_size=50)
    plt.show()
    plt.close()


def get_graph_info(graph : nx.DiGraph) -> dict[str, int | float]:
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


def generate_unique_node_pairs(node_range : tuple[int, int], amount : int) -> list[tuple[int, int]]:
    '''
        Generates a list of random unique node pairs.
    '''
    if amount > (node_range[1] - node_range[0] + 1) // 2:
        raise ValueError("Not enough unique nodes to generate the required number of pairs without overlap.")
    non_selected_nodes = list(range(*node_range))
    pairs = []
    for _ in range(amount):
        source = random.choice(non_selected_nodes)
        non_selected_nodes.remove(source)
        target = random.choice(non_selected_nodes)
        non_selected_nodes.remove(target)
        pairs.append((source, target))
    return pairs

def generate_collated_node_pairs(node_range : tuple[int, int], amount_intracluster : int, amount_intercluster : int,
                                 n_clusters : int = 10) -> list[tuple[int, int]]:
    '''
        Generates a list of nodes that forces intra-cluster and inter-cluster connections:

        - amount_intracluster: Number of intra-cluster connections for each cluster.
        - amount_intercluster: Number of inter-cluster connections for each pair of clusters.
        
        Note: It is assumed that the cluster nodes are sorted to be contiguous (i.e. cluster 0 nodes are 0, 1, 2, ..., cluster 1 nodes are n, n+1, n+2, ...).
    '''
    total_nodes = amount_intracluster * n_clusters * 2 + (amount_intercluster * n_clusters * (n_clusters-1))
    if total_nodes > (node_range[1] - node_range[0] + 1):
        raise ValueError("Not enough unique nodes to generate the required number of pairs without overlap.")
    
    size_cluster = (node_range[1] - node_range[0] + 1) // n_clusters

    pairs : list[tuple[int, int]] = []
    
    for i in range(n_clusters):
        pairs += generate_unique_node_pairs((i*size_cluster, (i+1)*size_cluster-1), amount_intracluster)

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
            for _ in range(amount_intercluster):
                source = random.choice(remaining_nodes[i])
                target = random.choice(remaining_nodes[j])
                pairs.append((source, target))
                remaining_nodes[i].remove(source)
                remaining_nodes[j].remove(target)
    
    return pairs

def generate_collated_graph(n_clusters : int, cluster_size : int, average_degree_cluster : float = 5.0, min_degree : int = 2) -> rw.PyGraph:
    '''
        Generates a collated graph with the specified number of clusters and nodes per cluster.

        A collated graph in this case is a graph composed by n subgraphs, each connected to each other by a single edge.
        This ensures that the maximum amount of nodes that are connect-able between each pair of subgraphs without reusing the edge is 1, 
        given a requirement to connect them all to each other.
    '''

    # Since the function generates intercluster connections without considering average_degree_cluster, 
    # it is possible to generate a graph with a much higher degree than expected when the number of clusters is high compared to the cluster size.
    min_connections_degree = (average_degree_cluster) + (n_clusters*(n_clusters-1)) / (n_clusters*cluster_size)
    if average_degree_cluster < (min_connections_degree - average_degree_cluster + 1) / 2:
        warnings.warn(f"Warning: Average degree per cluster ({average_degree_cluster}) will differ a lot from the total degree (~{min_connections_degree}) because of a high number of clusters per cluster size ratio.")
    
    if 2*min_degree > average_degree_cluster:
        raise ValueError("Minimum degree must be less than half of the average degree per cluster.")

    if average_degree_cluster > cluster_size:
        raise ValueError("Average degree per cluster must be less than the cluster size.")
    

    graph = rw.PyGraph()

    # Generate the clusters and immediately connect the nodes to ensure min_degree.
    for i in range(n_clusters):
        for j in range(cluster_size):
            graph.add_node(i * cluster_size + j)
            current_degree = 0
            while current_degree < min_degree and j > current_degree:
                # Create a weighted list to prefer higher numbers.
                # This ensures an even distribution of edges, because otherwise the first nodes would have a higher degree, being picked more often.
                choices = list(range(i * cluster_size, i * cluster_size + j))
                weights = np.logspace(1, 1+min_degree, len(choices))
                random_node = random.choices(choices, weights=weights, k=1)[0]
                if graph.has_edge(i * cluster_size + j, random_node):
                    continue
                graph.add_edge(i * cluster_size + j, random_node, None)
                current_degree += 1
        
    total_cluster_connections = int(average_degree_cluster * cluster_size / 2)

    # Add extra intra-cluster connections.
    for i in range(n_clusters):
        current_cluster_connections = (cluster_size - 1)*min_degree
        while current_cluster_connections < total_cluster_connections:
            random_node_1 = random.choice(range(i * cluster_size, (i+1) * cluster_size))
            random_node_2 = random.choice(range(i * cluster_size, (i+1) * cluster_size))
            if random_node_1 == random_node_2:
                continue
            graph.add_edge(random_node_1, random_node_2, None)
            current_cluster_connections += 1
    
    # Add inter-cluster connections.
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            random_node_1 = random.choice(range(i * cluster_size, (i+1) * cluster_size))
            random_node_2 = random.choice(range(j * cluster_size, (j+1) * cluster_size))
            graph.add_edge(random_node_1, random_node_2, None)

    return graph