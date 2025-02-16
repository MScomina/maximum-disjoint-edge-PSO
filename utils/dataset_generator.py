from utils.dataset_parser import save_graph, save_graph_draw
from utils.graph_utils import generate_collated_graph
import random
import rustworkx.generators as rwg
import numpy as np
import os

DEFAULT_GRAPHS_TEST = {
    "regular_graphs": {
        "n_nodes": [50, 75]
    },
    "collated_graphs": {
        "cluster_number_sizes": [
            (3, 10),
            (5, 10)
        ]
    },
    "grid_graphs": {
        "grid_shapes": [
            (8, 5)
        ]
    },
    "hex_lattice_graphs": {
        "hex_lattice_shapes": [
            (6, 4)
        ]
    }
}

DEFAULT_GRAPHS_SCALABILITY = {
    "regular_graphs": {
        "n_nodes": [50, 100, 150, 250]
    },
    "collated_graphs": {
        "cluster_number_sizes": [
            (5, 25),
            (6, 30),
            (8, 30),
            (10, 30),
            (10, 50),
        ]
    },
    "grid_graphs": {
        "grid_shapes": [
            (8, 6),
            (10, 8),
            (15, 12),
            (25, 15)
        ]
    },
    "hex_lattice_graphs": {
        "hex_lattice_shapes": [

        ]
    }
}

DEFAULT_SEED = 314159265
DEFAULT_PATH_TEST = "./data/generated/test"
DEFAULT_PATH_SCALABILITY = "./data/generated/scalability"
SAVE_IMAGES = True

def generate_save_default_graphs(graphs: dict = DEFAULT_GRAPHS_SCALABILITY, folder_path: str = DEFAULT_PATH_SCALABILITY,
                                 seed: int = DEFAULT_SEED, is_test_mode : bool = False, save_images : bool = SAVE_IMAGES) -> None:
    '''
        Generates and saves test graphs with the following formats:
        - Regular graphs: regular_graph_{n_nodes}.edgelist
        - Collated graphs: collated_graph_{cluster_number}_{nodes_per_cluster}.edgelist
        - Grid graphs: grid_graph_{rows}_{columns}.edgelist
        - Hexagonal lattice graphs: hex_lattice_graph_{rows}_{columns}.edgelist
    '''

    random.seed(seed)
    np.random.seed(seed)

    if is_test_mode:
        graphs = DEFAULT_GRAPHS_TEST
        folder_path = DEFAULT_PATH_TEST

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Regular graphs
    for n_nodes in graphs["regular_graphs"]["n_nodes"]:
        graph = generate_collated_graph(
            n_clusters=1, 
            cluster_size=n_nodes, 
            average_degree_cluster=4.0, 
            min_degree=2
        )
        save_path = f"{folder_path}/regular_graph_{n_nodes}.edgelist"
        save_graph(graph, save_path)
        save_image_path = f"{folder_path}/images/regular_graph_{n_nodes}.png"
        if save_images:
            if not os.path.exists(f"{folder_path}/images"):
                os.makedirs(f"{folder_path}/images")
            save_graph_draw(graph, save_image_path)

    # Collated graphs
    for n_clusters, cluster_size in graphs["collated_graphs"]["cluster_number_sizes"]:
        graph = generate_collated_graph(
            n_clusters=n_clusters, 
            cluster_size=cluster_size, 
            average_degree_cluster=5.0,
            min_degree=2
        )
        save_path = f"{folder_path}/collated_graph_{n_clusters}_{cluster_size}.edgelist"
        save_graph(graph, save_path)
        save_image_path = f"{folder_path}/images/collated_graph_{n_clusters}_{cluster_size}.png"
        if save_images:
            if not os.path.exists(f"{folder_path}/images"):
                os.makedirs(f"{folder_path}/images")
            save_graph_draw(graph, save_image_path)

    # Grid graphs
    for grid_shape in graphs["grid_graphs"]["grid_shapes"]:
        graph = rwg.grid_graph(*grid_shape)
        save_path = f"{folder_path}/grid_graph_{grid_shape[0]}_{grid_shape[1]}.edgelist"
        save_graph(graph, save_path)
        save_image_path = f"{folder_path}/images/grid_graph_{grid_shape[0]}_{grid_shape[1]}.png"
        if save_images:
            if not os.path.exists(f"{folder_path}/images"):
                os.makedirs(f"{folder_path}/images")
            save_graph_draw(graph, save_image_path)

    # Hexagonal lattice graphs
    for hex_lattice_shape in graphs["hex_lattice_graphs"]["hex_lattice_shapes"]:
        graph = rwg.hexagonal_lattice_graph(*hex_lattice_shape, periodic=True)
        save_path = f"{folder_path}/hex_lattice_graph_{hex_lattice_shape[0]}_{hex_lattice_shape[1]}.edgelist"
        save_graph(graph, save_path)
        save_image_path = f"{folder_path}/images/hex_lattice_graph_{hex_lattice_shape[0]}_{hex_lattice_shape[1]}.png"
        if save_images:
            if not os.path.exists(f"{folder_path}/images"):
                os.makedirs(f"{folder_path}/images")
            save_graph_draw(graph, save_image_path)