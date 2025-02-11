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
        "cluster_numbers": [3, 5],
        "cluster_sizes": [10]
    },
    "grid_graphs": {
        "grid_shapes": [
            (8,5)
        ]
    },
    "hex_lattice_graphs": {
        "hex_lattice_shapes": [
            (6,4)
        ]
    }
}

DEFAULT_GRAPHS_SCALABILITY = {
    "regular_graphs": {
        "n_nodes": [50, 100, 250, 500]
    },
    "collated_graphs": {
        "cluster_numbers": [3,5,10],
        "cluster_sizes": [10,25,50]
    },
    "grid_graphs": {
        "grid_shapes": [
            (10,8),
            (15,12),
            (20,15)
        ]
    },
    "hex_lattice_graphs": {
        "hex_lattice_shapes": [
            (8,6),
            (12,8)
        ]
    }
}

DEFAULT_SEED = 314159
DEFAULT_PATH_TEST = "./data/generated/test"
DEFAULT_PATH_SCALABILITY = "./data/generated/scalability"
SAVE_IMAGES = True

def generate_save_default_graphs(graphs: dict = DEFAULT_GRAPHS_SCALABILITY, folder: str = DEFAULT_PATH_SCALABILITY,
                                 seed: int = DEFAULT_SEED, test : bool = False, save_images : bool = SAVE_IMAGES) -> None:
    '''
        Generates and saves test graphs with the following formats:
        - Regular graphs: regular_graph_{n_nodes}.edgelist
        - Collated graphs: collated_graph_{cluster_number}_{nodes_per_cluster}.edgelist
        - Grid graphs: grid_graph_{rows}_{columns}.edgelist
        - Hexagonal lattice graphs: hex_lattice_graph_{rows}_{columns}.edgelist
    '''

    random.seed(seed)
    np.random.seed(seed)

    if test:
        graphs = DEFAULT_GRAPHS_TEST
        folder = DEFAULT_PATH_TEST

    if not os.path.exists(folder):
        os.makedirs(folder)

    for n_nodes in graphs["regular_graphs"]["n_nodes"]:
        graph = generate_collated_graph(
            n_clusters=1, 
            cluster_size=n_nodes, 
            average_degree_cluster=4.0, 
            min_degree=2
        )
        save_path = f"{folder}/regular_graph_{n_nodes}.edgelist"
        save_graph(graph, save_path)
        save_image_path = f"{folder}/images/regular_graph_{n_nodes}.png"
        if save_images:
            if not os.path.exists(f"{folder}/images"):
                os.makedirs(f"{folder}/images")
            save_graph_draw(graph, save_image_path)

    for n_clusters in graphs["collated_graphs"]["cluster_numbers"]:
        for cluster_size in graphs["collated_graphs"]["cluster_sizes"]:
            graph = generate_collated_graph(
                n_clusters=n_clusters, 
                cluster_size=cluster_size, 
                average_degree_cluster=5.0,
                min_degree=2
            )
            save_path = f"{folder}/collated_graph_{n_clusters}_{cluster_size}.edgelist"
            save_graph(graph, save_path)
            save_image_path = f"{folder}/images/collated_graph_{n_clusters}_{cluster_size}.png"
            if save_images:
                if not os.path.exists(f"{folder}/images"):
                    os.makedirs(f"{folder}/images")
                save_graph_draw(graph, save_image_path)

    for grid_shape in graphs["grid_graphs"]["grid_shapes"]:
        graph = rwg.grid_graph(*grid_shape)
        save_path = f"{folder}/grid_graph_{grid_shape[0]}_{grid_shape[1]}.edgelist"
        save_graph(graph, save_path)
        save_image_path = f"{folder}/images/grid_graph_{grid_shape[0]}_{grid_shape[1]}.png"
        if save_images:
            if not os.path.exists(f"{folder}/images"):
                os.makedirs(f"{folder}/images")
            save_graph_draw(graph, save_image_path)

    for hex_lattice_shape in graphs["hex_lattice_graphs"]["hex_lattice_shapes"]:
        graph = rwg.hexagonal_lattice_graph(*hex_lattice_shape, periodic=True)
        save_path = f"{folder}/hex_lattice_graph_{hex_lattice_shape[0]}_{hex_lattice_shape[1]}.edgelist"
        save_graph(graph, save_path)
        save_image_path = f"{folder}/images/hex_lattice_graph_{hex_lattice_shape[0]}_{hex_lattice_shape[1]}.png"
        if save_images:
            if not os.path.exists(f"{folder}/images"):
                os.makedirs(f"{folder}/images")
            save_graph_draw(graph, save_image_path)