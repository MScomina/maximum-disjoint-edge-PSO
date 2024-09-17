import networkx as nx
import random
import time
import copy

# Hyperparameters
VELOCITY_FACTOR = 0.1
GLOBAL_FACTOR = 0.05
SUBGRADIENT_FACTOR = 2
PERMUTATION_FACTOR = 0.5
SWARM_SIZE = 8

# Constraints
MAX_SECONDS = 3600
MAXITER = 30000
LBC = 10

# Lagrangian relaxed function: min λ>=0 LR(λ) = sum_((i,j) in E)(λ_ij) + |K| - sum_(k in K)(min{1, min_(s_k->t_k path P_k)(sum_(i,j) in P_k)(λ_ij)})

def LaPSO_MEDP(graph : nx.Graph, commodity_pairs : list[tuple[int, int]], max_seconds : int = MAX_SECONDS, max_iterations : int = MAXITER,
               velocity_factor : float = VELOCITY_FACTOR, global_factor : float = GLOBAL_FACTOR, subgradient_factor : float = SUBGRADIENT_FACTOR,
               permutation_factor : float = PERMUTATION_FACTOR, swarm_size : int = SWARM_SIZE) -> int:
    
    particles = initialize_particles(
        graph=graph, 
        n_commodities=len(commodity_pairs), 
        swarm_size=swarm_size
    )

    parameters = {
        "velocity_factor" : velocity_factor,
        "global_factor" : global_factor,
        "subgradient_factor" : subgradient_factor,
        "permutation_factor" : permutation_factor,
        "best_particle" : particles[0],
        "iteration" : 0
    }

    start_time = time.time()

    while time.time() - start_time < max_seconds and parameters["iteration"] < max_iterations:
        particles = particle_swarm_step(
            graph=graph, 
            commodity_pairs=commodity_pairs, 
            particles=particles,
            parameters=parameters
        )
        parameters["iteration"] += 1


def particle_swarm_step(graph : nx.Graph, commodity_pairs : list[tuple[int, int]], particles : list[dict[str, dict[tuple[int, int], float | list[float]]]],
                        parameters : dict) -> list[dict[str, dict[tuple[int, int], float | list[float]]]]:

    for particle in particles:
        relaxed_base_graph = generate_base_relaxed_graph(graph, particle["lambda"])
        commodity_sum = 0

        for commodity, (source, target) in enumerate(commodity_pairs):

            if parameters["iteration"] % LBC == 0:
                relaxed_perturbed_graph = copy.deepcopy(relaxed_base_graph)
            else:
                relaxed_perturbed_graph = generate_perturbed_graph(relaxed_base_graph, particle, commodity)

            length, _ = nx.single_source_dijkstra(relaxed_perturbed_graph, source, target, weight="weight", cutoff=1.0)
            commodity_sum += length


def initialize_particles(graph : nx.Graph, n_commodities : int, n_particles : int) -> list[dict[str, dict[tuple[int, int], float | list[float]]]]:
    '''
        Initializes the particles as a list of dictionaries, where each dictionary contains the following:
        - lambda: edge weights of the graph.
        - perturbation: list of perturbations for each edge, where the list is of size n_commodities.
        - lambda_velocity: velocity of the lambda values, as per PSO.
        - perturbation_velocity: velocity of the perturbation values, as per PSO.
    '''
    particles = []

    for _ in range(n_particles):

        lambda_values = {}
        lambda_velocities = {}
        perturbations = {}
        perturbation_velocities = {}

        for edge in graph.edges:
            lambda_values[edge] = random.uniform(0, 0.1)
            lambda_velocities[edge] = 0.0
            perturbations[edge] = [0.0 for _ in range(n_commodities)]
            perturbation_velocities[edge] = [0.0 for _ in range(n_commodities)]

        particles.append({
            "lambda" : lambda_values,
            "perturbation" : perturbations,
            "lambda_velocity" : lambda_velocities,
            "perturbation_velocity" : perturbation_velocities
        })

    return particles


# Note: the graph has to be a MultiGraph because of the potential case of having 
# the source and destination nodes already connected and having to add the weight 1 edge as per paper description.
def generate_base_relaxed_graph(particle : dict[str, dict[tuple[int, int], float | list[float]]], commodity_pairs : list[tuple[int, int]]) -> nx.MultiGraph:

    relaxed_base_graph = nx.MultiGraph()
    
    for edge in commodity_pairs:
        relaxed_base_graph.add_edge(edge[0], edge[1], weight=1.0, key="alternative")

    for edge in particle["lambda"].keys():
        relaxed_base_graph.add_edge(edge[0], edge[1], weight=particle["lambda"][edge], key="lambda")

    return relaxed_base_graph


def generate_perturbed_graph(graph : nx.MultiGraph, particle : dict[str, dict[tuple[int, int], float | list[float]]], commodity_number : int) -> nx.MultiGraph:

    perturbed_graph = copy.deepcopy(graph)

    for edge in particle["lambda"].keys():
        perturbed_graph[edge[0]][edge[1]]["lambda"]["weight"] += particle["perturbation"][edge][commodity_number]

    return perturbed_graph