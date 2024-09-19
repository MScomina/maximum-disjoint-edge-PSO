import networkx as nx
import random
import time
import copy

# Hyperparameters
VELOCITY_FACTOR = 0.1
GLOBAL_FACTOR = 0.05
SUBGRADIENT_FACTOR = 2
BETA_SUBG_FACTOR = 0.9
PERTURBATION_FACTOR = 0.5
SWARM_SIZE = 8

# Constraints
MAX_SECONDS = 3600
MAXITER = 30000
LBC = 10

HEURISTIC_TYPE = "LVP"
HEURISTIC_TYPES = ["RAND", "LVA", "LVP"]

# Lagrangian relaxed function: min λ>=0 LR(λ) = sum_((i,j) in E)(λ_ij) + |K| - sum_(k in K)(min{1, min_(s_k->t_k path P_k)(sum_(i,j) in P_k)(λ_ij)})

def LaPSO_MEDP(graph : nx.Graph, commodity_pairs : list[tuple[int, int]], max_seconds : int = MAX_SECONDS, max_iterations : int = MAXITER,
               velocity_factor : float = VELOCITY_FACTOR, global_factor : float = GLOBAL_FACTOR, subgradient_factor : float = SUBGRADIENT_FACTOR,
               perturbation_factor : float = PERTURBATION_FACTOR, beta_subg_factor : float = BETA_SUBG_FACTOR, swarm_size : int = SWARM_SIZE) -> int:
    
    particles = initialize_particles(
        graph=graph, 
        n_commodities=len(commodity_pairs), 
        swarm_size=swarm_size,
        subgradient_factor=subgradient_factor
    )

    parameters = {
        "velocity_factor" : velocity_factor,
        "global_factor" : global_factor,
        "beta_subg_factor" : beta_subg_factor,
        "perturbation_factor" : perturbation_factor,
        "best_particle" : particles[0],
        "lower_bound" : -float("inf"),
        "upper_bound" : float("inf"),
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


def particle_swarm_step(graph : nx.Graph, commodity_pairs : list[tuple[int, int]], particles : list[dict], parameters : dict) -> list[dict]:

    for particle in particles:
        relaxed_base_graph = generate_base_relaxed_graph(graph, particle["lambda"])
        commodity_sum = 0

        # Initialize the subgradient to all 1s on existing edges
        subgradient = {}
        for edge in relaxed_base_graph.edges:
            subgradient[edge] = 1.0

        paths = []

        for commodity, (source, target) in enumerate(commodity_pairs):

            if parameters["iteration"] % LBC == 0:
                relaxed_perturbed_graph = copy.deepcopy(relaxed_base_graph)
            else:
                relaxed_perturbed_graph = generate_perturbed_graph(relaxed_base_graph, particle, commodity)

            length, path = nx.single_source_dijkstra(
                relaxed_perturbed_graph, 
                source, 
                target, 
                weight="weight", 
                cutoff=1.0
            )

            # Update the subgradient of constraint (6): sum_k (x_ijk) <= 1, or "each edge can be used at most once"
            for i in range(len(path) - 1):
                subgradient[(path[i], path[i+1])] -= 1.0

            # Compute the length of the best path. length will always be less than or equal to 1 because of the graph structure (worst case scenario the path is the extra added edge).
            commodity_sum += length
            paths.append(path)

        # Compute the relaxed value of the particle
        current_relaxed_solution = sum(particle["lambda"].values()) + len(commodity_pairs) - commodity_sum
        particle["iterations_since_bound_update"] += 1

        if current_relaxed_solution > particle["lower_bound"]:
            particle["lower_bound"] = current_relaxed_solution
            particle["iterations_since_bound_update"] = 0
        elif particle["iterations_since_bound_update"] >= LBC:
            particle["subgradient_factor"] *= parameters["beta_subg_factor"]
            particle["iterations_since_bound_update"] = 0

        if current_relaxed_solution > parameters["lower_bound"]:
            parameters["lower_bound"] = current_relaxed_solution
        
        infractions = feasibility_check(paths)
        if sum(infractions) > 0:
            paths = repair_heuristic(graph, paths, infractions, particle["perturbation"])
        
        if sum(feasibility_check(paths)) == 0:
            connected_commodities = sum(1 for path in paths if len(path) > 0)
            if connected_commodities < parameters["upper_bound"]:
                parameters["upper_bound"] = connected_commodities
                parameters["best_particle"] = particle

        # Updates to velocities and positions are still missing, so that's what's left to do.

def initialize_particles(graph : nx.Graph, n_commodities : int, n_particles : int, subgradient_factor : float = SUBGRADIENT_FACTOR) -> list[dict]:
    '''
        Initializes the particles as a list of dictionaries, where each dictionary (particle) contains the following:
        - lambda: edge weights of the graph.
        - perturbation: list of perturbations for each edge, where the list is of size n_commodities.
        - lambda_velocity: velocity of the lambda values, as per PSO.
        - perturbation_velocity: velocity of the perturbation values, as per PSO.
        - subgradient_factor: factor to be used in the subgradient calculation.
        - lower_bound: current lower bound of the particle.
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
            "perturbation_velocity" : perturbation_velocities,
            "subgradient_factor": subgradient_factor,
            "lower_bound": 0.0,
            "iterations_since_bound_update": 0
        })

    return particles


# Note: the graph has to be a MultiGraph because of the potential case of having 
# the source and destination nodes already connected and having to add the weight 1 edge as per paper description.
def generate_base_relaxed_graph(particle : dict, commodity_pairs : list[tuple[int, int]]) -> nx.MultiGraph:

    relaxed_base_graph = nx.MultiGraph()
    
    for edge in commodity_pairs:
        relaxed_base_graph.add_edge(edge[0], edge[1], weight=1.0, key="alternative")

    for edge in particle["lambda"].keys():
        relaxed_base_graph.add_edge(edge[0], edge[1], weight=particle["lambda"][edge], key="lambda")

    return relaxed_base_graph


def generate_perturbed_graph(graph : nx.MultiGraph, particle : dict, commodity_number : int) -> nx.MultiGraph:

    perturbed_graph = copy.deepcopy(graph)

    for edge in particle["lambda"].keys():
        perturbed_graph[edge[0]][edge[1]]["lambda"]["weight"] += particle["perturbation"][edge][commodity_number]

    return perturbed_graph


def feasibility_check(paths : list[list[int]]) -> list[int]:
    '''
        Given a list of paths, checks for the number of infractions (same edge used more than once) for each path.
        An infraction is defined as a pair of edges that are used more than once in the paths.

        E.g.:
        If the paths are [[0, 1, 2, 3], [1, 2, 4, 5], [4, 5, 6]], the resulting output will be [1, 2, 1] (the number of infractions for each path), 
        since paths 1 and 2 both use the edge (1, 2) and paths 2 and 3 both use the edge (4, 5).
    '''
    pairs_count = {}

    # Add the pairs to the dictionary
    for path in paths:
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            if edge in pairs_count:
                pairs_count[edge] += 1
            else:
                pairs_count[edge] = 1

    infractions = [0 for _ in range(len(paths))]

    # Check whether the edge is used more than once
    for i, path in enumerate(paths):
        for j in range(len(path) - 1):
            edge = (path[j], path[j+1])
            if pairs_count[edge] > 1:
                infractions[i] += 1

    return infractions


def repair_heuristic(graph : nx.Graph, paths : list[list[int]], infractions : list[int], perturbations : dict[tuple[int, int], list[int]] = None,
                     heuristic_type : str = HEURISTIC_TYPE) -> list[list[int]]:
    '''
        Given a list of paths and the number of infractions for each path, repairs the paths by removing the infractions.
        The repair is done using a heuristic, which can be one of the following:
        - RAND: Randomly picks a commodity and tries to repair the path.
        - LVA: Largest Violation Arbitrary, picks the commodity with the largest number of infractions and tries to repair it.
        - LVP: Largest Violation Perturbation, picks the commodity with the largest number of infractions and tries to repair it using the perturbations as graph weights.
    '''

    if heuristic_type not in HEURISTIC_TYPES:
        raise ValueError(f"Invalid heuristic type. Must be one of {HEURISTIC_TYPES}.")
    
    if heuristic_type == "LVP" and perturbations is None:
        raise ValueError("Perturbations must be provided when using the LVP heuristic.")

    repaired_paths = copy.deepcopy(paths)
    
    # Generate a list with the indices of the paths that have infractions and sort them by the number of infractions in descending order
    indices_with_infractions = [i for i, infraction in enumerate(infractions) if infraction > 0]
    if len(indices_with_infractions) == 0:
        return repaired_paths
    indices_with_infractions.sort(key=lambda x: infractions[x], reverse=True)
    
    while len(indices_with_infractions) > 0:
        if heuristic_type == "RAND":
            picked_commodity_idx = indices_with_infractions[random.randint(0, len(indices_with_infractions) - 1)]
        else:
            picked_commodity_idx = indices_with_infractions[0]

        # Remove all used edges from the graph
        current_graph = copy.deepcopy(graph)

        # If using the LVP heuristic, calculate the minimum value of the perturbation so to offset the graph weights to positive values.
        if heuristic_type == "LVP":
            perturbation_offset = min(min(perturbation_list[picked_commodity_idx] for perturbation_list in perturbations.values()), 0)
            for edge in perturbations.keys():
                current_graph[edge[0]][edge[1]]["weight"] = perturbations[edge][picked_commodity_idx] - perturbation_offset

        for i, path in enumerate(repaired_paths):
            if i == picked_commodity_idx:
                continue
            for j in range(len(path) - 1):
                if (path[j], path[j+1]) in current_graph.edges:
                    current_graph.remove_edge(path[j], path[j+1])

        try:
            if heuristic_type == "LVP":
                shortest_path = nx.shortest_path(current_graph, repaired_paths[picked_commodity_idx][0], repaired_paths[picked_commodity_idx][-1], weight="weight")
            else:
                shortest_path = nx.shortest_path(current_graph, repaired_paths[picked_commodity_idx][0], repaired_paths[picked_commodity_idx][-1])
            repaired_paths[picked_commodity_idx] = shortest_path
            indices_with_infractions.remove(picked_commodity_idx)
        except nx.NetworkXNoPath:
            # Unfixable/unreroutable path, remove it from the list
            repaired_paths[picked_commodity_idx] = []
            indices_with_infractions.remove(picked_commodity_idx)

    return repaired_paths