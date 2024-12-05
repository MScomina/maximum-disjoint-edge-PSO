import networkx as nx
import random
import time
import copy
import numpy as np
import concurrent.futures

# Hyperparameters
VELOCITY_FACTOR = 0.1
GLOBAL_FACTOR = 0.05
SUBGRADIENT_FACTOR = 2
BETA_SUBG_FACTOR = 0.9
PERTURBATION_FACTOR = 0.5
SWARM_SIZE = 16
EPSILON = 0.01

# Constraints
MAX_SECONDS = 3600*6
MAXITER = 1000
LBC = 10

HEURISTIC_TYPE = "LVA"
HEURISTIC_TYPES = ["RAND", "LVA", "LVP"]

PARALLEL = True

# Lagrangian relaxed function: min λ>=0 LR(λ) = sum_((i,j) in E)(λ_ij) + |K| - sum_(k in K)(min{1, min_(s_k->t_k path P_k)(sum_(i,j) in P_k)(λ_ij)})

def LaPSO_MEDP(graph : nx.Graph, commodity_pairs : list[tuple[int, int]], max_seconds : int = MAX_SECONDS, max_iterations : int = MAXITER,
               velocity_factor : float = VELOCITY_FACTOR, global_factor : float = GLOBAL_FACTOR, subgradient_factor : float = SUBGRADIENT_FACTOR,
               perturbation_factor : float = PERTURBATION_FACTOR, beta_subg_factor : float = BETA_SUBG_FACTOR, swarm_size : int = SWARM_SIZE) -> tuple[int, dict]:
    
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
        "lower_bound" : float("inf"),
        "upper_bound" : -float("inf"),
        "iteration" : 0,
        "best_solution" : None,
        "best_path" : []
    }

    start_time = time.time()

    while time.time() - start_time < max_seconds and parameters["iteration"] < max_iterations and (parameters["iteration"] == 0 or abs((parameters["upper_bound"]-parameters["lower_bound"])/parameters["upper_bound"]) > EPSILON):
        particles = particle_swarm_step(
            graph=graph, 
            commodity_pairs=commodity_pairs, 
            particles=particles,
            parameters=parameters
        )
        parameters["iteration"] += 1
        print(f"Iteration {parameters['iteration']} - Best solution: {parameters['lower_bound']}-{parameters['upper_bound']}) - Time elapsed: {time.time() - start_time}")

    output = {}
    for commodity, (source, target) in enumerate(commodity_pairs):
        output[(source, target)] = parameters["best_path"][commodity]
    return (parameters["upper_bound"], output)


def process_particle(graph, commodity_pairs, particle, parameters):
    relaxed_base_graph = generate_base_relaxed_graph(particle)
    commodity_sum = 0

    # Initialize the subgradient to all 1s on existing edges
    subgradient = {edge: 1.0 for edge in graph.edges}
    paths = []

    for commodity, (source, target) in enumerate(commodity_pairs):
        if parameters["iteration"] % LBC == 0:
            relaxed_perturbed_graph = nx.Graph()
            relaxed_perturbed_graph.add_edges_from(relaxed_base_graph.edges(data=True))
        else:
            relaxed_perturbed_graph = generate_perturbed_graph(relaxed_base_graph, particle, commodity)
        try:
            length, path = nx.single_source_dijkstra(
                relaxed_perturbed_graph, 
                source, 
                target, 
                weight="weight", 
                cutoff=1.0
            )
        except nx.NetworkXNoPath:
            # No path found, rerouting to the alternative edge.
            length = 1.0
            path = []

        # No path shorter than 1 has been found, rerouting to the alternative edge.
        if length == float("inf"):
            length = 1.0
            path = []

        # Update the subgradient of constraint (6)
        for i in range(len(path) - 1):
            if (path[i], path[i+1]) in subgradient:
                subgradient[(path[i], path[i+1])] -= 1.0
            elif (path[i+1], path[i]) in subgradient:
                subgradient[(path[i+1], path[i])] -= 1.0

        commodity_sum += length
        paths.append(path)

    perturbation_offset = min(min(perturbation_list[commodity] for perturbation_list in particle["perturbation"].values()), 0)
    current_relaxed_solution = sum(particle["lambda"].values()) + (0 if parameters["iteration"] % LBC == 0 else sum(perturbation[commodity]-perturbation_offset for perturbation in particle["perturbation"].values())) + len(commodity_pairs) - commodity_sum
    particle["iterations_since_bound_update"] += 1

    if current_relaxed_solution < particle["lower_bound"]:
        particle["lower_bound"] = current_relaxed_solution
        particle["iterations_since_bound_update"] = 0
    elif particle["iterations_since_bound_update"] >= LBC:
        particle["subgradient_factor"] *= parameters["beta_subg_factor"]
        particle["iterations_since_bound_update"] = 0

    if current_relaxed_solution < parameters["lower_bound"]:
        parameters["lower_bound"] = current_relaxed_solution

    infractions = feasibility_check(paths)
    if sum(infractions) > 0:
        paths = repair_heuristic(graph, paths, infractions, particle["perturbation"])

    if sum(feasibility_check(paths)) == 0:
        connected_commodities = sum(1 for path in paths if len(path) > 0)
        if connected_commodities > parameters["upper_bound"]:
            parameters["upper_bound"] = connected_commodities
            parameters["best_particle"] = particle
            parameters["best_path"] = paths

    update_particle(particle, parameters, subgradient, paths)
    
    return particle, parameters

def particle_swarm_step(graph, commodity_pairs, particles, parameters, parallel=PARALLEL):
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_particle, graph, commodity_pairs, particle, parameters) for particle in particles]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Extract updated particles and parameters
        updated_particles = [result[0] for result in results]
        updated_parameters_list = [result[1] for result in results]
    else:
        updated_particles = []
        updated_parameters_list = []
        for particle in particles:
            updated_particle, updated_parameters = process_particle(graph, commodity_pairs, particle, parameters)
            updated_particles.append(updated_particle)
            updated_parameters_list.append(updated_parameters)

    # Aggregate changes to parameters
    for updated_parameters in updated_parameters_list:
        if updated_parameters["lower_bound"] < parameters["lower_bound"]:
            parameters["lower_bound"] = updated_parameters["lower_bound"]
        if updated_parameters["upper_bound"] > parameters["upper_bound"]:
            parameters["upper_bound"] = updated_parameters["upper_bound"]
            parameters["best_particle"] = updated_parameters["best_particle"]
            parameters["best_path"] = updated_parameters["best_path"]

    return updated_particles


def initialize_particles(graph : nx.Graph, n_commodities : int, swarm_size : int, subgradient_factor : float = SUBGRADIENT_FACTOR) -> list[dict]:
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

    for _ in range(swarm_size):

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
            "lower_bound": float("inf"),
            "iterations_since_bound_update": 0
        })

    return particles


def generate_base_relaxed_graph(particle : dict) -> nx.Graph:

    relaxed_base_graph = nx.Graph()

    for edge in particle["lambda"].keys():
        relaxed_base_graph.add_edge(edge[0], edge[1], weight=particle["lambda"][edge])

    return relaxed_base_graph


def generate_perturbed_graph(graph : nx.Graph, particle : dict, commodity_number : int) -> nx.Graph:

    perturbed_graph = nx.Graph()
    perturbed_graph.add_edges_from(graph.edges(data=True))

    perturbation_offset = min(min(perturbation_list[commodity_number] for perturbation_list in particle["perturbation"].values()), 0)

    for edge in particle["lambda"].keys():
        perturbed_graph[edge[0]][edge[1]]["weight"] += particle["perturbation"][edge][commodity_number] - perturbation_offset

    return perturbed_graph


def path_dict_creator(paths : list[list[int]]) -> dict[tuple[int, int], int]:
    '''
        Given a list of paths, creates a dictionary where the keys are the edges and the values are the number of times the edge is used in the paths.
    '''
    edge_dict = {}

    for path in paths:
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            if edge in edge_dict:
                edge_dict[edge] += 1
            else:
                edge_dict[edge] = 1
            edge_reversed = (path[i+1], path[i])
            if edge_reversed in edge_dict:
                edge_dict[edge_reversed] += 1
            else:
                edge_dict[edge_reversed] = 1

    return edge_dict


def feasibility_check(paths : list[list[int]]) -> list[int]:
    '''
        Given a list of paths, checks for the number of infractions (same edge used more than once) for each path.
        An infraction is defined as a pair of edges that are used more than once in the paths.

        E.g.:
        If the paths are [[0, 1, 2, 3], [1, 2, 4, 5], [4, 5, 6]], the resulting output will be [1, 2, 1] (the number of infractions for each path), 
        since paths 1 and 2 both use the edge (1, 2) and paths 2 and 3 both use the edge (4, 5).
    '''
    pairs_count = path_dict_creator(paths)

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
                elif (path[j+1], path[j]) in current_graph.edges:
                    current_graph.remove_edge(path[j+1], path[j])

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


def update_particle(particle : dict, parameters : dict, subgradient : dict, paths : list[list[int]]) -> dict:
    '''
        Updates the velocities of the particle's lambda values and perturbations.
    '''
    random_factor_l, random_factor_g = random.random(), random.random()

    path_dict = path_dict_creator(paths)
    
    for edge in particle["lambda"].keys():
        first_factor = parameters["velocity_factor"] * particle["lambda_velocity"][edge]
        second_factor = random_factor_l * particle["subgradient_factor"] * (parameters["upper_bound"] - parameters["lower_bound"]) * subgradient[edge] / float(np.linalg.norm(list(subgradient.values())))
        third_factor = random_factor_g * parameters["global_factor"] * (parameters["best_particle"]["lambda"][edge] - particle["lambda"][edge])
        
        particle["lambda_velocity"][edge] = first_factor + second_factor + third_factor
        particle["lambda"][edge] = max(0, particle["lambda"][edge] + particle["lambda_velocity"][edge])
        
        for i in range(len(particle["perturbation"][edge])):
            particle["perturbation_velocity"][edge][i] = (parameters["velocity_factor"] * particle["perturbation_velocity"][edge][i]) + \
                (random_factor_g * parameters["global_factor"] * (parameters["best_particle"]["perturbation"][edge][i] - particle["perturbation"][edge][i])) + \
                (parameters["perturbation_factor"] * random_factor_g * parameters["global_factor"] * (-1.0 if edge in path_dict and path_dict[edge] > 0 else 1.0))
            
            particle["perturbation"][edge][i] *= 0.5
            particle["perturbation"][edge][i] += particle["perturbation_velocity"][edge][i]

    return particle