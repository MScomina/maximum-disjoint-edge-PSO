import rustworkx as rw
import random
import time
import numpy as np
import concurrent.futures
import copy
from math import exp

# Hyperparameters
GLOBAL_FACTOR = 0.1
SUBGRADIENT_FACTOR = 2.0
BETA_SUBG_FACTOR = 0.9
PERTURBATION_FACTOR = 0.5

INITIAL_VELOCITY_FACTOR = 0.75
FINAL_VELOCITY_FACTOR = 0.1
VELOCITY_DECAY = 0.0025

PERTURBATION_DECAY = 0.5

SWARM_SIZE = 8
EPSILON = 0.01

# Constraints
MAX_SECONDS = 3600*8
MAX_ITERATIONS = 1000
LBC = 10

HEURISTIC_TYPE = "LVP"
HEURISTIC_TYPES = ["RAND", "LVA", "LVP"]

PARALLEL = True

# Lagrangian relaxed function: min λ>=0 LR(λ) = sum_((i,j) in E)(λ_ij) + |K| - sum_(k in K)(min{1, min_(s_k->t_k path P_k)(sum_(i,j) in P_k)(λ_ij)})
def LaPSO_MEDP(graph : rw.PyGraph, commodity_pairs : list[tuple[int, int]], max_seconds : int = MAX_SECONDS, max_iterations : int = MAX_ITERATIONS,
               initial_velocity_factor : float = INITIAL_VELOCITY_FACTOR, final_velocity_factor : float = FINAL_VELOCITY_FACTOR, 
               velocity_decay : float = VELOCITY_DECAY, global_factor : float = GLOBAL_FACTOR, subgradient_factor : float = SUBGRADIENT_FACTOR, 
               perturbation_factor : float = PERTURBATION_FACTOR, beta_subg_factor : float = BETA_SUBG_FACTOR, swarm_size : int = SWARM_SIZE) -> tuple[int, dict]:
    '''
        Solves the Minimum Edge Disjoint Paths problem using the LaPSO algorithm.

        This is an implementation of the algorithm presented in "Solving the maximum edge disjoint path problem 
        using a modified Lagrangian particle swarm optimisation hybrid." by Jake Weiner et al.
    '''

    particles = _initialize_particles(
        graph=graph, 
        n_commodities=len(commodity_pairs), 
        swarm_size=swarm_size,
        subgradient_factor=subgradient_factor
    )

    parameters = {
        "velocity_factor" : initial_velocity_factor,
        "initial_velocity_factor" : initial_velocity_factor,
        "final_velocity_factor" : final_velocity_factor,
        "velocity_decay" : velocity_decay,
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

    while time.time() - start_time < max_seconds and parameters["iteration"] < max_iterations and (parameters["iteration"] == 0 or abs((parameters["upper_bound"]-int(parameters["lower_bound"]))/parameters["upper_bound"]) > EPSILON):
        particles = _particle_swarm_step(
            graph=graph, 
            commodity_pairs=commodity_pairs, 
            particles=particles,
            parameters=parameters
        )
        parameters["iteration"] += 1
        print(f"Iteration {parameters['iteration']} - Best solution: {parameters['upper_bound']}-{int(parameters['lower_bound'])}) - Time elapsed: {time.time() - start_time}")

    output = {}
    for commodity, (source, target) in enumerate(commodity_pairs):
        output[(source, target)] = parameters["best_path"][commodity]
    return (parameters["upper_bound"], output)


def _process_particle(graph : rw.PyGraph, commodity_pairs, particle, parameters):
    relaxed_base_graph = _generate_base_relaxed_graph(particle)
    commodity_sum = 0

    # Initialize the subgradient to all 1s on existing edges
    subgradient = {edge: 1.0 for edge in graph.edge_list()}
    paths = []

    for commodity, (source, target) in enumerate(commodity_pairs):
        if parameters["iteration"] % LBC == 0:
            _update_graph_weights(relaxed_base_graph, particle, commodity, perturbated=False)
        else:
            _update_graph_weights(relaxed_base_graph, particle, commodity, perturbated=True)
        try:
            path = rw.dijkstra_shortest_paths(
                relaxed_base_graph, 
                source, 
                target, 
                weight_fn=lambda edge: edge["weight"]
            )[target]
            length = sum(relaxed_base_graph.get_edge_data(edge[0], edge[1])["weight"] for edge in zip(path[:-1], path[1:]))
            if length > 1.0:
                length = 1.0
                path = []
        except rw.NoPathFound:
            # No path found, rerouting to the alternative edge.
            length = 1.0
            path = []

        # No path shorter than 1 found, rerouting to the alternative edge.
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
    lambda_sum = sum(particle["lambda"].values())
    perturbation_sum = 0 if parameters["iteration"] % LBC == 0 else sum(perturbation[commodity] - perturbation_offset for perturbation in particle["perturbation"].values())
    current_relaxed_solution = lambda_sum + perturbation_sum + len(commodity_pairs) - commodity_sum

    particle["iterations_since_bound_update"] += 1

    if current_relaxed_solution < particle["lower_bound"]:
        particle["lower_bound"] = current_relaxed_solution
        particle["iterations_since_bound_update"] = 0
    elif particle["iterations_since_bound_update"] >= LBC:
        particle["subgradient_factor"] *= parameters["beta_subg_factor"]
        particle["iterations_since_bound_update"] = 0

    if current_relaxed_solution < parameters["lower_bound"]:
        parameters["lower_bound"] = current_relaxed_solution

    infractions = _feasibility_check(paths)
    if sum(infractions) > 0:
        paths = _repair_heuristic(graph, paths, infractions, particle["perturbation"])

    if sum(_feasibility_check(paths)) == 0:
        connected_commodities = sum(1 for path in paths if len(path) > 0)
        if connected_commodities > parameters["upper_bound"]:
            particle["upper_bound"] = connected_commodities
        if connected_commodities > parameters["upper_bound"]:
            parameters["upper_bound"] = connected_commodities
            parameters["best_particle"] = copy.deepcopy(particle)
            parameters["best_path"] = copy.deepcopy(paths)

    _update_particle(particle, parameters, subgradient, paths)
    
    return particle, parameters

def _particle_swarm_step(graph : rw.PyGraph, commodity_pairs : list[tuple[int, int]], particles : dict, parameters : dict, parallel : bool = PARALLEL):

    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(_process_particle, graph, commodity_pairs, particle, parameters) for particle in particles]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        updated_particles = [result[0] for result in results]
        updated_parameters_list = [result[1] for result in results]
    else:
        updated_particles = []
        updated_parameters_list = []
        for particle in particles:
            updated_particle, updated_parameters = _process_particle(graph, commodity_pairs, particle, parameters)
            updated_particles.append(updated_particle)
            updated_parameters_list.append(updated_parameters)

    for updated_parameters in updated_parameters_list:
        if updated_parameters["lower_bound"] < parameters["lower_bound"]:
            parameters["lower_bound"] = updated_parameters["lower_bound"]
        if updated_parameters["upper_bound"] > parameters["upper_bound"]:
            parameters["upper_bound"] = updated_parameters["upper_bound"]
            parameters["best_particle"] = updated_parameters["best_particle"]
            parameters["best_path"] = updated_parameters["best_path"]

    parameters["velocity_factor"] = _current_velocity_factor(
        parameters["iteration"], 
        parameters["initial_velocity_factor"],
        parameters["final_velocity_factor"], 
        parameters["velocity_decay"]
        )

    return updated_particles


def _initialize_particles(graph : rw.PyGraph, n_commodities : int, swarm_size : int, subgradient_factor : float = SUBGRADIENT_FACTOR) -> list[dict]:
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

        for edge in _edge_generator(graph):
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
            "upper_bound": -float("inf"),
            "iterations_since_bound_update": 0
        })

    return particles


def _generate_base_relaxed_graph(particle : dict) -> rw.PyGraph:
    '''
        Generates the base relaxed graph for the particle, where the edge weights are the lambda values of the particle.
    '''
    relaxed_base_graph = rw.PyGraph()
    for edge in _edge_generator(particle):
        relaxed_base_graph.extend_from_weighted_edge_list([(edge[0], edge[1], {"weight" : particle["lambda"][edge]})])

    return relaxed_base_graph


def _update_graph_weights(graph : rw.PyGraph, particle : dict, commodity_number : int, perturbated : bool = True) -> None:
    '''
        Updates the graph weights in-place according to the lambda values of the particle.
    '''
    if perturbated:
        perturbation_offset = min(min(perturbation_list[commodity_number] for perturbation_list in particle["perturbation"].values()), 0)
        for edge in _edge_generator(graph):
            if edge in particle["lambda"]:
                graph.update_edge(edge[0], edge[1], {"weight": particle["lambda"][edge] + particle["perturbation"][edge][commodity_number] - perturbation_offset})
            else:
                graph.update_edge(edge[0], edge[1], {"weight": particle["lambda"][(edge[1],edge[0])] + particle["perturbation"][(edge[1],edge[0])][commodity_number] - perturbation_offset})
    else:
        for edge in _edge_generator(graph):
            if edge in particle["lambda"]:
                graph.update_edge(edge[0], edge[1], {"weight": particle["lambda"][edge]})
            else:
                graph.update_edge(edge[0], edge[1], {"weight": particle["lambda"][(edge[1],edge[0])]})

def _path_dict_creator(paths : list[list[int]]) -> dict[tuple[int, int], int]:
    '''
        Given a list of paths, creates a dictionary where the keys are the edges and the values are the number of times the edge is used in the paths.
        
        Note: The resulting dictionary is symmetric (i.e. if (u, v) is in the dictionary, then (v, u) is also in the dictionary and has the same value).
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


def _feasibility_check(paths : list[list[int]]) -> list[int]:
    '''
        Given a list of paths, checks for the number of infractions (same edge used more than once) for each path.
        An infraction is defined as a pair of edges that are used more than once in the paths.

        E.g.:
        If the paths are [[0, 1, 2, 3], [1, 2, 4, 5], [4, 5, 6]], the resulting output will be [1, 2, 1] (the number of infractions for each path), 
        since paths 1 and 2 both use the edge (1, 2) and paths 2 and 3 both use the edge (4, 5).
    '''
    pairs_count = _path_dict_creator(paths)

    infractions = [0 for _ in range(len(paths))]

    for i, path in enumerate(paths):
        for j in range(len(path) - 1):
            edge = (path[j], path[j+1])
            if pairs_count[edge] > 1:
                infractions[i] += 1

    return infractions


def _repair_heuristic(graph : rw.PyGraph, paths : list[list[int]], infractions : list[int], perturbations : dict[tuple[int, int], list[float]] = None,
                     heuristic_type : str = HEURISTIC_TYPE) -> list[list[int]]:
    '''
        Given a list of paths and the number of infractions for each path, repairs the paths by removing the infractions.

        The repair is done using a heuristic, which can be one of the following:
        - RAND: Randomly picks a commodity and tries to repair the path.
        - LVA: Largest Violation Arbitrary, picks the commodity with the largest number of infractions and tries to repair it.
        - LVP: Largest Violation Perturbation, picks the commodity with the largest number of infractions and tries to repair it using the perturbations as graph weights.
    
        All of the repairs are attempted using Dijkstra's algorithm to find the shortest path between the source and target nodes.
    '''

    if heuristic_type not in HEURISTIC_TYPES:
        raise ValueError(f"Invalid heuristic type. Must be one of {HEURISTIC_TYPES}.")
    
    if heuristic_type == "LVP" and perturbations is None:
        raise ValueError("Perturbations must be provided when using the LVP heuristic.")

    repaired_paths = [path[:] for path in paths]

    '''
        Create a stripped graph with the edges that are not in the repaired paths.
        The idea is that using this graph and adding the edges back each time the 
        computation is faster than just copying the graph over and over for big graphs.
    '''
    current_path = []
    stripped_graph = rw.PyGraph()
    stripped_graph.extend_from_weighted_edge_list([(u, v, {"weight": 1.0}) for u, v in graph.edge_list()])

    edge_set = set(graph.edge_list())
    original_edge_set = edge_set.copy()

    for path in repaired_paths:
        for u, v in zip(path[:-1], path[1:]):
            if (u, v) in edge_set:
                stripped_graph.remove_edge(u, v)
                edge_set.remove((u, v))
            elif (v, u) in edge_set:
                stripped_graph.remove_edge(v, u)
                edge_set.remove((v, u))
    
    # Generate the list of indices of commodities with infractions, sorted by the number of infractions
    indices_with_infractions = [i for i, infraction in enumerate(infractions) if infraction > 0]
    if len(indices_with_infractions) == 0:
        return repaired_paths
    indices_with_infractions.sort(key=lambda x: infractions[x], reverse=True)
    
    while len(indices_with_infractions) > 0:
        if heuristic_type == "RAND":
            picked_commodity_idx = indices_with_infractions[random.randint(0, len(indices_with_infractions) - 1)]
        else:
            picked_commodity_idx = indices_with_infractions[0]

        current_path = repaired_paths[picked_commodity_idx]
        pairs_count = _path_dict_creator(repaired_paths)

        for i in range(len(current_path) - 1):
            edge = (current_path[i], current_path[i+1])
            if pairs_count[edge] == 1:
                if edge in original_edge_set:
                    stripped_graph.add_edge(edge[0], edge[1], {"weight": 1.0})
                    edge_set.add(edge)
                elif (edge[1], edge[0]) in original_edge_set:
                    stripped_graph.add_edge(edge[1], edge[0], {"weight": 1.0})
                    edge_set.add((edge[1], edge[0]))

        if heuristic_type == "LVP":
            perturbation_offset = min(min(perturbation_list[picked_commodity_idx] for perturbation_list in perturbations.values()), 0)
            for edge in _edge_generator(stripped_graph):
                if edge in perturbations:
                    stripped_graph.update_edge(edge[0], edge[1], {"weight": perturbations[edge][picked_commodity_idx] - perturbation_offset})
                else:
                    stripped_graph.update_edge(edge[0], edge[1], {"weight": perturbations[(edge[1], edge[0])][picked_commodity_idx] - perturbation_offset})

        try:
            if heuristic_type == "LVP":
                shortest_path = rw.dijkstra_shortest_paths(
                    stripped_graph,
                    repaired_paths[picked_commodity_idx][0],
                    repaired_paths[picked_commodity_idx][-1],
                    weight_fn=lambda edge: edge["weight"]
                )
                if repaired_paths[picked_commodity_idx][-1] not in shortest_path:
                    shortest_path = []
                else:
                    shortest_path = list(shortest_path[repaired_paths[picked_commodity_idx][-1]])
            else:
                shortest_path = rw.dijkstra_shortest_paths(
                    stripped_graph,
                    repaired_paths[picked_commodity_idx][0],
                    repaired_paths[picked_commodity_idx][-1],
                    default_weight=1.0
                )
                if repaired_paths[picked_commodity_idx][-1] not in shortest_path:
                    shortest_path = []
                else:
                    shortest_path = list(shortest_path[repaired_paths[picked_commodity_idx][-1]])
            repaired_paths[picked_commodity_idx] = shortest_path
            indices_with_infractions.remove(picked_commodity_idx)
        except rw.NoPathFound:
            repaired_paths[picked_commodity_idx] = []
            indices_with_infractions.remove(picked_commodity_idx)

        for i in range(len(repaired_paths[picked_commodity_idx]) - 1):
            edge = (repaired_paths[picked_commodity_idx][i], repaired_paths[picked_commodity_idx][i+1])
            if edge in edge_set:
                stripped_graph.remove_edge(edge[0], edge[1])
                edge_set.remove(edge)
            elif (edge[1], edge[0]) in edge_set:
                stripped_graph.remove_edge(edge[1], edge[0])
                edge_set.remove((edge[1], edge[0]))

    return repaired_paths

def _edge_generator(source : dict | rw.PyGraph):
    '''
        Lazy generator function that yields the edges of a graph, given either a dictionary or a RustworkX PyGraph.
    '''
    if isinstance(source, rw.PyGraph):
        for edge in source.edge_list():
            yield edge
    else:
        for edge in source["lambda"].keys():
            yield edge

def _current_velocity_factor(iteration : int, start_velocity_factor : float = INITIAL_VELOCITY_FACTOR, 
                             end_velocity_factor : float = FINAL_VELOCITY_FACTOR, decay_rate : float = VELOCITY_DECAY) -> float:
    '''
        Calculates the current velocity/inertia factor for the PSO algorithm, given the current iteration.
    '''
    return end_velocity_factor + (start_velocity_factor - end_velocity_factor) * exp(-decay_rate * iteration)

def _update_particle(particle: dict, parameters: dict, subgradient: dict, paths: list[list[int]]) -> dict:
    '''
        Updates the velocities of the particle's lambda values and perturbations, as well as the lambda values themselves according to the PSO algorithm.
    '''
    random_factor_l, random_factor_g = random.random(), random.random()
    random_factor_p = random.random()

    path_dict = _path_dict_creator(paths)
    subgradient_norm = float(np.linalg.norm(list(subgradient.values())))

    for edge in _edge_generator(particle):
        lambda_velocity = particle["lambda_velocity"][edge]
        lambda_value = particle["lambda"][edge]
        best_lambda_value = parameters["best_particle"]["lambda"][edge]
        subgrad_value = subgradient[edge]

        # Equation (13)
        first_factor = parameters["velocity_factor"] * lambda_velocity
        second_factor = random_factor_l * particle["subgradient_factor"] * (parameters["upper_bound"] - parameters["lower_bound"]) * subgrad_value / subgradient_norm
        third_factor = random_factor_g * parameters["global_factor"] * (best_lambda_value - lambda_value)
        
        new_lambda_velocity = first_factor + second_factor + third_factor
        particle["lambda_velocity"][edge] = new_lambda_velocity

        # Equation (15)
        particle["lambda"][edge] = max(0, lambda_value + new_lambda_velocity)
        
        perturbation_velocity = particle["perturbation_velocity"][edge]
        perturbation = particle["perturbation"][edge]
        best_perturbation = parameters["best_particle"]["perturbation"][edge]

        for i in range(len(perturbation)):
            # Equation (14)
            perturbation_velocity[i] = (parameters["velocity_factor"] * perturbation_velocity[i]) + \
                (random_factor_p * parameters["global_factor"] * (best_perturbation[i] - perturbation[i])) + \
                (parameters["perturbation_factor"] * random_factor_p * parameters["global_factor"] * (-1.0 if edge in path_dict and path_dict[edge] > 0 else 1.0))
            
            # Equation (16)
            perturbation[i] = PERTURBATION_DECAY * perturbation[i] + perturbation_velocity[i]

    return particle