import gurobipy as gp
from networkx import Graph

def generate_gurobi_model(graph : Graph, commodity_pairs : list[tuple[int, int]], name : str = "MEDP", verbose : bool = True) -> gp.Model:
    '''
        Generates a Gurobi model for the MEDP given the graph and the commodity pairs, where commodity pairs are tuples of the form (source_node, target_node).
        Note: This is the most de-facto implementation of the model, as it directly follows the paper's formulation.
        The issue is that it generates a lot of redundant variables, especially for big graphs. 
        They usually get trimmed by Gurobi's presolve, but the program may still go out of memory before that.
    '''

    n_nodes = graph.number_of_nodes()
    n_commodities = len(commodity_pairs)
    graph_edges = set(graph.edges())


    model = gp.Model(name)

    if verbose:
        model.setParam("OutputFlag", 1)
    else:
        model.setParam("OutputFlag", 0)

    '''
        Constraint number (7): Decision variables are binary.
    '''
    x = model.addVars(n_nodes, n_nodes, n_commodities, vtype=gp.GRB.BINARY, name="x")

    for k, (start, end) in enumerate(commodity_pairs):
        '''
            Constraint number (4): Flow through nodes must be preserved, unless the node is a source or a destination node. 
        '''
        for i in range(n_nodes):
            if i == start or i == end:
                continue
            model.addConstr(gp.quicksum(x[i, j, k] for j in range(n_nodes)) - gp.quicksum(x[j, i, k] for j in range(n_nodes)) == 0, name="node_flow")

        '''
            Constraint number (5): Every source node has at max one exiting flow.

            Note: The constraint on the paper only required to check "outgoing flows", but that ends up making the solution loop back to the source node,
            since there's no discernable difference between source and destination nodes in the formulation when it comes to the ending node.
            Forcing the undirected graph constraint x_ijk - x_jik = 0 doesn't work when combined with constraint (6),
            since every edge would count as two, collapsing the solution to 0.

            Therefore, the constraint is modified to check both incoming and outgoing flows (instead of sum(x_{s_k}jk) it's sum(x_{s_k}jk + x_j{s_k}k)).
        '''
        model.addConstr(gp.quicksum((x[start, j, k]+x[j, start, k]) for j in range(n_nodes)) <= 1, name="source_flow")

    for i in range(n_nodes):
        for j in range(n_nodes):
            '''
                Constraint number (6): Every edge of the graph can only be used at most by a single commodity in either direction.
            '''
            if i < j:
                model.addConstr(gp.quicksum((x[i, j, k] + x[j, i, k]) for k in range(n_commodities)) <= 1, name="disjoint_edges")

            '''
                Constraint number (7): Every edge pair must be in the given graph.
            '''
            if (i, j) not in graph_edges and (j, i) not in graph_edges:
                model.addConstr(gp.quicksum(x[i, j, k] for k in range(n_commodities)) == 0, name="edge_existance")

    '''
        Set target function to optimize (3).
    '''
    model.setObjective(gp.quicksum(x.sum(start, "*", k) for k, (start, _) in enumerate(commodity_pairs)), gp.GRB.MAXIMIZE)

    '''
        Set Gurobi parameters.
    '''
    model.setParam(gp.GRB.Param.TimeLimit, 3600*8)
    model.setParam(gp.GRB.Param.Presolve, 2)

    return model

def generate_gurobi_model_efficient(graph : Graph, commodity_pairs : list[tuple[int, int]], name : str = "MEDP", verbose : bool = True) -> gp.Model:
    '''
        Generates a Gurobi model for the MEDP given the graph and the commodity pairs, where commodity pairs are tuples of the form (source_node, target_node).
        Note: This function is different from generate_gurobi_model because it directly implements only the variables that are actually part of the graph.
        It is, therefore, much more lenient on RAM usage.
    '''

    n_nodes = graph.number_of_nodes()
    n_commodities = len(commodity_pairs)

    graph = graph.to_undirected()
    adjacency_list = {int(node): {int(neighbor) for neighbor in neighbors} for node, neighbors in graph.adj.items()}

    model = gp.Model(name)

    '''
        Constraint number (7): Decision variables are binary, and are part of the graph.
        Note: This is the main difference between this function and the previous one. The variables are only created if they are part of the graph.
    '''
    x = {}
    for i in range(n_nodes):
        for j in adjacency_list[i]:
            for k in range(n_commodities):
                x[i, j, k] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{i}_{j}_{k}")
                x[j, i, k] = model.addVar(vtype=gp.GRB.BINARY, name=f"x_{j}_{i}_{k}")

    model.update()

    for k, (start, end) in enumerate(commodity_pairs):
        '''
            Constraint number (4): Flow through nodes must be preserved, unless the node is a source or a destination node. 
        '''
        for i in range(n_nodes):
            if i == start or i == end:
                continue
            model.addConstr(
                (gp.quicksum(x[i, j, k] for j in adjacency_list[i]) -
                gp.quicksum(x[j, i, k] for j in adjacency_list[i])) == 0,
                name=f"node_flow_{i}_{k}"
            )

        '''
            Constraint number (5): Every source node has at max one exiting flow.

            Note: The constraint on the paper only required to check "outgoing flows", but that ends up making the solution loop back to the source node,
            since there's no discernable difference between source and destination nodes in the formulation when it comes to the ending node.
            Forcing the undirected graph constraint x_ijk - x_jik = 0 doesn't work when combined with constraint (6),
            since every edge would count as two, collapsing the solution to 0.

            Therefore, the constraint is modified to check both incoming and outgoing flows (instead of sum(x_{s_k}jk) it's sum(x_{s_k}jk + x_j{s_k}k)).
        '''
        model.addConstr(
            gp.quicksum((x[start, j, k]+x[j, start, k]) for j in adjacency_list[start]) <= 1,
            name=f"source_flow_{start}_{k}"
        )

    for i in range(n_nodes):
        for j in adjacency_list[i]:
                '''
                    Constraint number (6): Every edge of the graph can only be used at most by a single commodity in either direction.
                '''
                model.addConstr(
                    gp.quicksum((x[i, j, k] + x[j, i, k]) for k in range(n_commodities)) <= 1,
                    name=f"disjoint_edges_{i}_{j}"
                )

    '''
        Set target function to optimize (3).
    '''
    model.setObjective(
        gp.quicksum((x[start, j, k]) for k, (start, _) in enumerate(commodity_pairs) for j in adjacency_list[start]),
        gp.GRB.MAXIMIZE
    )

    '''
        Set Gurobi parameters.
    '''
    #model.setParam(gp.GRB.Param.NodefileStart, 0.5)
    model.setParam(gp.GRB.Param.TimeLimit, 3600*8)
    #model.setParam(gp.GRB.Param.Threads, 4)
    model.setParam(gp.GRB.Param.Presolve, 2)
    if verbose:
        model.setParam("OutputFlag", 1)
    else:
        model.setParam("OutputFlag", 0)

    return model