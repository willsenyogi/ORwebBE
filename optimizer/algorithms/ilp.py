import time
import gurobipy as gp
from gurobipy import GRB
from utils import DistanceMatrix
# ------------------------------------
# ILP implementation using Gurobi
# ------------------------------------
class SolverILP:
    def __init__(self, depots, customers, vehicles_per_depot, vehicle_capacities, customer_demands, time_limit=60):
        self.depots, self.customers, self.vehicles_per_depot, self.vehicle_capacities, self.customer_demands = depots, customers, vehicles_per_depot, vehicle_capacities, customer_demands
        self.m, self.n = len(depots), len(customers)
        self.time_limit = time_limit
        self.total_vehicles = sum(self.vehicles_per_depot)
        self.vehicle_depot_map = [idx for idx, num in enumerate(self.vehicles_per_depot) for _ in range(num)]

    def run(self, verbose=False):
        start = time.time()
        all_points = self.depots + self.customers; num_points = len(all_points)
        dist_matrix = DistanceMatrix.build_distance_matrix(all_points)
        full_demands = [0] * self.m + self.customer_demands
        depot_indices, customer_indices, vehicle_indices = list(range(self.m)), list(range(self.m, num_points)), list(range(self.total_vehicles))
        model = gp.Model("MDVRP_ILP_Dynamic_Capacity")
        if not verbose: model.setParam('OutputFlag', 0)
        x = model.addVars(num_points, num_points, self.total_vehicles, vtype=GRB.BINARY, name="x")
        u = model.addVars(num_points, self.total_vehicles, vtype=GRB.CONTINUOUS, name="u")
        model.setObjective(gp.quicksum(dist_matrix[i, j] * x[i, j, k] for i in range(num_points) for j in range(num_points) for k in vehicle_indices if i != j), GRB.MINIMIZE)
        for j in customer_indices: model.addConstr(gp.quicksum(x[i, j, k] for i in range(num_points) for k in vehicle_indices if i != j) == 1)
        for k in vehicle_indices: model.addConstr(gp.quicksum(x[self.vehicle_depot_map[k], j, k] for j in customer_indices) <= 1)
        for k in vehicle_indices:
            for j in customer_indices: model.addConstr(gp.quicksum(x[i, j, k] for i in range(num_points) if i != j) == gp.quicksum(x[j, i, k] for i in range(num_points) if i != j))
        for k in vehicle_indices: model.addConstr(gp.quicksum(x[self.vehicle_depot_map[k], j, k] for j in customer_indices) == gp.quicksum(x[j, self.vehicle_depot_map[k], k] for j in customer_indices))
        for k in vehicle_indices:
            for i in range(num_points):
                for j in customer_indices:
                    if i != j: model.addConstr((x[i, j, k] == 1) >> (u[i, k] + full_demands[j] == u[j, k]))
        for k in vehicle_indices:
            for i in customer_indices:
                model.addConstr(u[i, k] >= full_demands[i])
                model.addConstr(u[i, k] <= self.vehicle_capacities[k])
        
        # model.setParam('NodefileStart', 0.7)
        # model.setParam('TimeLimit', self.time_limit)
        model.Params.NodefileStart = 0.5
        model.Params.NodefileDir = "/tmp"
        model.Params.Cuts = 1
        model.Params.Presolve = 2
        model.Params.Aggregate = 2
        model.Params.PreCrush = 1
        model.Params.MIPGap = 0.0
        model.optimize()
        end = time.time()
        best_assignment = [-1] * self.n
        best_value = model.ObjVal if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else float('inf')
        if best_value < float('inf'):
            solution = model.getAttr('x', x)
            for i in range(num_points):
                for j in customer_indices:
                    for k in vehicle_indices:
                        if solution[i, j, k] > 0.5:
                            customer_idx_in_list = j - self.m
                            best_assignment[customer_idx_in_list] = k
        return {"best_assignment": best_assignment, "best_value": best_value, "time": end - start, "history": []}