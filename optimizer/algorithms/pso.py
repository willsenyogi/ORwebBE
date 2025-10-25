import time
import random
import gurobipy as gp
from gurobipy import GRB
from .utils import DistanceMatrix, HeuristicFunctions
import numpy as np

heuristic = HeuristicFunctions()

# -----------------------------
# PSO implementation
# -----------------------------
class ParticleSwarmOptimization:
    def __init__(self, depots, customers, vehicles_per_depot, vehicle_capacities, customer_demands, pop_size=50, w=0.8, c1=1.5, c2=1.5, iters=200, seed=None):
        # Inisialisasi data problem
        self.depots, self.customers, self.vehicles_per_depot, self.vehicle_capacities, self.customer_demands = depots, customers, vehicles_per_depot, vehicle_capacities, np.array(customer_demands)
        self.m, self.n = len(depots), len(customers)
        self.total_vehicles = sum(self.vehicles_per_depot)
        self.vehicle_depot_map = [idx for idx, num in enumerate(self.vehicles_per_depot) for _ in range(num)]
        
        # Hyperparameter PSO: pop_size, inertia weight w, cognitive c1, social c2, jumlah iterasi
        self.pop_size, self.w, self.c1, self.c2, self.iters = pop_size, w, c1, c2, iters
        self.v_max = 0.5  # batas max velocity
        self.v_min = -0.5 # batas min velocity
        
        if seed is not None: np.random.seed(seed); random.seed(seed)
        
        # Matriks jarak antar customer
        self.customer_distance_matrix = DistanceMatrix.build_distance_matrix(customers)
        
        # Inisialisasi posisi dan velocity partikel secara random
        self.positions = np.random.uniform(-1, 1, size=(pop_size, self.n, self.total_vehicles))
        self.velocities = np.random.uniform(-0.1, 0.1, size=(pop_size, self.n, self.total_vehicles))
        
        # Simpan personal best (pbest) tiap partikel
        self.pbest_positions = self.positions.copy()
        self.pbest_assignments = [self._positions_to_assignment(p) for p in self.positions] # mapping posisi ke vehicle assignment
        self.pbest_values = [self._eval_assignment(a) for a in self.pbest_assignments]
        
        # Tentukan global best (gbest)
        best_idx = int(np.argmin(self.pbest_values))
        self.gbest_position, self.gbest_assignment, self.gbest_value = self.pbest_positions[best_idx].copy(), self.pbest_assignments[best_idx].copy(), self.pbest_values[best_idx]
        
    # Konversi posisi partikel ke assignment kendaraan
    def _positions_to_assignment(self, pos_matrix):
        return [int(np.argmax(pos_matrix[i])) for i in range(self.n)]  # tiap customer di-assign ke kendaraan dengan nilai terbesar
        
    # Evaluasi fitness assignment
    def _eval_assignment(self, assignment):
        vehicle_groups = {v_idx: [] for v_idx in range(self.total_vehicles)}
        for cust_idx, vehicle_idx in enumerate(assignment):
            vehicle_groups[vehicle_idx].append(cust_idx)
        
        total_distance, total_penalty = 0.0, 0.0
        for vehicle_idx, assigned_cust_indices in vehicle_groups.items():
            if not assigned_cust_indices: continue
            current_demand = sum(self.customer_demands[c_idx] for c_idx in assigned_cust_indices)
            
            # Penalti jika kendaraan kelebihan kapasitas
            if current_demand > self.vehicle_capacities[vehicle_idx]:
                total_penalty += 999999 + (current_demand - self.vehicle_capacities[vehicle_idx]) * 1000

            # Hitung jarak rute kendaraan
            home_depot_idx = self.vehicle_depot_map[vehicle_idx]
            home_depot_coords = self.depots[home_depot_idx]
            _, route_dist = heuristic.build_route_and_length(home_depot_coords, assigned_cust_indices, self.customers, self.customer_distance_matrix)
            total_distance += route_dist
        return total_distance + total_penalty
    
    # Main PSO loop
    def run(self, verbose=False):
        start = time.time()
        history = [self.gbest_value]  # catatan perkembangan global best
        
        for it in range(self.iters):
            for p in range(self.pop_size):
                # Random untuk faktor kognitif & sosial
                r1, r2 = np.random.rand(self.n, self.total_vehicles), np.random.rand(self.n, self.total_vehicles)
                
                # Komponen velocity: cognitive (ke lbest) + social (ke gbest)
                cognitive = self.c1 * r1 * (self.pbest_positions[p] - self.positions[p])
                social = self.c2 * r2 * (self.gbest_position - self.positions[p])
                
                # Update velocity dan posisi partikel
                self.velocities[p] = self.w * self.velocities[p] + cognitive + social
                self.positions[p] += self.velocities[p]
                
                # Evaluasi posisi baru
                assignment = self._positions_to_assignment(self.positions[p])
                val = self._eval_assignment(assignment)
                
                # Update personal best
                if val < self.pbest_values[p]:
                    self.pbest_values[p], self.pbest_positions[p], self.pbest_assignments[p] = val, self.positions[p].copy(), assignment
                    # Update global best
                    if val < self.gbest_value:
                        self.gbest_value, self.gbest_position, self.gbest_assignment = val, self.positions[p].copy(), assignment.copy()
            
            history.append(self.gbest_value)
            if verbose:
                print(f"[PSO] iter {it}/{self.iters} best={self.gbest_value:.4f}")
        
        return {"best_assignment": self.gbest_assignment, "best_value": self.gbest_value, "time": time.time() - start, "history": history}
