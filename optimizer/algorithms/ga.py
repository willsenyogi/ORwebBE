import time
import random
import copy
import gurobipy as gp
from gurobipy import GRB
from .utils import DistanceMatrix, HeuristicFunctions
import numpy as np

heuristic = HeuristicFunctions()

# -----------------------------
# Genetic Algorithm implementation
# -----------------------------
class GeneticAlgorithm:
    def __init__(self, depots, customers, vehicles_per_depot, vehicle_capacities, customer_demands, pop_size=100, cx_prob=0.8, mut_prob=0.2, iters=200, seed=None):
        # Data input: depot locations, customer locations, jumlah kendaraan per depot, kapasitas kendaraan, demand tiap customer
        self.depots, self.customers, self.vehicles_per_depot, self.vehicle_capacities, self.customer_demands = depots, customers, vehicles_per_depot, vehicle_capacities, np.array(customer_demands)
        self.m, self.n = len(depots), len(customers)
        self.total_vehicles = sum(self.vehicles_per_depot)  # total kendaraan dari semua depot
        # Mapping kendaraan ke depot, misal kendaraan ke-0 dari depot 0, dst
        self.vehicle_depot_map = [idx for idx, num in enumerate(self.vehicles_per_depot) for _ in range(num)]
        # Parameter GA: ukuran populasi, probabilitas crossover, probabilitas mutasi, jumlah iterasi
        self.pop_size, self.cx_prob, self.mut_prob, self.iters = pop_size, cx_prob, mut_prob, iters
        if seed is not None: random.seed(seed); np.random.seed(seed)
        # Precompute distance matrix antar customer
        self.customer_distance_matrix = DistanceMatrix.build_distance_matrix(customers)
        # Inisialisasi populasi random
        self.pop = [self._random_individual() for _ in range(self.pop_size)]
        # Evaluasi fitness awal tiap individu
        self.fitness_cache = [self._eval(ind) for ind in self.pop]
        
    # Generate solusi random: assign tiap customer ke kendaraan acak
    def _random_individual(self): 
        return [random.randrange(self.total_vehicles) for _ in range(self.n)]

    # Evaluasi fitness: total distance + penalty jika kendaraan overload
    def _eval(self, individual):
        vehicle_groups = {v_idx: [] for v_idx in range(self.total_vehicles)}
        for cust_idx, vehicle_idx in enumerate(individual):
            vehicle_groups[vehicle_idx].append(cust_idx)
        
        total_distance, total_penalty = 0.0, 0.0
        for vehicle_idx, assigned_cust_indices in vehicle_groups.items():
            if not assigned_cust_indices: continue
            current_demand = sum(self.customer_demands[c_idx] for c_idx in assigned_cust_indices)

            # Tambahin penalty besar kalau kapasitas kendaraan terlampaui
            if current_demand > self.vehicle_capacities[vehicle_idx]:
                total_penalty += 999999 + (current_demand - self.vehicle_capacities[vehicle_idx]) * 1000
            
            # Hitung jarak rute kendaraan dari depot ke semua customer yg ditugaskan
            home_depot_idx = self.vehicle_depot_map[vehicle_idx]
            home_depot_coords = self.depots[home_depot_idx]
            _, route_dist = heuristic.build_route_and_length(home_depot_coords, assigned_cust_indices, self.customers, self.customer_distance_matrix)
            total_distance += route_dist
        # Fitness = total distance + penalty
        return total_distance + total_penalty
        
    # Selection: tournament selection (pilih terbaik dari k kandidat random)
    def _tournament_select(self, k=3):
        best = None
        for _ in range(k):
            i = random.randrange(self.pop_size)
            if best is None or self.fitness_cache[i] < self.fitness_cache[best]:
                best = i
        return self.pop[best]

    # Crossover: ordered crossover
    def _crossover(self, p1, p2):
        size = len(p1)
        cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))
        # Tukar potongan tengah antar parent
        c1 = p1[:cxpoint1] + p2[cxpoint1:cxpoint2] + p1[cxpoint2:]
        c2 = p2[:cxpoint1] + p1[cxpoint1:cxpoint2] + p2[cxpoint2:]
        return c1, c2

    # Mutasi (tournament selection): tiap customer bisa di-assign ulang ke kendaraan lain secara random
    def _mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mut_prob:
                individual[i] = random.randrange(self.total_vehicles)

    # Main GA loop
    def run(self, verbose=False):
        start = time.time()
        best_val = min(self.fitness_cache)
        best_ind = self.pop[int(np.argmin(self.fitness_cache))]
        history = [best_val]
        
        for it in range(self.iters):
            new_pop = [copy.deepcopy(best_ind)]  # bawa best individu ke generasi baru
            while len(new_pop) < self.pop_size:
                p1, p2 = self._tournament_select(), self._tournament_select()
                # Crossover 
                if random.random() < self.cx_prob:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                # Mutation
                self._mutate(c1)
                self._mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size: new_pop.append(c2)
            
            # Update populasi & fitness
            self.pop = new_pop
            self.fitness_cache = [self._eval(ind) for ind in self.pop]
            cur_best_val = min(self.fitness_cache)
            if cur_best_val < best_val:
                best_val = cur_best_val
                best_ind = self.pop[int(np.argmin(self.fitness_cache))]
            history.append(best_val)
            if verbose:
                print(f"[GA] iter {it}/{self.iters} best={best_val:.4f}")
        
        return {"best_assignment": best_ind, "best_value": best_val, "time": time.time() - start, "history": history}
