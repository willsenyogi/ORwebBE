import math
import random
import time
import copy
from typing import List, Tuple
import numpy as np

# -----------------------------
# Utility: Haversine & distance matrix
# -----------------------------

R_EARTH = 6371.0  # km
class HaversineFormulation():
    def haversine(a: Tuple[float,float], b: Tuple[float,float]) -> float:
        """a, b = (lon, lat) in degrees -> distance in km"""
        lon1, lat1 = math.radians(a[0]), math.radians(a[1])
        lon2, lat2 = math.radians(b[0]), math.radians(b[1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        sa = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(sa))
        return R_EARTH * c

class DistanceMatrix():
    def build_distance_matrix(points: List[Tuple[float,float]]) -> np.ndarray:
        n = len(points)
        D = np.zeros((n,n), dtype=float)
        for i in range(n):
            for j in range(i+1, n):
                d = HaversineFormulation.haversine(points[i], points[j])
                D[i,j] = d
                D[j,i] = d
        return D
    
# -----------------------------
# Heuristic: nearest-neighbor route for a depot given its assigned customers
# -----------------------------
class HeuristicFunctions():
    def build_route_and_length(self, depot_coords, customers_idx, customers_coords, customer_distance_matrix):
        """
        Versi yang disederhanakan: hanya butuh 4 argumen.
        Menghitung urutan rute nearest-neighbor dan total jaraknya.
        """
        if not customers_idx: 
            return [], 0.0
            
        remaining = customers_idx.copy()
        route = []
        
        # Mulai dari pelanggan terdekat ke depot
        depot_to_customers = [HaversineFormulation.haversine(depot_coords, customers_coords[c]) for c in remaining]
        cur_choice = remaining[np.argmin(depot_to_customers)]
        route.append(cur_choice)
        remaining.remove(cur_choice)
        
        # Lanjutkan dengan pelanggan terdekat berikutnya
        while remaining:
            last = route[-1]
            dists = [customer_distance_matrix[last, r] for r in remaining]
            next_choice = remaining[np.argmin(dists)]
            route.append(next_choice)
            remaining.remove(next_choice)
            
        # Hitung total jarak
        total_dist = HaversineFormulation.haversine(depot_coords, customers_coords[route[0]])
        for i in range(len(route) - 1):
            total_dist += customer_distance_matrix[route[i], route[i+1]]
        total_dist += HaversineFormulation.haversine(customers_coords[route[-1]], depot_coords)
        
        return route, total_dist

    # -----------------------------
    # Objective: given assignment (list of depot indices per customer) compute total distance
    # -----------------------------
    def total_distance_for_assignment(self, assignment: List[int],
                                    depots: List[Tuple[float,float]],
                                    customers: List[Tuple[float,float]],
                                    customer_distance_matrix: np.ndarray) -> float:
        m = len(depots)
        groups = {i: [] for i in range(m)}
        for cust_idx, dep_idx in enumerate(assignment):
            groups[dep_idx].append(cust_idx)
        total = 0.0
        for dep_idx in range(m):
            _, dist = self.build_route_and_length(dep_idx, depots[dep_idx], groups[dep_idx], customers, customer_distance_matrix)
            total += dist
        return total
    
class RoutesPrinter:
    def print_routes(result, label, depots, customers, customer_demands, vehicle_depot_map, vehicle_capacities, Dcust):
        assign = result.get("best_assignment", [])
        total_dist = result.get("best_value", float('inf'))
        comp_time = result.get("time", 0)

        if not assign or -1 in assign:
            print(f"\n--- {label} FAILED OR DID NOT FIND A VALID SOLUTION ---")
            print(f"{label} computation time = {comp_time:.4f} s")
            return
        
        vehicle_groups = {v_idx: [] for v_idx in range(len(vehicle_depot_map))}
        for cust_idx, vehicle_idx in enumerate(assign):
            vehicle_groups[vehicle_idx].append(cust_idx)
        
        depot_vehicle_groups = {d_idx: [] for d_idx in range(len(depots))}
        for v_idx, d_idx in enumerate(vehicle_depot_map):
            depot_vehicle_groups[d_idx].append(v_idx)
        
        print(f"\n--- {label} BEST ROUTE DETAILS ---")
        print(f"{label} total distance = {total_dist:.4f} km")
        print(f"{label} computation time = {comp_time:.4f} s")

        for d_idx in sorted(depot_vehicle_groups.keys()):
            print(f"\n  Depot {d_idx}:")
            depot_vehicles = depot_vehicle_groups[d_idx]
            if not any(vehicle_groups[v_idx] for v_idx in depot_vehicles):
                print("    - No vehicles used from this depot.")
                continue
                
            for v_idx in depot_vehicles:
                cust_list = vehicle_groups[v_idx]
                if cust_list:
                    home_depot_coords = depots[d_idx]
                    route, dist = HeuristicFunctions.build_route_and_length(home_depot_coords, cust_list, customers, Dcust)
                    
                    total_demand = sum(customer_demands[c] for c in cust_list)
                    current_vehicle_capacity = vehicle_capacities[v_idx]
                    
                    route_str = str(route) if len(route) < 10 else f"[{route[0]}, ..., {route[-1]}]"
                    print(f"    - Vehicle {v_idx}: serves {len(cust_list)} customers (Demand: {total_demand}/{current_vehicle_capacity}) -> route {route_str} -> dist {dist:.2f} km")
                else:
                    print(f"    - Vehicle {v_idx}: Not used.")