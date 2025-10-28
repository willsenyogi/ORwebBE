import time
import gurobipy as gp
from gurobipy import GRB
from .utils import DistanceMatrix
from pathlib import Path

class SolverILP:
    def __init__(self, depots, customers, vehicles_per_depot, vehicle_capacities, customer_demands, time_limit=60):
        # Inisialisasi parameter dasar dari model MDVRP
        self.depots = depots
        self.customers = customers
        self.vehicles_per_depot = vehicles_per_depot
        self.vehicle_capacities = vehicle_capacities
        self.customer_demands = customer_demands
        self.m = len(depots)  # jumlah depot
        self.n = len(customers)  # jumlah pelanggan
        self.time_limit = time_limit
        self.total_vehicles = sum(vehicles_per_depot)  # total kendaraan di semua depot
        # Map global kendaraan ke depot masing-masing
        self.vehicle_depot_map = [idx for idx, num in enumerate(self.vehicles_per_depot) for _ in range(num)]

    def run(self, verbose=False, run_ilp=True):
        if run_ilp:
            start = time.time()
            all_points = self.depots + self.customers
            num_points = len(all_points)
            
            # Bangun matriks jarak antar titik (depot + pelanggan)
            dist_matrix = DistanceMatrix.build_distance_matrix(all_points)
            
            # Gabungkan demand pelanggan dan depot (depot = 0 demand)
            full_demands = [0] * self.m + self.customer_demands

            depot_indices = list(range(self.m))
            customer_indices = list(range(self.m, num_points))

            # Inisialisasi model Gurobi
            model = gp.Model("MDVRP_ILP")
            if not verbose:
                model.setParam('OutputFlag', 0)

            # ======================================================
            # Variabel Keputusan
            # ======================================================
            # x[i,j,m,k] = 1 jika kendaraan ke-k dari depot m berpindah dari i ke j
            # y[i,m,k] = 1 jika kendaraan ke-k dari depot m melayani pelanggan i
            # u[i,m,k] = variabel continuous untuk eliminasi subtour (MTZ)
            x = model.addVars(num_points, num_points, self.m, max(self.vehicles_per_depot), vtype=GRB.BINARY, name="x")
            y = model.addVars(num_points, self.m, max(self.vehicles_per_depot), vtype=GRB.BINARY, name="y")
            u = model.addVars(num_points, self.m, max(self.vehicles_per_depot), vtype=GRB.CONTINUOUS, name="u")

            # ======================================================
            # (2.1) Fungsi Objektif
            # ======================================================
            # Meminimalkan total jarak perjalanan seluruh kendaraan
            model.setObjective(
                gp.quicksum(dist_matrix[i, j] * x[i, j, m, k]
                            for m in depot_indices
                            for k in range(self.vehicles_per_depot[m])
                            for i in range(num_points)
                            for j in range(num_points)
                            if i != j),
                GRB.MINIMIZE
            )

            # ======================================================
            # (2.2) Kendala Kapasitas Kendaraan
            # ======================================================
            vehicle_offset = 0
            for m in depot_indices:
                for k in range(self.vehicles_per_depot[m]):
                    global_k = vehicle_offset + k
                    model.addConstr(
                        gp.quicksum(full_demands[i] * y[i, m, k] for i in customer_indices) <= self.vehicle_capacities[global_k],
                        name=f"capacity_m{m}_k{k}"
                    )
                vehicle_offset += self.vehicles_per_depot[m]

            # ======================================================
            # (2.6) Setiap Pelanggan Harus Dilayani Tepat Sekali
            # ======================================================
            for i in customer_indices:
                model.addConstr(
                    gp.quicksum(y[i, m, k] for m in depot_indices for k in range(self.vehicles_per_depot[m])) == 1,
                    name=f"serve_once_i{i}"
                )

            # ======================================================
            # (2.4) Jumlah pelanggan per rute sesuai depot assignment
            # ======================================================
            for m in depot_indices:
                model.addConstr(
                    gp.quicksum(y[i, m, k] for k in range(self.vehicles_per_depot[m]) for i in customer_indices) >= 0,
                    name=f"depot_has_customer_m{m}"
                )

            # ======================================================
            # (Flow Consistency) dan Hubungan antara x ↔ y
            # ======================================================
            for m in depot_indices:
                for k in range(self.vehicles_per_depot[m]):
                    # Keseimbangan aliran untuk setiap pelanggan
                    for i in customer_indices:
                        # Kendaraan keluar dari pelanggan i
                        model.addConstr(
                            gp.quicksum(x[i, j, m, k] for j in range(num_points) if j != i) == y[i, m, k],
                            name=f"link_out_m{m}_k{k}_i{i}"
                        )
                        # Kendaraan masuk ke pelanggan i
                        model.addConstr(
                            gp.quicksum(x[j, i, m, k] for j in range(num_points) if j != i) == y[i, m, k],
                            name=f"link_in_m{m}_k{k}_i{i}"
                        )

                    # Kendaraan harus keluar dan kembali ke depot yang sama
                    depot_node = m
                    model.addConstr(
                        gp.quicksum(x[depot_node, j, m, k] for j in customer_indices) ==
                        gp.quicksum(x[j, depot_node, m, k] for j in customer_indices),
                        name=f"depot_flow_m{m}_k{k}"
                    )

            # ======================================================
            # Subtour Elimination Constraint (MTZ)
            # ======================================================
            vehicle_offset_mtz = 0
            for m in depot_indices:
                for k in range(self.vehicles_per_depot[m]):
                    global_k = vehicle_offset_mtz + k
                    current_capacity = self.vehicle_capacities[global_k]

                    for i in customer_indices:
                        model.addConstr(u[i, m, k] >= full_demands[i])
                        model.addConstr(u[i, m, k] <= current_capacity)
                        for j in customer_indices:
                            if i != j:
                                model.addConstr(
                                    u[i, m, k] + full_demands[j] - u[j, m, k] <= current_capacity * (1 - x[i, j, m, k]),
                                    name=f"mtz_m{m}_k{k}_i{i}_j{j}"
                                )
                vehicle_offset_mtz += self.vehicles_per_depot[m]

            # ======================================================
            # Kendala tambahan: mencegah perpindahan antar depot
            # ======================================================
            for m in depot_indices:
                for k in range(self.vehicles_per_depot[m]):
                    for d in depot_indices:
                        if d != m:
                            for j in range(num_points):
                                if j != d:
                                    model.addConstr(x[d, j, m, k] == 0)
                                    model.addConstr(x[j, d, m, k] == 0)

            # ======================================================
            # Pengaturan Parameter Model Gurobi
            # ======================================================
            model.Params.NodefileStart = 0.1
            model.Params.NodefileDir = str(Path.home() / "Desktop" / "gurobi_tmp")
            model.Params.Presolve = 1
            model.Params.Aggregate = 1
            model.Params.Cuts = 1
            model.Params.PreCrush = 1
            model.Params.MIPGap = 0.0
            # model.Params.TimeLimit = self.time_limit  # opsional

            # ======================================================
            # Proses Optimasi ILP
            # ======================================================
            model.optimize()
            end = time.time()

            # ======================================================
            # Ekstraksi Hasil Solusi
            # ======================================================
            best_assignment = [-1] * self.n
            best_value = model.ObjVal if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] else float('inf')
            best_routes_ilp = {}

            if best_value < float('inf'):
                solution_x = model.getAttr('x', x)
                global_vehicle_idx = 0
                for m in depot_indices:
                    depot_node = m
                    for k in range(self.vehicles_per_depot[m]):
                        
                        vehicle_route = []
                        vehicle_dist = 0.0
                        
                        # Cari node pertama setelah depot
                        current_node = depot_node
                        for j in customer_indices:
                            if solution_x[current_node, j, m, k] > 0.5:
                                cust_idx = j - self.m
                                vehicle_route.append(cust_idx)
                                vehicle_dist += dist_matrix[current_node, j]
                                current_node = j
                                best_assignment[cust_idx] = global_vehicle_idx
                                break

                        # Lanjutkan rute sampai kembali ke depot
                        while current_node != depot_node:
                            found_next = False
                            for j in range(num_points):
                                if j != current_node and solution_x[current_node, j, m, k] > 0.5:
                                    vehicle_dist += dist_matrix[current_node, j]
                                    if j in customer_indices:
                                        cust_idx = j - self.m
                                        vehicle_route.append(cust_idx)
                                        best_assignment[cust_idx] = global_vehicle_idx
                                        current_node = j
                                    else:
                                        current_node = depot_node
                                    found_next = True
                                    break
                            if not found_next:
                                break

                        # Simpan rute valid
                        if vehicle_route:
                            best_routes_ilp[global_vehicle_idx] = {
                                'route': vehicle_route,
                                'dist': vehicle_dist
                            }
                        global_vehicle_idx += 1

            # ======================================================
            # Return hasil akhir optimasi
            # ======================================================
            return {
                "best_assignment": best_assignment,  # Mapping pelanggan ke kendaraan
                "best_value": best_value,            # Nilai objektif minimum
                "time": end - start,                 # Waktu eksekusi
                "history": [],                       # Kosong (bisa isi log nanti)
                "best_routes_ilp": best_routes_ilp   # Rute terbaik tiap kendaraan
            }
