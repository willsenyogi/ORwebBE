from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .algorithms.ilp import SolverILP
from .algorithms.pso import ParticleSwarmOptimization
from .algorithms.ga import GeneticAlgorithm

from .algorithms.plot_generator import PlotGenerator
from .algorithms.utils import HaversineFormulation, DistanceMatrix, RoutesPrinter

class OptimizeView(APIView):
    def post(self, request):
        run_ilp = request.query_params.get('run_ilp', False)
        data = request.data
        
        depots = data['depots']
        customers = data['customers']
        vehicles_per_depot = data['vehicle_per_depot']
        vehicle_capacities = data['vehichle_capacities']
        customer_demands = data['customer_demands']

        total_vehicles = sum(vehicles_per_depot)
        vehicle_depot_map = [idx for idx, num in enumerate(vehicles_per_depot) for _ in range(num)]
        Dcust = DistanceMatrix.build_distance_matrix(customers)
        all_results, all_labels = [], []

        if run_ilp == 'True':
            run_ilp = True
        else:
            run_ilp = False
            
        if run_ilp == True:
            ilp=SolverILP(
                depots=depots, 
                customers=customers,
                vehicles_per_depot=vehicles_per_depot,
                vehicle_capacities=vehicle_capacities,
                customer_demands=customer_demands,
                time_limit=72000
            )
            
            res_ilp=ilp.run(verbose=True, run_ilp=run_ilp)
            all_results.append(res_ilp)
            all_labels.append("ILP")

        pso=ParticleSwarmOptimization(
            depots=depots, customers=customers, vehicles_per_depot=vehicles_per_depot, 
            vehicle_capacities=vehicle_capacities, 
            customer_demands=customer_demands, 
            pop_size=100, iters=100, seed=None
        )
        
        res_pso=pso.run(verbose=True)
        all_results.append(res_pso)
        all_labels.append("PSO")

        ga=GeneticAlgorithm(
            depots=depots, customers=customers, vehicles_per_depot=vehicles_per_depot, 
            vehicle_capacities=vehicle_capacities, 
            customer_demands=customer_demands, 
            pop_size=100, iters=100, seed=None
        )
        
        res_ga=ga.run(verbose=True)
        all_results.append(res_ga)
        all_labels.append("GA")

        if run_ilp == True:
            PlotGenerator.plot_solution_with_dropdown(res_ilp, "ILP (Optimal Value)", depots, customers, Dcust, vehicle_depot_map)

        pso_plot = PlotGenerator.plot_solution_with_dropdown(res_pso, "PSO", depots, customers, Dcust, vehicle_depot_map)

        ga_plot = PlotGenerator.plot_solution_with_dropdown(res_ga, "GA", depots, customers, Dcust, vehicle_depot_map)

        response = {
            "PSO": res_pso,
            "GA" : res_ga
        }
        return Response(response, status=200)
        



