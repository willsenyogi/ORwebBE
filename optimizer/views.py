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
        data = request.data

        print(data)

        # Body Data (Depots, Customers, Vehicles per Depot, Vehicle Capacities, Customer Demands)
        depots = data['depots']
        customers = data['customers']
        vehicles_per_depot = data['vehicle_per_depot']
        vehicle_capacities = data['vehicle_capacities']
        customer_demands = data['customer_demands']

        # Hyperparameters
        hyperparams = data.get('parameters')

        # Gurobi toggle
        run_ilp = hyperparams['run_ilp']

        # Max iterations and population size
        iters = hyperparams['maxIterations']
        pop_size = hyperparams['populationSize']

        # PSO Hyperparameters
        c1 = hyperparams.get('pso_c1')
        c2 = hyperparams.get('pso_c2')
        w = hyperparams.get('pso_w')

        # GA Hyperparameters
        mutation_rate = hyperparams.get('ga_mutation_prob')
        crossover_rate = hyperparams.get('ga_crossover_prob')

        total_vehicles = sum(vehicles_per_depot)
        vehicle_depot_map = [idx for idx, num in enumerate(vehicles_per_depot) for _ in range(num)]
        Dcust = DistanceMatrix.build_distance_matrix(customers)
        all_results, all_labels = [], []
        
        if run_ilp == True:
            try:
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
            except MemoryError as ex:
                return Response({"error": "Memory Error"}, status=400)

        pso=ParticleSwarmOptimization(
            depots=depots, customers=customers, vehicles_per_depot=vehicles_per_depot, 
            vehicle_capacities=vehicle_capacities, 
            customer_demands=customer_demands, 
            pop_size=pop_size, iters=iters, seed=None,
            c1=c1, c2=c2, w=w
        )
        
        res_pso=pso.run(verbose=True)
        all_results.append(res_pso)
        all_labels.append("PSO")

        ga=GeneticAlgorithm(
            depots=depots, customers=customers, vehicles_per_depot=vehicles_per_depot, 
            vehicle_capacities=vehicle_capacities, 
            customer_demands=customer_demands, 
            pop_size=pop_size, iters=iters, seed=None,
            cx_prob=crossover_rate, mut_prob=mutation_rate
        )
        
        res_ga=ga.run(verbose=True)
        all_results.append(res_ga)
        all_labels.append("GA")

        if run_ilp == True:
            PlotGenerator.plot_solution_with_dropdown(res_ilp, "ILP (Optimal Value)", depots, customers, Dcust, vehicle_depot_map)

        pso_plot = PlotGenerator.plot_solution_with_dropdown(res_pso, "PSO", depots, customers, Dcust, vehicle_depot_map)

        ga_plot = PlotGenerator.plot_solution_with_dropdown(res_ga, "GA", depots, customers, Dcust, vehicle_depot_map)

        res_pso['plot'] = pso_plot
        res_ga['plot'] = ga_plot

        if run_ilp == True:
            response = {
                "success" : True,
                "data": {
                    "ilp" : res_ilp,
                    "pso": res_pso,
                    "ga" : res_ga
                }
            }
        else:    
            response = {
                "success" : True,
                "data": {
                    "pso": res_pso,
                    "ga" : res_ga
                }
            }
        return Response(response, status=200)
        



