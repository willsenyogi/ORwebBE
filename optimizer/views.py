from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .algorithms.ilp import SolverILP
from .algorithms.pso import ParticleSwarmOptimization
from .algorithms.ga import GeneticAlgorithm

class OptimizeView(APIView):
    def post(self, request):
        run_ilp = request.query_params.get('run_ilp', False)
        data = request.data
        
        depots = data['depots']
        customers = data['customers']
        vehicles_per_depot = data['vehicle_per_depot']
        vehicle_capacities = data['vehichle_capacities']
        customer_demands = data['customer_demands']
        
        if run_ilp == 'True':
            run_ilp = True
        else:
            run_ilp = False
            
        
        ilp=SolverILP(
            depots=depots,
            customers=customers,
            vehicles_per_depot=vehicles_per_depot,
            vehicle_capacities=vehicle_capacities,
            customer_demands=customer_demands
        )
        
        res_ilp=ilp.run(verbose=True, run_ilp=run_ilp)
        
        return Response({"result": "result"}, status=200)
        



