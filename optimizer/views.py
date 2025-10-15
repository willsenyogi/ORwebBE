from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .algorithms.ilp import SolverILP
from .algorithms.pso import ParticleSwarmOptimization
from .algorithms.ga import GeneticAlgorithm

class OptimizeView(APIView):
    def post(self, request):
        data = request.data
        return Response({"result": "result"}, status=200)
        



