from django.urls import path
from . import views

urlpatterns = [
    path('optimize/', views.OptimizeView.as_view(), name='optimize'),
]
