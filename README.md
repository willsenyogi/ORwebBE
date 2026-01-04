# MDVRP Simulation - Backend
Aplikasi web simulasi Multi-Depot Vehicle Routing Problem (MDVRP) yang membandingkan kinerja tiga algoritma optimasi: **PSO** (Particle Swarm Optimization), **GA** (Genetic Algorithm), dan **ILP** (Integer Linear Programming).

## Prasyarat
Sebelum instalasi pastikan : 
**Python** ≥ 3.8 dan pip sudah terinstall di perangkat

**Gurobi Software** sudah terinstall ([Download](https://www.gurobi.com/downloads/gurobi-software/))


## Instalasi Library
```bash
pip install django
pip install djangorestframework
pip install numpy
pip install pandas
pip install plotly
pip install gurobipy
```
**Catatan :**

**gurobipy** memerlukan lisensi Gurobi yang aktif. Tanpa lisensi, model ILP tidak dapat dijalankan.

## Buka Terminal Baru
```bash
cd ORWebBE
```

## Menjalankan server
```bash
python manage.py runserver
```
Backend akan berjalan di `http://127.0.0.1:8000/`

# Repo frontend : ([OR-web-FE](https://github.com/willsenyogi/OR-web-FE))
