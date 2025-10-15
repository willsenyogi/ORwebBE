import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
from .utils import DistanceMatrix, HeuristicFunctions

class PlotGenerator:
    def plot_solution_with_dropdown(result, label, depots, customers, Dcust, vehicle_depot_map):
        assign = result.get("best_assignment", [])
        if not assign or -1 in assign or result.get("best_value", float('inf')) > 900000:
            print(f"\n--- {label} TIDAK MENEMUKAN SOLUSI LAYAK, PLOT DIBATALKAN ---")
            return

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        
        # 3 jejak per kendaraan: [titik_V0, garis_V0, nomor_V0, titik_V1, ...]
        for v_idx in range(len(vehicle_depot_map)):
            home_depot_idx = vehicle_depot_map[v_idx]
            color = colors[home_depot_idx % len(colors)]
            cust_list = [c_idx for c_idx, assigned_v in enumerate(assign) if assigned_v == v_idx]
            depot_coord = depots[home_depot_idx]
            
            cust_df = pd.DataFrame([customers[i] for i in cust_list], columns=['lon', 'lat'])
            cust_df['id'] = [f"C{i}" for i in cust_list]
            
            hover_text = cust_df['id'].astype(str) + f" → Vehicle {v_idx} (from D{home_depot_idx})"
            
            fig.add_trace(go.Scattermapbox(
                lat=cust_df['lat'], lon=cust_df['lon'], 
                mode='markers', marker={'size': 10, 'color': color}, 
                hoverinfo='text', text=hover_text, 
                name=f'Route V{v_idx}'
            ))
            
            route, _ = HeuristicFunctions.build_route_and_length(depot_coord, cust_list, customers, Dcust)
            path_lon, path_lat, route_texts, route_df = [], [], [], pd.DataFrame(columns=['lon', 'lat'])
            if route:
                path_coords = [depot_coord] + [customers[i] for i in route] + [depot_coord]
                path_lon, path_lat = [p[0] for p in path_coords], [p[1] for p in path_coords]
                route_df = pd.DataFrame([customers[i] for i in route], columns=['lon', 'lat'])
                route_texts = [str(i+1) for i in range(len(route))]
            
            fig.add_trace(go.Scattermapbox(lat=path_lat, lon=path_lon, mode='lines', line=dict(width=2, color=color), hoverinfo='none', showlegend=False))
            fig.add_trace(go.Scattermapbox(lat=route_df['lat'], lon=route_df['lon'], mode='text', text=route_texts, textfont={'size': 10, 'color': 'white', 'family': 'Arial Black'}, hoverinfo='none', showlegend=False))

        depot_df = pd.DataFrame(depots, columns=['lon', 'lat'])
        depot_df['id'] = [f"Depot {i}" for i in range(len(depots))]
        fig.add_trace(go.Scattermapbox(lat=depot_df['lat'], lon=depot_df['lon'], mode='markers+text', marker={'size': 20, 'color': 'black', 'symbol': 'circle'}, text=[f"D{i}" for i in range(len(depots))], textfont={'color':'white', 'size':10}, hoverinfo='text', name='Depot'))
        
        buttons = []
        num_vehicles = len(vehicle_depot_map)
        num_traces_per_vehicle = 3
        
        all_visibility = [True if i % num_traces_per_vehicle == 0 else False for i in range(num_traces_per_vehicle * num_vehicles)] + [True]
        buttons.append(dict(label="All Routes (Clusters)", method="restyle", args=[{"visible": all_visibility}]))
        
        for i in range(num_vehicles):
            vehicle_visibility = [False] * (num_traces_per_vehicle * num_vehicles + 1)
            start_index = i * num_traces_per_vehicle
            vehicle_visibility[start_index:start_index+3] = [True, True, True]
            vehicle_visibility[-1] = True
            buttons.append(dict(label=f"Vehicle {i} (from D{vehicle_depot_map[i]})", method="restyle", args=[{"visible": vehicle_visibility}]))
        
        fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, x=0.01, xanchor="left", y=0.99, yanchor="top")])
        fig.update_layout(title=f'Visualisasi Rute Interaktif - {label}', mapbox_style="carto-positron", mapbox_center_lon=np.mean([c[0] for c in customers]), mapbox_center_lat=np.mean([c[1] for c in customers]), mapbox_zoom=10, margin={"r":0,"t":40,"l":0,"b":0})
        fig.show()
        
    def plot_final_comparison(results, labels):
        distances = [res.get('best_value', 0) if res.get('best_value', float('inf')) < 900000 else 0 for res in results]
        times = [res.get('time', 0) for res in results]
        fig = go.Figure(data=[go.Bar(name='Jarak', x=labels, y=distances, text=[f"{d:.2f}" for d in distances], textposition='auto')])
        fig.update_layout(title_text='Perbandingan Final Total Jarak', yaxis_title='Total Jarak (km)')
        fig.show()
        fig = go.Figure(data=[go.Bar(name='Waktu', x=labels, y=times, text=[f"{t:.2f}" for t in times], textposition='auto')])
        fig.update_layout(title_text='Perbandingan Waktu Komputasi', yaxis_title='Waktu Komputasi (detik)')
        fig.show()