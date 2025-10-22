import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
from .utils import DistanceMatrix, HeuristicFunctions
heuristic = HeuristicFunctions()

class PlotGenerator:
    def plot_solution_with_dropdown(result, label, depots, customers, Dcust, vehicle_depot_map):
        assign = result.get("best_assignment", [])
        if not assign or -1 in assign or result.get("best_value", float('inf')) > 900000:
            print(f"\n--- {label} TIDAK MENEMUKAN SOLUSI LAYAK, PLOT DIBATALKAN ---")
            return {}

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        
        route_info_strings = {}
        active_vehicle_indices = [] 

        for v_idx in range(len(vehicle_depot_map)):
            home_depot_idx = vehicle_depot_map[v_idx]
            color = colors[home_depot_idx % len(colors)]
            cust_list = [c_idx for c_idx, assigned_v in enumerate(assign) if assigned_v == v_idx]
            depot_coord = depots[home_depot_idx]
            
            if cust_list:
                active_vehicle_indices.append(v_idx)
                
            cust_df = pd.DataFrame([customers[i] for i in cust_list], columns=['lon', 'lat'])
            cust_df['id'] = [f"C{i+1}" for i in cust_list]
            
            hover_text = cust_df['id'].astype(str) + f" → Vehicle {v_idx+1} (from D{home_depot_idx+1})"
            
            fig.add_trace(go.Scattermapbox(
                lat=cust_df['lat'], lon=cust_df['lon'], 
                mode='markers', marker={'size': 10, 'color': color}, 
                hoverinfo='text', text=hover_text, 
                name=f'Route V{v_idx+1}'
            ))
            
            route, _ = heuristic.build_route_and_length(depot_coord, cust_list, customers, Dcust)
            path_lon, path_lat, route_texts, route_df = [], [], [], pd.DataFrame(columns=['lon', 'lat'])
            
            if route:
                path_coords = [depot_coord] + [customers[i] for i in route] + [depot_coord]
                path_lon, path_lat = [p[0] for p in path_coords], [p[1] for p in path_coords]
                route_df = pd.DataFrame([customers[i] for i in route], columns=['lon', 'lat'])
                route_texts = [str(i+1) for i in range(len(route))]
                
                route_str = f"Urutan Rute V{v_idx+1}: D{home_depot_idx+1} → " + " → ".join([f"C{c+1}" for c in route]) + f" → D{home_depot_idx+1}"
            else:
                route_str = f"Urutan Rute V{v_idx+1}: Tidak digunakan"
            
            route_info_strings[v_idx] = route_str
            
            fig.add_trace(go.Scattermapbox(lat=path_lat, lon=path_lon, mode='lines', line=dict(width=2, color=color), hoverinfo='none', showlegend=False))
            fig.add_trace(go.Scattermapbox(lat=route_df['lat'], lon=route_df['lon'], mode='text', text=route_texts, textfont={'size': 10, 'color': 'white', 'family': 'Arial Black'}, hoverinfo='none', showlegend=False))

        depot_df = pd.DataFrame(depots, columns=['lon', 'lat'])
        depot_df['id'] = [f"Depot {i+1}" for i in range(len(depots))]
        fig.add_trace(go.Scattermapbox(
            lat=depot_df['lat'], lon=depot_df['lon'], 
            mode='markers+text', marker={'size': 20, 'color': 'black', 'symbol': 'circle'}, 
            text=[f"D{i+1}" for i in range(len(depots))], 
            textfont={'color':'white', 'size':10}, 
            hoverinfo='text', name='Depot'
        ))
        

        buttons = []
        num_vehicles = len(vehicle_depot_map)
        num_traces_per_vehicle = 3
        
        all_visibility = [True if i % num_traces_per_vehicle in [0, 1] else False for i in range(num_traces_per_vehicle * num_vehicles)] + [True]
        all_routes_text = "Menampilkan semua rute. Pilih satu kendaraan untuk rincian."

        buttons.append(dict(label="All Routes", 
                            method="update", 
                            args=[{"visible": all_visibility}, 
                                {"annotations[0].text": all_routes_text}]))
        
        for i in range(num_vehicles):
            if i in active_vehicle_indices:
                vehicle_visibility = [False] * (num_traces_per_vehicle * num_vehicles + 1)
                start_index = i * num_traces_per_vehicle
                vehicle_visibility[start_index:start_index+3] = [True, True, True]
                vehicle_visibility[-1] = True
                
                buttons.append(dict(label=f"Vehicle {i+1} (from D{vehicle_depot_map[i]+1})", 
                                    method="update", 
                                    args=[{"visible": vehicle_visibility}, 
                                        {"annotations[0].text": route_info_strings[i]}]))
        
        # Update layout dengan anotasi
        fig.update_layout(
            updatemenus=[dict(active=0, buttons=buttons, x=0.01, xanchor="left", y=1.0, yanchor="top")],
            annotations=[
                go.layout.Annotation(
                    text=all_routes_text,
                    align='left',
                    showarrow=False,
                    xref='paper', yref='paper',
                    x=0.5, y=0.99,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(size=12)
                )
            ]
        )
        
        if customers:
            avg_lon = np.mean([c[0] for c in customers])
            avg_lat = np.mean([c[1] for c in customers])
        else:
            avg_lon, avg_lat = np.mean([d[0] for d in depots]), np.mean([d[1] for d in depots]) # Fallback ke depot

        fig.update_layout(
            title=f'Visualisasi Rute Interaktif - {label}', 
            mapbox_style="carto-positron", 
            mapbox_center_lon=avg_lon, 
            mapbox_center_lat=avg_lat, 
            mapbox_zoom=10, 
            margin={"r":0,"t":40,"l":0,"b":0}
        )
        return fig.to_dict()
        
    def plot_final_comparison(results, labels):
        distances = [res.get('best_value', 0) if res.get('best_value', float('inf')) < 900000 else 0 
                     for res in results]
        times = [res.get('time', 0) for res in results]
        
        # Distance comparison
        fig_distance = go.Figure(data=[
            go.Bar(
                name='Jarak', 
                x=labels, 
                y=distances, 
                text=[f"{d:.2f}" for d in distances], 
                textposition='auto',
                marker_color='#030213'
            )
        ])
        fig_distance.update_layout(
            title_text='Perbandingan Total Jarak',
            yaxis_title='Total Jarak (km)',
            showlegend=False
        )
        
        # Time comparison
        fig_time = go.Figure(data=[
            go.Bar(
                name='Waktu', 
                x=labels, 
                y=times, 
                text=[f"{t:.2f}" for t in times], 
                textposition='auto',
                marker_color='#6366f1'
            )
        ])
        fig_time.update_layout(
            title_text='Perbandingan Waktu Komputasi',
            yaxis_title='Waktu Komputasi (detik)',
            showlegend=False
        )
        
        return fig_distance.to_dict(), fig_time.to_dict()