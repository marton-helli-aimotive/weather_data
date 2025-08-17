"""Interactive visualization components for the weather dashboard."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def create_time_series_plot(data: List[Dict[str, Any]], parameters: List[str]) -> go.Figure:
    """Create an interactive time series plot with multiple parameters.
    
    Args:
        data: List of weather data points
        parameters: List of parameters to plot
        
    Returns:
        Plotly figure with time series data
    """
    if not data or not parameters:
        return go.Figure()
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplots for multiple parameters
    fig = make_subplots(
        rows=len(parameters),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[param.replace('_', ' ').title() for param in parameters]
    )
    
    # Color palette for different cities
    colors = px.colors.qualitative.Set3
    
    # Get unique cities
    cities = df['city'].unique() if 'city' in df.columns else ['Unknown']
    
    for i, param in enumerate(parameters, 1):
        if param not in df.columns:
            continue
            
        for j, city in enumerate(cities):
            city_data = df[df['city'] == city] if 'city' in df.columns else df
            
            if city_data.empty or city_data[param].isna().all():
                continue
            
            fig.add_trace(
                go.Scatter(
                    x=city_data['timestamp'],
                    y=city_data[param],
                    mode='lines+markers',
                    name=f"{city} - {param}",
                    line=dict(color=colors[j % len(colors)]),
                    marker=dict(size=4),
                    hovertemplate=(
                        f"<b>{city}</b><br>"
                        f"{param.replace('_', ' ').title()}: %{{y}}<br>"
                        "Time: %{x}<br>"
                        "<extra></extra>"
                    ),
                    legendgroup=city,
                    showlegend=(i == 1)  # Show legend only for first subplot
                ),
                row=i, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=150 * len(parameters) + 100,
        title={
            'text': "Weather Parameters Over Time",
            'x': 0.5,
            'xanchor': 'center'
        },
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update x-axes
    fig.update_xaxes(title_text="Time", row=len(parameters), col=1)
    
    # Update y-axes with units
    parameter_units = {
        'temperature': '°C',
        'humidity': '%',
        'pressure': 'hPa',
        'wind_speed': 'm/s',
        'precipitation': 'mm',
        'visibility': 'km',
        'cloud_cover': '%',
        'uv_index': ''
    }
    
    for i, param in enumerate(parameters, 1):
        unit = parameter_units.get(param, '')
        fig.update_yaxes(
            title_text=f"{param.replace('_', ' ').title()} ({unit})" if unit else param.replace('_', ' ').title(),
            row=i, col=1
        )
    
    return fig


def create_geographic_map(data: List[Dict[str, Any]]) -> go.Figure:
    """Create an enhanced interactive geographic map with weather data.
    
    Args:
        data: List of weather data points with coordinates
        
    Returns:
        Plotly figure with geographic visualization
    """
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    # Get latest data for each city
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest_data = df.loc[df.groupby('city')['timestamp'].idxmax()]
    else:
        latest_data = df.drop_duplicates('city', keep='last')
    
    # Extract coordinates
    if 'coordinates' in latest_data.columns:
        latest_data['lat'] = latest_data['coordinates'].apply(
            lambda x: x.get('latitude') if isinstance(x, dict) else None
        )
        latest_data['lon'] = latest_data['coordinates'].apply(
            lambda x: x.get('longitude') if isinstance(x, dict) else None
        )
    else:
        # Default coordinates for common cities (this should not happen with the new data manager)
        city_coords = {
            'new_york': {'lat': 40.7128, 'lon': -74.0060},
            'london': {'lat': 51.5074, 'lon': -0.1278},
            'tokyo': {'lat': 35.6762, 'lon': 139.6503},
            'sydney': {'lat': -33.8688, 'lon': 151.2093},
            'paris': {'lat': 48.8566, 'lon': 2.3522},
        }
        latest_data['lat'] = latest_data['city'].map(lambda x: city_coords.get(x, {}).get('lat'))
        latest_data['lon'] = latest_data['city'].map(lambda x: city_coords.get(x, {}).get('lon'))
    
    # Filter out rows without coordinates
    latest_data = latest_data.dropna(subset=['lat', 'lon'])
    
    if latest_data.empty:
        return go.Figure()
    
    # Create the map
    fig = go.Figure()
    
    # Add temperature contour/heatmap background
    if 'temperature' in latest_data.columns and len(latest_data) > 3:
        # Create a grid for interpolation
        lat_min, lat_max = latest_data['lat'].min() - 5, latest_data['lat'].max() + 5
        lon_min, lon_max = latest_data['lon'].min() - 5, latest_data['lon'].max() + 5
        
        # Create interpolated temperature surface
        from scipy.interpolate import griddata
        import numpy as np
        
        # Grid points for interpolation
        grid_lat = np.linspace(lat_min, lat_max, 50)
        grid_lon = np.linspace(lon_min, lon_max, 50)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)
        
        # Interpolate temperature data
        try:
            grid_temp = griddata(
                (latest_data['lon'], latest_data['lat']),
                latest_data['temperature'],
                (grid_lon_mesh, grid_lat_mesh),
                method='cubic',
                fill_value=latest_data['temperature'].mean()
            )
            
            # Add temperature contour
            fig.add_trace(
                go.Densitymapbox(
                    lat=grid_lat_mesh.flatten(),
                    lon=grid_lon_mesh.flatten(),
                    z=grid_temp.flatten(),
                    radius=40,
                    colorscale='RdYlBu_r',
                    opacity=0.6,
                    showscale=True,
                    colorbar=dict(
                        title="Temperature (°C)",
                        x=0.02,
                        len=0.7
                    ),
                    name="Temperature Field"
                )
            )
        except ImportError:
            # Fallback if scipy not available
            pass
    
    # Add city markers with enhanced styling
    if 'temperature' in latest_data.columns:
        # Size markers based on temperature (but keep reasonable range)
        # Handle NaN values properly
        temp_values = latest_data['temperature'].fillna(latest_data['temperature'].mean())
        temp_min, temp_max = temp_values.min(), temp_values.max()
        
        # Avoid division by zero
        if temp_max - temp_min > 0:
            marker_size = ((temp_values - temp_min) / (temp_max - temp_min) * 15) + 10
        else:
            marker_size = [15] * len(temp_values)  # Default size if all temps are the same
        
        # Ensure no NaN values in marker_size
        marker_size = marker_size.fillna(15)
        
        fig.add_trace(
            go.Scattermapbox(
                lat=latest_data['lat'],
                lon=latest_data['lon'],
                mode='markers+text',
                marker=dict(
                    size=marker_size.tolist(),  # Convert to list to avoid NaN issues
                    color=latest_data['temperature'],
                    colorscale='Viridis',
                    opacity=0.8,
                    sizemode='diameter'
                ),
                text=latest_data['city'],
                textposition="top center",
                textfont=dict(size=10, color='black'),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Temperature: %{marker.color:.1f}°C<br>"
                    "Humidity: " + latest_data.get('humidity', pd.Series(['N/A'] * len(latest_data))).astype(str) + "%<br>"
                    "Pressure: " + latest_data.get('pressure', pd.Series(['N/A'] * len(latest_data))).astype(str) + " hPa<br>"
                    "Wind Speed: " + latest_data.get('wind_speed', pd.Series(['N/A'] * len(latest_data))).astype(str) + " m/s<br>"
                    "Lat: %{lat:.2f}<br>"
                    "Lon: %{lon:.2f}<br>"
                    "<extra></extra>"
                ),
                name="Weather Stations"
            )
        )
    else:
        # Just show markers if no temperature data
        fig.add_trace(
            go.Scattermapbox(
                lat=latest_data['lat'],
                lon=latest_data['lon'],
                mode='markers+text',
                marker=dict(size=15, color='blue', opacity=0.8),
                text=latest_data['city'],
                textposition="top center",
                textfont=dict(size=10, color='black'),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Lat: %{lat:.2f}<br>"
                    "Lon: %{lon:.2f}<br>"
                    "<extra></extra>"
                ),
                name="Cities"
            )
        )
    
    # Update layout with enhanced styling
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=latest_data['lat'].mean(),
                lon=latest_data['lon'].mean()
            ),
            zoom=1.5
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        title={
            'text': "Global Weather Data Distribution",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
    )
    
    return fig


def create_3d_surface_plot(data: List[Dict[str, Any]]) -> go.Figure:
    """Create enhanced 3D analysis showing meaningful weather relationships.
    
    Args:
        data: List of weather data points
        
    Returns:
        Plotly figure with 3D analysis visualization
    """
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    # Check if we have the necessary columns
    required_cols = ['temperature', 'pressure', 'humidity']
    available_cols = [col for col in required_cols if col in df.columns]
    
    if len(available_cols) < 2:
        # Return empty figure if insufficient data
        return go.Figure().add_annotation(
            text="Insufficient data for 3D visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
    
    # Clean data - remove NaN values
    clean_df = df[available_cols + ['city']].dropna(subset=available_cols)
    
    if clean_df.empty or len(clean_df) < 3:
        return go.Figure().add_annotation(
            text="Not enough valid data points for 3D visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
    
    fig = go.Figure()
    
    # Create multiple analysis views
    if len(available_cols) >= 3:
        # Main 3D scatter with climate zones
        x = clean_df['temperature']
        y = clean_df['pressure']
        z = clean_df['humidity']
        
        # Calculate comfort index (simplified)
        comfort_index = []
        for _, row in clean_df.iterrows():
            temp = row['temperature']
            humid = row['humidity']
            # Simple comfort calculation (optimal around 22°C, 40-60% humidity)
            temp_comfort = 100 - abs(temp - 22) * 3
            humid_comfort = 100 - abs(humid - 50) * 1.5
            comfort = (temp_comfort + humid_comfort) / 2
            comfort_index.append(max(0, min(100, comfort)))
        
        # Add main 3D scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=comfort_index,
                    colorscale='RdYlGn',
                    colorbar=dict(
                        title="Comfort Index",
                        x=1.1,
                        len=0.6
                    ),
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                text=clean_df['city'],
                textposition="top center",
                name="Cities",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Temperature: %{x:.1f}°C<br>"
                    "Pressure: %{y:.1f} hPa<br>"
                    "Humidity: %{z:.1f}%<br>"
                    "Comfort Index: %{marker.color:.1f}<br>"
                    "<extra></extra>"
                )
            )
        )
        
        # Add climate zone boundaries (simplified)
        try:
            import numpy as np
            # Create reference surfaces for climate zones
            
            # Ideal comfort zone (around 22°C, 1013 hPa, 50% humidity)
            ideal_temp = np.full((10, 10), 22)
            ideal_pressure = np.linspace(1000, 1025, 10)
            ideal_humidity = np.linspace(40, 60, 10)
            P_ideal, H_ideal = np.meshgrid(ideal_pressure, ideal_humidity)
            
            fig.add_trace(
                go.Surface(
                    x=ideal_temp,
                    y=P_ideal,
                    z=H_ideal,
                    opacity=0.3,
                    colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
                    showscale=False,
                    name="Comfort Zone"
                )
            )
            
        except ImportError:
            pass  # Skip advanced visualization if numpy not available
        
        # Update layout for 3D scene
        fig.update_layout(
            title={
                'text': "3D Weather Analysis: Climate Comfort Zones",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            scene=dict(
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (hPa)",
                zaxis_title="Humidity (%)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            height=700,
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
        
    else:
        # Fallback to 2D plot if only 2 parameters available
        x_col = available_cols[0]
        y_col = available_cols[1]
        
        fig.add_trace(
            go.Scatter(
                x=clean_df[x_col],
                y=clean_df[y_col],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=clean_df.index,
                    colorscale='viridis',
                    opacity=0.7
                ),
                text=clean_df['city'],
                textposition="top center",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"{x_col.replace('_', ' ').title()}: %{{x}}<br>"
                    f"{y_col.replace('_', ' ').title()}: %{{y}}<br>"
                    "<extra></extra>"
                )
            )
        )
        
        fig.update_layout(
            title=f"2D Analysis: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            height=600
        )
    
    return fig


def create_animated_plot(data: List[Dict[str, Any]], frame_index: int = 0) -> go.Figure:
    """Create an animated visualization showing weather evolution over time.
    
    Args:
        data: List of weather data points
        frame_index: Current frame index for animation
        
    Returns:
        Plotly figure with animation
    """
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    if 'timestamp' not in df.columns:
        return go.Figure().add_annotation(
            text="No timestamp data available for animation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
    
    # Convert timestamp and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create time bins for animation frames
    time_range = df['timestamp'].max() - df['timestamp'].min()
    if time_range.total_seconds() == 0:
        return go.Figure().add_annotation(
            text="Insufficient time range for animation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
    
    # Create hourly bins
    df['hour_bin'] = df['timestamp'].dt.floor('h')
    unique_hours = sorted(df['hour_bin'].unique())
    
    if not unique_hours:
        return go.Figure()
    
    # Calculate frame index based on available data
    max_frames = len(unique_hours)
    actual_frame = min(frame_index, max_frames - 1) if max_frames > 0 else 0
    
    # Get data up to current frame
    current_time = unique_hours[actual_frame]
    current_data = df[df['hour_bin'] <= current_time]
    
    # Create animated temperature evolution
    if 'temperature' in current_data.columns and 'city' in current_data.columns:
        fig = go.Figure()
        
        cities = current_data['city'].unique()
        colors = px.colors.qualitative.Set3
        
        for i, city in enumerate(cities):
            city_data = current_data[current_data['city'] == city]
            
            fig.add_trace(
                go.Scatter(
                    x=city_data['timestamp'],
                    y=city_data['temperature'],
                    mode='lines+markers',
                    name=city,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"<b>{city}</b><br>"
                        "Temperature: %{y}°C<br>"
                        "Time: %{x}<br>"
                        "<extra></extra>"
                    )
                )
            )
        
        # Add current time indicator as an annotation instead
        fig.add_annotation(
            x=current_time,
            y=1,
            xref="x",
            yref="paper",
            text=f"Current: {current_time.strftime('%Y-%m-%d %H:%M')}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red"
        )
        
        fig.update_layout(
            title=f"Temperature Evolution (Frame {actual_frame + 1}/{max_frames})",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            hovermode='x unified',
            height=500
        )
        
    else:
        # Fallback visualization
        fig = px.bar(
            current_data.groupby('city')['temperature'].mean().reset_index() if 'temperature' in current_data.columns else pd.DataFrame(),
            x='city', y='temperature',
            title=f"Average Temperature by City (Frame {actual_frame + 1}/{max_frames})"
        )
        fig.update_layout(height=500)
    
    return fig
