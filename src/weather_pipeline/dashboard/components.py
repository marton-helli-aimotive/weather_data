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
    """Create an interactive geographic map with weather data.
    
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
        # Default coordinates for common cities
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
    
    # Add temperature heatmap if available
    if 'temperature' in latest_data.columns:
        fig.add_trace(
            go.Scattermapbox(
                lat=latest_data['lat'],
                lon=latest_data['lon'],
                mode='markers',
                marker=dict(
                    size=20,
                    color=latest_data['temperature'],
                    colorscale='RdYlBu_r',
                    colorbar=dict(
                        title="Temperature (°C)",
                        x=0.02
                    ),
                    sizemode='diameter'
                ),
                text=latest_data['city'],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Temperature: %{marker.color}°C<br>"
                    "Lat: %{lat}<br>"
                    "Lon: %{lon}<br>"
                    "<extra></extra>"
                ),
                name="Temperature"
            )
        )
    else:
        # Just show markers if no temperature data
        fig.add_trace(
            go.Scattermapbox(
                lat=latest_data['lat'],
                lon=latest_data['lon'],
                mode='markers',
                marker=dict(size=15, color='blue'),
                text=latest_data['city'],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Lat: %{lat}<br>"
                    "Lon: %{lon}<br>"
                    "<extra></extra>"
                ),
                name="Cities"
            )
        )
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=latest_data['lat'].mean(),
                lon=latest_data['lon'].mean()
            ),
            zoom=2
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        title={
            'text': "Weather Data Geographic Distribution",
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    return fig


def create_3d_surface_plot(data: List[Dict[str, Any]]) -> go.Figure:
    """Create a 3D surface plot showing relationships between weather parameters.
    
    Args:
        data: List of weather data points
        
    Returns:
        Plotly figure with 3D surface visualization
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
    clean_df = df[available_cols].dropna()
    
    if clean_df.empty or len(clean_df) < 3:
        return go.Figure().add_annotation(
            text="Not enough valid data points for 3D visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
    
    # Create 3D surface plot
    if len(available_cols) >= 3:
        # Use temperature, pressure, humidity
        x = clean_df['temperature']
        y = clean_df['pressure']
        z = clean_df['humidity']
        
        # Create a mesh grid for surface
        x_range = np.linspace(x.min(), x.max(), 20)
        y_range = np.linspace(y.min(), y.max(), 20)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Interpolate Z values
        from scipy.interpolate import griddata
        points = np.column_stack((x.values, y.values))
        Z = griddata(points, z.values, (X, Y), method='linear', fill_value=z.mean())
        
        fig = go.Figure(data=[
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                colorbar=dict(title="Humidity (%)")
            )
        ])
        
        # Add scatter points for actual data
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=5,
                    color=z,
                    colorscale='Viridis',
                    opacity=0.8
                ),
                name="Data Points",
                hovertemplate=(
                    "Temperature: %{x}°C<br>"
                    "Pressure: %{y} hPa<br>"
                    "Humidity: %{z}%<br>"
                    "<extra></extra>"
                )
            )
        )
        
        fig.update_layout(
            title="3D Weather Parameter Relationships",
            scene=dict(
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (hPa)",
                zaxis_title="Humidity (%)",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            height=600
        )
        
    else:
        # Fallback to 2D scatter if only 2 parameters available
        x_col, y_col = available_cols[:2]
        fig = px.scatter(
            clean_df, x=x_col, y=y_col,
            title=f"2D Relationship: {x_col.title()} vs {y_col.title()}",
            labels={
                x_col: f"{x_col.replace('_', ' ').title()}",
                y_col: f"{y_col.replace('_', ' ').title()}"
            }
        )
        fig.update_layout(height=600)
    
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
