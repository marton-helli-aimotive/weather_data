"""Main dashboard application using Dash."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from ..config.settings import Settings
from .. import settings
from ..core.container import DIContainer
from .components import (
    create_time_series_plot,
    create_geographic_map,
    create_3d_surface_plot,
    create_animated_plot,
)
from .auth import AuthManager
from .data_manager import DashboardDataManager
from .exports import ExportManager

logger = logging.getLogger(__name__)


class WeatherDashboard:
    """Main weather dashboard class."""
    
    def __init__(self, container: DIContainer):
        """Initialize the dashboard."""
        self.container = container
        self.settings = container.get(Settings) if container.get_optional(Settings) else settings
        self.auth_manager = AuthManager(self.settings)
        self.data_manager = DashboardDataManager(container)
        self.export_manager = ExportManager()
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                'https://codepen.io/chriddyp/pen/bWLwgP.css',
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
            ],
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
        )
        
        self.app.title = "Weather Data Dashboard"
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self) -> None:
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            self._create_header(),
            
            # Main content
            html.Div([
                # Sidebar for controls
                html.Div([
                    self._create_sidebar()
                ], className="three columns", style={"padding": "20px"}),
                
                # Main dashboard area
                html.Div([
                    self._create_main_content()
                ], className="nine columns")
            ], className="row"),
            
            # Interval component for real-time updates
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            ),
            
            # Store components for data
            dcc.Store(id='weather-data-store'),
            dcc.Store(id='user-session-store'),
        ])
    
    def _create_header(self) -> html.Div:
        """Create the dashboard header."""
        return html.Div([
            html.H1(
                [
                    html.I(className="fas fa-cloud-sun", style={"margin-right": "10px"}),
                    "Weather Data Dashboard"
                ],
                className="header-title",
                style={
                    "color": "#2c3e50",
                    "text-align": "center",
                    "margin-bottom": "20px",
                    "padding": "20px"
                }
            ),
            
            # Status indicator
            html.Div([
                html.Span("●", id="status-indicator", style={"color": "green", "font-size": "20px"}),
                html.Span("Live Data", style={"margin-left": "5px", "font-weight": "bold"})
            ], style={"text-align": "center", "margin-bottom": "20px"})
        ])
    
    def _create_sidebar(self) -> html.Div:
        """Create the sidebar with controls."""
        return html.Div([
            # City selection
            html.H4("Location Selection", style={"color": "#34495e"}),
            dcc.Dropdown(
                id='city-dropdown',
                options=[
                    {'label': 'New York, NY', 'value': 'new_york'},
                    {'label': 'London, UK', 'value': 'london'},
                    {'label': 'Tokyo, JP', 'value': 'tokyo'},
                    {'label': 'Sydney, AU', 'value': 'sydney'},
                    {'label': 'Paris, FR', 'value': 'paris'},
                ],
                value=['new_york', 'london'],
                multi=True,
                placeholder="Select cities..."
            ),
            
            html.Hr(),
            
            # Time range selection
            html.H4("Time Range", style={"color": "#34495e"}),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now(),
                display_format='YYYY-MM-DD'
            ),
            
            html.Hr(),
            
            # Weather parameters
            html.H4("Parameters", style={"color": "#34495e"}),
            dcc.Checklist(
                id='parameter-checklist',
                options=[
                    {'label': 'Temperature', 'value': 'temperature'},
                    {'label': 'Humidity', 'value': 'humidity'},
                    {'label': 'Pressure', 'value': 'pressure'},
                    {'label': 'Wind Speed', 'value': 'wind_speed'},
                    {'label': 'Precipitation', 'value': 'precipitation'},
                ],
                value=['temperature', 'humidity', 'pressure'],
                style={"margin-bottom": "20px"}
            ),
            
            html.Hr(),
            
            # Export options
            html.H4("Export", style={"color": "#34495e"}),
            html.Button(
                [html.I(className="fas fa-file-pdf"), " Export PDF"],
                id="export-pdf-btn",
                className="button-primary",
                style={"width": "100%", "margin-bottom": "10px"}
            ),
            html.Button(
                [html.I(className="fas fa-file-excel"), " Export Excel"],
                id="export-excel-btn", 
                className="button-primary",
                style={"width": "100%"}
            ),
            
            # Download component
            dcc.Download(id="download-component")
        ])
    
    def _create_main_content(self) -> html.Div:
        """Create the main content area."""
        return html.Div([
            # Tab component for different views
            dcc.Tabs(id="main-tabs", value="time-series", children=[
                dcc.Tab(label="Time Series", value="time-series", children=[
                    html.Div([
                        dcc.Graph(id='time-series-graph')
                    ], style={"padding": "20px"})
                ]),
                
                dcc.Tab(label="Geographic Map", value="geographic", children=[
                    html.Div([
                        dcc.Graph(id='geographic-map')
                    ], style={"padding": "20px"})
                ]),
                
                dcc.Tab(label="3D Analysis", value="3d-analysis", children=[
                    html.Div([
                        dcc.Graph(id='3d-surface-plot')
                    ], style={"padding": "20px"})
                ]),
                
                dcc.Tab(label="Animation", value="animation", children=[
                    html.Div([
                        dcc.Graph(id='animated-plot'),
                        html.Div([
                            html.Button("⏸️", id="play-pause-btn", style={"margin-right": "10px"}),
                            dcc.Slider(
                                id="animation-slider",
                                min=0,
                                max=100,
                                value=0,
                                marks={i: str(i) for i in range(0, 101, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={"padding": "20px"})
                    ], style={"padding": "20px"})
                ]),
            ])
        ])
        
    def _setup_callbacks(self) -> None:
        """Set up all dashboard callbacks."""
        
        @self.app.callback(
            Output('weather-data-store', 'data'),
            [Input('interval-component', 'n_intervals'),
             Input('city-dropdown', 'value'),
             Input('date-picker-range', 'start_date'),
             Input('date-picker-range', 'end_date')]
        )
        def update_data_store(n_intervals, selected_cities, start_date, end_date):
            """Update the data store with fresh weather data."""
            if not selected_cities:
                raise PreventUpdate
                
            try:
                # Fetch data using the data manager
                data = asyncio.run(self.data_manager.get_weather_data(
                    cities=selected_cities,
                    start_date=start_date,
                    end_date=end_date
                ))
                return data
            except Exception as e:
                logger.error(f"Failed to fetch weather data: {e}")
                return []
        
        @self.app.callback(
            Output('time-series-graph', 'figure'),
            [Input('weather-data-store', 'data'),
             Input('parameter-checklist', 'value')]
        )
        def update_time_series(data, selected_parameters):
            """Update the time series plot."""
            if not data or not selected_parameters:
                return go.Figure()
                
            return create_time_series_plot(data, selected_parameters)
        
        @self.app.callback(
            Output('geographic-map', 'figure'),
            [Input('weather-data-store', 'data')]
        )
        def update_geographic_map(data):
            """Update the geographic map."""
            if not data:
                return go.Figure()
                
            return create_geographic_map(data)
        
        @self.app.callback(
            Output('3d-surface-plot', 'figure'),
            [Input('weather-data-store', 'data')]
        )
        def update_3d_plot(data):
            """Update the 3D surface plot."""
            if not data:
                return go.Figure()
                
            return create_3d_surface_plot(data)
        
        @self.app.callback(
            Output('animated-plot', 'figure'),
            [Input('weather-data-store', 'data'),
             Input('animation-slider', 'value')]
        )
        def update_animated_plot(data, frame_index):
            """Update the animated plot."""
            if not data:
                return go.Figure()
                
            return create_animated_plot(data, frame_index)
        
        @self.app.callback(
            Output('download-component', 'data'),
            [Input('export-pdf-btn', 'n_clicks'),
             Input('export-excel-btn', 'n_clicks')],
            [State('weather-data-store', 'data')],
            prevent_initial_call=True
        )
        def export_data(pdf_clicks, excel_clicks, data):
            """Handle data export."""
            if not data:
                raise PreventUpdate
                
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'export-pdf-btn':
                return self.export_manager.export_pdf_report(data)
            elif button_id == 'export-excel-btn':
                return self.export_manager.export_excel_report(data)
            
            raise PreventUpdate

    def run(self, debug: bool = False, host: str = "127.0.0.1", port: int = 8050) -> None:
        """Run the dashboard application."""
        logger.info(f"Starting dashboard on {host}:{port}")
        self.app.run(debug=debug, host=host, port=port)


def create_dashboard_app(container: DIContainer) -> WeatherDashboard:
    """Factory function to create a dashboard application."""
    return WeatherDashboard(container)
