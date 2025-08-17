"""Main dashboard application using Dash."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import dash
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
            # Login overlay
            html.Div(
                id='login-overlay',
                children=[
                    html.Div([
                        html.H2("Weather Dashboard Login", style={"text-align": "center", "margin-bottom": "20px"}),
                        html.Div([
                            html.Label("Username:", style={"display": "block", "margin-bottom": "5px"}),
                            dcc.Input(
                                id="username-input",
                                type="text",
                                placeholder="Enter username",
                                style={"width": "100%", "padding": "8px", "margin-bottom": "15px", "border": "1px solid #ddd", "border-radius": "4px"}
                            ),
                            html.Label("Password:", style={"display": "block", "margin-bottom": "5px"}),
                            dcc.Input(
                                id="password-input",
                                type="password",
                                placeholder="Enter password",
                                style={"width": "100%", "padding": "8px", "margin-bottom": "20px", "border": "1px solid #ddd", "border-radius": "4px"}
                            ),
                            html.Button(
                                "Login",
                                id="login-button",
                                style={"width": "100%", "padding": "10px", "background-color": "#007bff", "color": "white", "border": "none", "border-radius": "4px", "cursor": "pointer"}
                            ),
                            html.Div(id="login-error", style={"color": "red", "margin-top": "10px", "text-align": "center"}),
                            html.Div([
                                html.P("Demo accounts:", style={"margin-top": "20px", "font-size": "12px", "color": "#666"}),
                                html.Ul([
                                    html.Li("admin / admin123 (full access)"),
                                    html.Li("viewer / viewer123 (view only)"),
                                    html.Li("demo / demo123 (view only)")
                                ], style={"font-size": "11px", "color": "#666"})
                            ])
                        ])
                    ], style={
                        "max-width": "400px",
                        "margin": "100px auto",
                        "padding": "30px",
                        "border": "1px solid #ddd",
                        "border-radius": "8px",
                        "background-color": "white",
                        "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)"
                    })
                ],
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "width": "100%",
                    "height": "100%",
                    "background-color": "rgba(0, 0, 0, 0.5)",
                    "z-index": "1000",
                    "display": "block"
                }
            ),
            
            # Main dashboard content (hidden until login)
            html.Div(
                id='main-dashboard',
                children=[
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
                ],
                style={"display": "none"}
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
            # City selection with search
            html.H4("Location Selection", style={"color": "#34495e"}),
            dcc.Dropdown(
                id='city-dropdown',
                options=[
                    {'label': 'New York, NY', 'value': 'new_york'},
                    {'label': 'London, UK', 'value': 'london'},
                    {'label': 'Tokyo, JP', 'value': 'tokyo'},
                    {'label': 'Sydney, AU', 'value': 'sydney'},
                    {'label': 'Paris, FR', 'value': 'paris'},
                    {'label': 'Berlin, DE', 'value': 'berlin'},
                    {'label': 'Moscow, RU', 'value': 'moscow'},
                    {'label': 'Beijing, CN', 'value': 'beijing'},
                    {'label': 'Mumbai, IN', 'value': 'mumbai'},
                    {'label': 'Rio de Janeiro, BR', 'value': 'rio_de_janeiro'},
                    {'label': 'Los Angeles, CA', 'value': 'los_angeles'},
                    {'label': 'Mexico City, MX', 'value': 'mexico_city'},
                    {'label': 'Toronto, CA', 'value': 'toronto'},
                    {'label': 'Amsterdam, NL', 'value': 'amsterdam'},
                    {'label': 'Stockholm, SE', 'value': 'stockholm'},
                    {'label': 'Rome, IT', 'value': 'rome'},
                    {'label': 'Madrid, ES', 'value': 'madrid'},
                    {'label': 'Vienna, AT', 'value': 'vienna'},
                    {'label': 'Prague, CZ', 'value': 'prague'},
                    {'label': 'Warsaw, PL', 'value': 'warsaw'},
                    {'label': 'Bangkok, TH', 'value': 'bangkok'},
                    {'label': 'Singapore, SG', 'value': 'singapore'},
                    {'label': 'Seoul, KR', 'value': 'seoul'},
                    {'label': 'Hong Kong, HK', 'value': 'hong_kong'},
                    {'label': 'Dubai, AE', 'value': 'dubai'},
                    {'label': 'Istanbul, TR', 'value': 'istanbul'},
                    {'label': 'Cairo, EG', 'value': 'cairo'},
                    {'label': 'Cape Town, ZA', 'value': 'cape_town'},
                    {'label': 'Buenos Aires, AR', 'value': 'buenos_aires'},
                    {'label': 'Santiago, CL', 'value': 'santiago'},
                ],
                value=['new_york', 'london'],
                multi=True,
                searchable=True,
                placeholder="Search and select cities...",
                style={"margin-bottom": "20px"}
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
            
            # Weather data source selection
            html.H4("Data Source", style={"color": "#34495e"}),
            dcc.Dropdown(
                id='source-dropdown',
                options=[
                    {'label': 'Mock Data (Demo)', 'value': 'mock'},
                    {'label': '7timer API', 'value': 'seven_timer'},
                    {'label': 'WeatherAPI', 'value': 'weatherapi'},
                    {'label': 'OpenWeatherMap', 'value': 'openweather'},
                ],
                value='mock',
                searchable=False,
                placeholder="Select data source...",
                style={"margin-bottom": "20px"}
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
                            html.Button("▶️", id="play-pause-btn", 
                                      style={"margin-right": "10px", "font-size": "20px", "border": "none", "background": "transparent", "cursor": "pointer"}),
                            dcc.Slider(
                                id="animation-slider",
                                min=0,
                                max=100,
                                value=0,
                                marks={i: str(i) for i in range(0, 101, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={"padding": "20px"}),
                        # Animation control interval
                        dcc.Interval(
                            id='animation-interval',
                            interval=1000,  # Update every second
                            n_intervals=0,
                            disabled=True
                        ),
                        # Store for animation state
                        dcc.Store(id='animation-state', data={'playing': False, 'frame': 0})
                    ], style={"padding": "20px"})
                ]),
            ])
        ])
        
    def _setup_callbacks(self) -> None:
        """Set up all dashboard callbacks."""
        
        @self.app.callback(
            [Output('login-overlay', 'style'),
             Output('main-dashboard', 'style'),
             Output('login-error', 'children'),
             Output('user-session-store', 'data')],
            [Input('login-button', 'n_clicks')],
            [State('username-input', 'value'),
             State('password-input', 'value')],
            prevent_initial_call=True
        )
        def handle_login(n_clicks, username, password):
            """Handle user login."""
            if not n_clicks or not username or not password:
                raise PreventUpdate
            
            # Authenticate user
            session_token = self.auth_manager.authenticate_user(username, password)
            
            if session_token:
                # Hide login overlay, show dashboard
                return (
                    {"display": "none"}, 
                    {"display": "block"},
                    "",
                    {"session_token": session_token, "username": username}
                )
            else:
                # Show error, keep login visible
                return (
                    dash.no_update,
                    dash.no_update,
                    "Invalid username or password. Please try again.",
                    dash.no_update
                )
        
        @self.app.callback(
            [Output('animation-state', 'data'),
             Output('play-pause-btn', 'children'),
             Output('animation-interval', 'disabled')],
            [Input('play-pause-btn', 'n_clicks')],
            [State('animation-state', 'data')],
            prevent_initial_call=True
        )
        def toggle_animation(n_clicks, current_state):
            """Toggle animation play/pause."""
            if not n_clicks:
                raise PreventUpdate
            
            playing = not current_state.get('playing', False)
            button_text = "⏸️" if playing else "▶️"
            interval_disabled = not playing
            
            return (
                {'playing': playing, 'frame': current_state.get('frame', 0)},
                button_text,
                interval_disabled
            )
        
        @self.app.callback(
            [Output('animation-slider', 'value'),
             Output('animation-state', 'data', allow_duplicate=True)],
            [Input('animation-interval', 'n_intervals')],
            [State('animation-state', 'data'),
             State('weather-data-store', 'data')],
            prevent_initial_call=True
        )
        def update_animation_frame(n_intervals, animation_state, data):
            """Update animation frame automatically."""
            if not animation_state.get('playing', False) or not data:
                raise PreventUpdate
            
            # Calculate max frames based on data
            max_frames = min(100, len(data)) if data else 100
            current_frame = animation_state.get('frame', 0)
            
            # Advance frame
            next_frame = (current_frame + 1) % max_frames
            
            return (
                next_frame,
                {'playing': animation_state['playing'], 'frame': next_frame}
            )
        
        @self.app.callback(
            Output('animation-state', 'data', allow_duplicate=True),
            [Input('animation-slider', 'value')],
            [State('animation-state', 'data')],
            prevent_initial_call=True
        )
        def update_animation_state_from_slider(slider_value, animation_state):
            """Update animation state when slider is moved manually."""
            return {
                'playing': animation_state.get('playing', False),
                'frame': slider_value
            }
        
        @self.app.callback(
            Output('weather-data-store', 'data'),
            [Input('interval-component', 'n_intervals'),
             Input('city-dropdown', 'value'),
             Input('date-picker-range', 'start_date'),
             Input('date-picker-range', 'end_date'),
             Input('source-dropdown', 'value')]
        )
        def update_data_store(n_intervals, selected_cities, start_date, end_date, source):
            """Update the data store with fresh weather data."""
            if not selected_cities:
                raise PreventUpdate
                
            try:
                # TODO: In future, use different data sources based on 'source' parameter
                # For now, all sources use mock data
                logger.info(f"Fetching data from source: {source}")
                
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
             Input('animation-state', 'data')]
        )
        def update_animated_plot(data, animation_state):
            """Update the animated plot."""
            if not data:
                return go.Figure()
                
            frame_index = animation_state.get('frame', 0) if animation_state else 0
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
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            
            if not data:
                logger.warning("No data available for export")
                raise PreventUpdate
                
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            try:
                if button_id == 'export-pdf-btn':
                    logger.info("Exporting PDF report...")
                    result = self.export_manager.export_pdf_report(data)
                    if result:
                        logger.info(f"PDF export successful: {result.get('filename')}")
                        return result
                    else:
                        logger.error("PDF export returned empty result")
                        raise PreventUpdate
                        
                elif button_id == 'export-excel-btn':
                    logger.info("Exporting Excel report...")
                    result = self.export_manager.export_excel_report(data)
                    if result:
                        logger.info(f"Excel export successful: {result.get('filename')}")
                        return result
                    else:
                        logger.error("Excel export returned empty result")
                        raise PreventUpdate
                        
            except Exception as e:
                logger.error(f"Export failed for {button_id}: {e}")
                raise PreventUpdate
            
            raise PreventUpdate

    def run(self, debug: bool = False, host: str = "127.0.0.1", port: int = 8050) -> None:
        """Run the dashboard application."""
        logger.info(f"Starting dashboard on {host}:{port}")
        self.app.run(debug=debug, host=host, port=port)


def create_dashboard_app(container: DIContainer) -> WeatherDashboard:
    """Factory function to create a dashboard application."""
    return WeatherDashboard(container)
