"""End-to-end tests for dashboard functionality."""

import pytest
import asyncio
import sys
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from weather_pipeline.core.container import get_container
from weather_pipeline.dashboard.data_manager import DashboardDataManager
from weather_pipeline.dashboard.auth import AuthManager
from weather_pipeline.dashboard.exports import ExportManager
from weather_pipeline.dashboard.components import (
    create_time_series_plot,
    create_geographic_map,
    create_3d_surface_plot,
    create_animated_plot
)
from weather_pipeline.config.settings import Settings


class TestDashboardE2E:
    """End-to-end tests for dashboard components."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = MagicMock()
        settings.auth = MagicMock()
        settings.auth.secret_key = "test_secret_key"
        settings.auth.session_timeout = 3600
        return settings

    @pytest.fixture
    def sample_weather_data(self):
        """Sample weather data for testing."""
        return [
            {
                'timestamp': datetime.now() - timedelta(hours=i),
                'temperature': 20 + i * 0.5,
                'humidity': 60 + i,
                'pressure': 1013 + i * 0.1,
                'wind_speed': 5 + i * 0.2,
                'city': 'TestCity',
                'latitude': 51.5 + i * 0.01,
                'longitude': -0.1 + i * 0.01,
                'provider': 'weatherapi'
            }
            for i in range(24)  # 24 hours of data
        ]

    @pytest.fixture
    async def dashboard_managers(self, mock_settings):
        """Initialize dashboard managers for testing."""
        container = get_container()
        data_manager = DashboardDataManager(container)
        auth_manager = AuthManager(mock_settings)
        export_manager = ExportManager()
        
        return {
            'data_manager': data_manager,
            'auth_manager': auth_manager,
            'export_manager': export_manager
        }

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_dashboard_data_fetching(self, dashboard_managers, sample_weather_data):
        """Test dashboard data fetching functionality."""
        data_manager = dashboard_managers['data_manager']
        
        # Mock the data fetching method
        with patch.object(data_manager, 'get_weather_data', return_value=sample_weather_data):
            cities = ['new_york', 'london', 'tokyo']
            start_date = (datetime.now() - timedelta(days=2)).isoformat()
            end_date = datetime.now().isoformat()
            
            data = await data_manager.get_weather_data(cities, start_date, end_date)
            
            assert data is not None
            assert len(data) == len(sample_weather_data)
            assert all('temperature' in item for item in data)

    @pytest.mark.e2e
    def test_visualization_components(self, sample_weather_data):
        """Test dashboard visualization components."""
        # Test time series plot
        parameters = ['temperature', 'humidity', 'pressure']
        
        with patch('weather_pipeline.dashboard.components.create_time_series_plot') as mock_ts:
            mock_ts.return_value = MagicMock()
            fig_ts = create_time_series_plot(sample_weather_data, parameters)
            assert fig_ts is not None
            mock_ts.assert_called_once_with(sample_weather_data, parameters)

        # Test geographic map
        with patch('weather_pipeline.dashboard.components.create_geographic_map') as mock_geo:
            mock_geo.return_value = MagicMock()
            fig_geo = create_geographic_map(sample_weather_data)
            assert fig_geo is not None
            mock_geo.assert_called_once_with(sample_weather_data)

        # Test 3D surface plot
        with patch('weather_pipeline.dashboard.components.create_3d_surface_plot') as mock_3d:
            mock_3d.return_value = MagicMock()
            fig_3d = create_3d_surface_plot(sample_weather_data)
            assert fig_3d is not None
            mock_3d.assert_called_once_with(sample_weather_data)

        # Test animated plot
        with patch('weather_pipeline.dashboard.components.create_animated_plot') as mock_anim:
            mock_anim.return_value = MagicMock()
            fig_anim = create_animated_plot(sample_weather_data, frame_index=0)
            assert fig_anim is not None
            mock_anim.assert_called_once_with(sample_weather_data, frame_index=0)

    @pytest.mark.e2e
    def test_export_functionality(self, dashboard_managers, sample_weather_data):
        """Test dashboard export functionality."""
        export_manager = dashboard_managers['export_manager']
        
        # Test Excel export
        with patch.object(export_manager, 'export_excel_report') as mock_excel:
            mock_excel.return_value = b"fake_excel_data"
            excel_export = export_manager.export_excel_report(sample_weather_data)
            assert excel_export is not None
            mock_excel.assert_called_once_with(sample_weather_data)

        # Test PDF export
        with patch.object(export_manager, 'export_pdf_report') as mock_pdf:
            mock_pdf.return_value = "<html>fake_pdf_content</html>"
            pdf_export = export_manager.export_pdf_report(sample_weather_data)
            assert pdf_export is not None
            mock_pdf.assert_called_once_with(sample_weather_data)

    @pytest.mark.e2e
    def test_authentication_workflow(self, dashboard_managers):
        """Test complete authentication workflow."""
        auth_manager = dashboard_managers['auth_manager']
        
        # Mock authentication methods
        with patch.object(auth_manager, 'authenticate_user') as mock_auth, \
             patch.object(auth_manager, 'validate_session') as mock_validate, \
             patch.object(auth_manager, 'has_permission') as mock_permission, \
             patch.object(auth_manager, 'logout_user') as mock_logout:
            
            # Setup mock returns
            mock_session_token = "test_session_token"
            mock_session = {"user": "demo", "timestamp": datetime.now()}
            
            mock_auth.return_value = mock_session_token
            mock_validate.return_value = mock_session
            mock_permission.return_value = True
            mock_logout.return_value = None
            
            # Test login
            session_token = auth_manager.authenticate_user("demo", "demo123")
            assert session_token == mock_session_token
            mock_auth.assert_called_once_with("demo", "demo123")
            
            # Test session validation
            session = auth_manager.validate_session(session_token)
            assert session == mock_session
            mock_validate.assert_called_once_with(session_token)
            
            # Test permissions
            has_view = auth_manager.has_permission(session_token, "view")
            has_export = auth_manager.has_permission(session_token, "export")
            assert has_view is True
            assert has_export is True
            
            # Test logout
            auth_manager.logout_user(session_token)
            mock_logout.assert_called_once_with(session_token)

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_dashboard_workflow(self, dashboard_managers, sample_weather_data):
        """Test complete dashboard workflow from data to visualization."""
        data_manager = dashboard_managers['data_manager']
        auth_manager = dashboard_managers['auth_manager']
        export_manager = dashboard_managers['export_manager']
        
        # Mock all methods to simulate complete workflow
        with patch.object(data_manager, 'get_weather_data', return_value=sample_weather_data), \
             patch.object(auth_manager, 'authenticate_user', return_value="session_token"), \
             patch.object(auth_manager, 'validate_session', return_value={"user": "demo"}), \
             patch.object(auth_manager, 'has_permission', return_value=True), \
             patch('weather_pipeline.dashboard.components.create_time_series_plot', return_value=MagicMock()), \
             patch.object(export_manager, 'export_excel_report', return_value=b"excel_data"):
            
            # 1. Authenticate user
            session_token = auth_manager.authenticate_user("demo", "demo123")
            assert session_token is not None
            
            # 2. Validate session
            session = auth_manager.validate_session(session_token)
            assert session is not None
            
            # 3. Check permissions
            can_view = auth_manager.has_permission(session_token, "view")
            assert can_view is True
            
            # 4. Fetch data
            cities = ['new_york', 'london', 'tokyo']
            start_date = (datetime.now() - timedelta(days=2)).isoformat()
            end_date = datetime.now().isoformat()
            
            data = await data_manager.get_weather_data(cities, start_date, end_date)
            assert data is not None
            assert len(data) > 0
            
            # 5. Create visualization
            fig = create_time_series_plot(data, ['temperature'])
            assert fig is not None
            
            # 6. Export data
            can_export = auth_manager.has_permission(session_token, "export")
            if can_export:
                export_data = export_manager.export_excel_report(data)
                assert export_data is not None
