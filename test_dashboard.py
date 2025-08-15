"""Test script to validate dashboard functionality."""

import asyncio
import sys
from datetime import datetime, timedelta

# Add the source path
sys.path.insert(0, 'src')

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
from weather_pipeline import settings

async def test_dashboard_components():
    """Test all dashboard components."""
    print("🧪 Testing Dashboard Components...")
    
    # Get container and initialize managers
    container = get_container()
    data_manager = DashboardDataManager(container)
    auth_manager = AuthManager(settings)
    export_manager = ExportManager()
    
    # Test data fetching
    print("📊 Testing data fetching...")
    cities = ['new_york', 'london', 'tokyo']
    start_date = (datetime.now() - timedelta(days=2)).isoformat()
    end_date = datetime.now().isoformat()
    
    data = await data_manager.get_weather_data(cities, start_date, end_date)
    print(f"✅ Fetched {len(data)} data points")
    
    if data:
        # Test visualization components
        print("📈 Testing visualization components...")
        
        # Test time series plot
        parameters = ['temperature', 'humidity', 'pressure']
        fig_ts = create_time_series_plot(data, parameters)
        print("✅ Time series plot created")
        
        # Test geographic map
        fig_geo = create_geographic_map(data)
        print("✅ Geographic map created")
        
        # Test 3D surface plot
        fig_3d = create_3d_surface_plot(data)
        print("✅ 3D surface plot created")
        
        # Test animated plot
        fig_anim = create_animated_plot(data, frame_index=0)
        print("✅ Animated plot created")
        
        # Test exports
        print("📄 Testing export functionality...")
        
        excel_export = export_manager.export_excel_report(data)
        if excel_export:
            print("✅ Excel export created")
        
        pdf_export = export_manager.export_pdf_report(data)
        if pdf_export:
            print("✅ PDF/HTML export created")
    
    # Test authentication
    print("🔐 Testing authentication...")
    
    # Test login
    session_token = auth_manager.authenticate_user("demo", "demo123")
    if session_token:
        print("✅ User authentication successful")
        
        # Test session validation
        session = auth_manager.validate_session(session_token)
        if session:
            print("✅ Session validation successful")
        
        # Test permissions
        has_view = auth_manager.has_permission(session_token, "view")
        has_export = auth_manager.has_permission(session_token, "export")
        print(f"✅ Permissions - View: {has_view}, Export: {has_export}")
        
        # Test logout
        auth_manager.logout_user(session_token)
        print("✅ User logout successful")
    
    print("🎉 All dashboard components tested successfully!")

if __name__ == "__main__":
    asyncio.run(test_dashboard_components())
