"""Real-time data updates for the dashboard using WebSocket-like functionality."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

from ..core.container import DIContainer
from .data_manager import DashboardDataManager

logger = logging.getLogger(__name__)


class RealTimeUpdater:
    """Manages real-time data updates for the dashboard."""
    
    def __init__(self, container: DIContainer):
        """Initialize the real-time updater."""
        self.container = container
        self.data_manager = DashboardDataManager(container)
        self.subscribers: Dict[str, Callable] = {}
        self.update_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        
    def subscribe(self, subscriber_id: str, callback: Callable[[List[Dict[str, Any]]], None]) -> None:
        """Subscribe to real-time data updates.
        
        Args:
            subscriber_id: Unique identifier for the subscriber
            callback: Function to call with updated data
        """
        self.subscribers[subscriber_id] = callback
        logger.info(f"Subscriber {subscriber_id} registered for real-time updates")
    
    def unsubscribe(self, subscriber_id: str) -> None:
        """Unsubscribe from real-time data updates.
        
        Args:
            subscriber_id: Unique identifier for the subscriber
        """
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]
            logger.info(f"Subscriber {subscriber_id} unsubscribed from real-time updates")
    
    async def start_real_time_updates(
        self,
        cities: List[str],
        update_interval: int = 30
    ) -> None:
        """Start real-time data updates.
        
        Args:
            cities: List of cities to monitor
            update_interval: Update interval in seconds
        """
        if self.is_running:
            logger.warning("Real-time updates already running")
            return
        
        self.is_running = True
        logger.info(f"Starting real-time updates for cities: {cities}, interval: {update_interval}s")
        
        try:
            while self.is_running:
                # Fetch fresh data
                try:
                    updated_data = await self.data_manager.get_real_time_data(cities)
                    
                    # Notify all subscribers
                    for subscriber_id, callback in self.subscribers.items():
                        try:
                            await self._safe_callback(callback, updated_data)
                        except Exception as e:
                            logger.error(f"Error notifying subscriber {subscriber_id}: {e}")
                    
                    logger.debug(f"Real-time update completed, {len(updated_data)} data points")
                    
                except Exception as e:
                    logger.error(f"Error fetching real-time data: {e}")
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
        except asyncio.CancelledError:
            logger.info("Real-time updates cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in real-time updates: {e}")
        finally:
            self.is_running = False
    
    async def _safe_callback(self, callback: Callable, data: List[Dict[str, Any]]) -> None:
        """Safely execute callback with error handling."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.error(f"Callback execution failed: {e}")
    
    def stop_real_time_updates(self) -> None:
        """Stop real-time data updates."""
        self.is_running = False
        logger.info("Real-time updates stopped")
    
    async def get_alert_data(self, cities: List[str]) -> List[Dict[str, Any]]:
        """Check for weather alerts and extreme conditions.
        
        Args:
            cities: List of cities to check
            
        Returns:
            List of weather alerts
        """
        alerts = []
        
        try:
            # Get current data
            current_data = await self.data_manager.get_real_time_data(cities)
            
            for data_point in current_data:
                city_alerts = self._check_extreme_conditions(data_point)
                alerts.extend(city_alerts)
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
        
        return alerts
    
    def _check_extreme_conditions(self, data_point: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for extreme weather conditions in a data point.
        
        Args:
            data_point: Weather data point
            
        Returns:
            List of alerts for extreme conditions
        """
        alerts = []
        city = data_point.get('city', 'Unknown')
        
        # Temperature alerts
        temp = data_point.get('temperature')
        if temp is not None:
            if temp > 40:
                alerts.append({
                    'type': 'extreme_heat',
                    'severity': 'high',
                    'city': city,
                    'message': f"Extreme heat warning: {temp}°C",
                    'timestamp': datetime.utcnow().isoformat(),
                    'value': temp
                })
            elif temp < -20:
                alerts.append({
                    'type': 'extreme_cold',
                    'severity': 'high',
                    'city': city,
                    'message': f"Extreme cold warning: {temp}°C",
                    'timestamp': datetime.utcnow().isoformat(),
                    'value': temp
                })
        
        # Wind speed alerts
        wind_speed = data_point.get('wind_speed')
        if wind_speed is not None and wind_speed > 20:
            alerts.append({
                'type': 'high_wind',
                'severity': 'medium',
                'city': city,
                'message': f"High wind warning: {wind_speed} m/s",
                'timestamp': datetime.utcnow().isoformat(),
                'value': wind_speed
            })
        
        # Precipitation alerts
        precipitation = data_point.get('precipitation')
        if precipitation is not None and precipitation > 10:
            alerts.append({
                'type': 'heavy_rain',
                'severity': 'medium',
                'city': city,
                'message': f"Heavy precipitation: {precipitation} mm",
                'timestamp': datetime.utcnow().isoformat(),
                'value': precipitation
            })
        
        # Visibility alerts
        visibility = data_point.get('visibility')
        if visibility is not None and visibility < 1:
            alerts.append({
                'type': 'low_visibility',
                'severity': 'medium',
                'city': city,
                'message': f"Low visibility: {visibility} km",
                'timestamp': datetime.utcnow().isoformat(),
                'value': visibility
            })
        
        return alerts
    
    async def start_alert_monitoring(self, cities: List[str], check_interval: int = 60) -> None:
        """Start monitoring for weather alerts.
        
        Args:
            cities: List of cities to monitor
            check_interval: Check interval in seconds
        """
        logger.info(f"Starting alert monitoring for cities: {cities}")
        
        try:
            while self.is_running:
                alerts = await self.get_alert_data(cities)
                
                if alerts:
                    logger.warning(f"Weather alerts detected: {len(alerts)} alerts")
                    # In a full implementation, you'd send these to a notification system
                    for alert in alerts:
                        logger.warning(f"ALERT - {alert['city']}: {alert['message']}")
                
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            logger.info("Alert monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in alert monitoring: {e}")


class DashboardMetrics:
    """Collect and manage dashboard performance metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {
            'page_loads': 0,
            'data_requests': 0,
            'chart_renders': 0,
            'export_requests': 0,
            'last_update': None,
            'active_users': 0,
            'error_count': 0
        }
        self.start_time = datetime.utcnow()
    
    def increment_metric(self, metric_name: str, value: int = 1) -> None:
        """Increment a metric counter.
        
        Args:
            metric_name: Name of the metric
            value: Value to increment by
        """
        if metric_name in self.metrics:
            self.metrics[metric_name] += value
        else:
            self.metrics[metric_name] = value
        
        self.metrics['last_update'] = datetime.utcnow().isoformat()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary of current metrics
        """
        uptime = datetime.utcnow() - self.start_time
        
        return {
            **self.metrics,
            'uptime_seconds': uptime.total_seconds(),
            'uptime_hours': uptime.total_seconds() / 3600
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = {key: 0 for key in self.metrics}
        self.metrics['last_update'] = None
        self.start_time = datetime.utcnow()
