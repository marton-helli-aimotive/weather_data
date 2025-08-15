"""Dependency injection container for the weather pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class DependencyInjectionError(Exception):
    """Raised when dependency injection fails."""


class DIContainer:
    """Simple dependency injection container."""
    
    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """Register a singleton instance."""
        key = self._get_key(service_type)
        self._singletons[key] = instance
    
    def register_factory(
        self, 
        service_type: Type[T], 
        factory: Callable[[], T]
    ) -> None:
        """Register a factory function."""
        key = self._get_key(service_type)
        self._factories[key] = factory
    
    def register_transient(self, service_type: Type[T], implementation: Type[T]) -> None:
        """Register a transient service (new instance each time)."""
        key = self._get_key(service_type)
        self._factories[key] = implementation
    
    def get(self, service_type: Type[T]) -> T:
        """Get an instance of the requested service."""
        key = self._get_key(service_type)
        
        # Check if singleton exists
        if key in self._singletons:
            return self._singletons[key]
        
        # Check if factory exists
        if key in self._factories:
            factory = self._factories[key]
            if callable(factory):
                instance = factory()
                return instance
        
        raise DependencyInjectionError(
            f"No registration found for service type: {service_type.__name__}"
        )
    
    def get_optional(self, service_type: Type[T]) -> Optional[T]:
        """Get an instance or None if not registered."""
        try:
            return self.get(service_type)
        except DependencyInjectionError:
            return None
    
    def _get_key(self, service_type: Type[Any]) -> str:
        """Get a string key for the service type."""
        return f"{service_type.__module__}.{service_type.__qualname__}"
    
    def clear(self) -> None:
        """Clear all registrations (mainly for testing)."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global dependency injection container."""
    return _container


def register_singleton(service_type: Type[T], instance: T) -> None:
    """Register a singleton in the global container."""
    _container.register_singleton(service_type, instance)


def register_factory(service_type: Type[T], factory: Callable[[], T]) -> None:
    """Register a factory in the global container."""
    _container.register_factory(service_type, factory)


def register_transient(service_type: Type[T], implementation: Type[T]) -> None:
    """Register a transient service in the global container."""
    _container.register_transient(service_type, implementation)


def get_service(service_type: Type[T]) -> T:
    """Get a service from the global container."""
    return _container.get(service_type)


def get_optional_service(service_type: Type[T]) -> Optional[T]:
    """Get an optional service from the global container."""
    return _container.get_optional(service_type)


def register_processing_services() -> None:
    """Register all processing-related services."""
    from ..processing.time_series import TimeSeriesAnalyzer
    from ..processing.geospatial import GeospatialAnalyzer
    from ..processing.feature_engineering import FeatureEngineer
    from ..processing.data_quality import DataQualityMonitor
    from ..processing.streaming import StreamProcessor, StreamingPipeline
    from ..processing.cache_manager import CacheManager, MemoryCache
    from ..processing.performance import PerformanceComparator
    
    # Register processing services as transients (new instance each time)
    register_transient(TimeSeriesAnalyzer, TimeSeriesAnalyzer)
    register_transient(GeospatialAnalyzer, GeospatialAnalyzer)
    register_transient(FeatureEngineer, FeatureEngineer)
    register_transient(DataQualityMonitor, DataQualityMonitor)
    register_transient(PerformanceComparator, PerformanceComparator)
    
    # Register streaming services
    register_factory(StreamProcessor, lambda: StreamProcessor())
    register_factory(StreamingPipeline, lambda: StreamingPipeline())
    
    # Register cache services
    register_factory(CacheManager, lambda: CacheManager(MemoryCache()))
