"""Health check endpoints and monitoring for the weather pipeline."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any, Dict, List

import psutil
from pydantic import BaseModel

from ..config.settings import get_settings
from ..core.container import get_container
from ..core.logging import get_logger


class HealthStatus(BaseModel):
    """Health check status model."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: float
    uptime: float
    version: str
    checks: Dict[str, Any]


class HealthChecker:
    """Health monitoring and checks for the application."""
    
    def __init__(self):
        self.logger = get_logger("health")
        self.settings = get_settings()
        self.start_time = time.time()
    
    async def check_health(self) -> HealthStatus:
        """Perform comprehensive health check."""
        checks = {}
        overall_status = "healthy"
        
        # Basic system checks
        try:
            checks["system"] = await self._check_system()
        except Exception as e:
            checks["system"] = {"status": "error", "error": str(e)}
            overall_status = "degraded"
        
        # Configuration check
        try:
            checks["config"] = await self._check_config()
        except Exception as e:
            checks["config"] = {"status": "error", "error": str(e)}
            overall_status = "degraded"
        
        # Dependency injection check
        try:
            checks["container"] = await self._check_container()
        except Exception as e:
            checks["container"] = {"status": "error", "error": str(e)}
            overall_status = "degraded"
        
        # External dependencies check
        try:
            checks["external"] = await self._check_external_dependencies()
        except Exception as e:
            checks["external"] = {"status": "error", "error": str(e)}
            overall_status = "unhealthy"
        
        # Determine overall status
        if any(check.get("status") == "error" for check in checks.values()):
            if checks.get("external", {}).get("status") == "error":
                overall_status = "unhealthy"
            else:
                overall_status = "degraded"
        
        return HealthStatus(
            status=overall_status,
            timestamp=time.time(),
            uptime=time.time() - self.start_time,
            version=self.settings.app_version,
            checks=checks
        )
    
    async def _check_system(self) -> Dict[str, Any]:
        """Check system resources and status."""
        try:
            # Get system information
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check if resources are within acceptable limits
            status = "ok"
            warnings = []
            
            if cpu_percent > 90:
                warnings.append(f"High CPU usage: {cpu_percent}%")
                status = "warning"
            
            if memory.percent > 90:
                warnings.append(f"High memory usage: {memory.percent}%")
                status = "warning"
            
            if disk.percent > 90:
                warnings.append(f"High disk usage: {disk.percent}%")
                status = "warning"
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "warnings": warnings
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_config(self) -> Dict[str, Any]:
        """Check configuration validity."""
        try:
            # Verify critical configuration
            issues = []
            
            # Check API keys
            if not self.settings.api.weatherapi_key and not self.settings.api.openweather_api_key:
                issues.append("No API keys configured")
            
            # Check data directory
            if not self.settings.data_dir.exists():
                issues.append("Data directory does not exist")
            
            status = "ok" if not issues else "warning"
            
            return {
                "status": status,
                "environment": self.settings.environment.value,
                "debug": self.settings.debug,
                "issues": issues
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_container(self) -> Dict[str, Any]:
        """Check dependency injection container."""
        try:
            container = get_container()
            
            # Test container resolution of key services
            services = []
            
            # Add more service checks as they become available
            # Example: container.cache_manager()
            
            return {
                "status": "ok",
                "services_available": len(services),
                "container_initialized": True
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_external_dependencies(self) -> Dict[str, Any]:
        """Check external service dependencies."""
        try:
            dependencies = {}
            
            # Check API endpoints
            if self.settings.api.weatherapi_key:
                dependencies["weatherapi"] = await self._check_api_endpoint(
                    "https://api.weatherapi.com/v1/current.json",
                    {"key": self.settings.api.weatherapi_key, "q": "London"}
                )
            
            if self.settings.api.openweather_api_key:
                dependencies["openweather"] = await self._check_api_endpoint(
                    "https://api.openweathermap.org/data/2.5/weather",
                    {"appid": self.settings.api.openweather_api_key, "q": "London"}
                )
            
            # Determine overall external status
            if not dependencies:
                status = "warning"
                message = "No external APIs configured"
            elif all(dep["status"] == "ok" for dep in dependencies.values()):
                status = "ok"
                message = "All external dependencies available"
            else:
                status = "degraded"
                message = "Some external dependencies unavailable"
            
            return {
                "status": status,
                "message": message,
                "dependencies": dependencies
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _check_api_endpoint(self, url: str, params: Dict[str, str]) -> Dict[str, Any]:
        """Check if an API endpoint is accessible."""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return {"status": "ok", "response_time": response.headers.get("X-Response-Time")}
                    else:
                        return {"status": "error", "http_status": response.status}
        except asyncio.TimeoutError:
            return {"status": "error", "error": "timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def readiness_check(self) -> bool:
        """Simple readiness check for container orchestration."""
        try:
            # Check if application is ready to serve requests
            return (
                self.settings.data_dir.exists() and
                get_container() is not None
            )
        except Exception:
            return False
    
    def liveness_check(self) -> bool:
        """Simple liveness check for container orchestration."""
        try:
            # Basic check that the process is alive and responsive
            return True
        except Exception:
            return False


# CLI entry point for health checks
async def main():
    """CLI entry point for health checks."""
    checker = HealthChecker()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--readiness":
        exit(0 if checker.readiness_check() else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "--liveness":
        exit(0 if checker.liveness_check() else 1)
    else:
        # Full health check
        health = await checker.check_health()
        print(health.model_dump_json(indent=2))
        exit(0 if health.status == "healthy" else 1)


if __name__ == "__main__":
    asyncio.run(main())
