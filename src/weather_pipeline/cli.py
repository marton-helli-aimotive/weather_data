"""Command-line interface for the weather pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, configure_logging, get_logger, settings
from .core.container import get_container


app = typer.Typer(
    name="weather-pipeline",
    help="Advanced Weather Data Engineering Pipeline",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"Weather Pipeline v{__version__}")


@app.command()
def config() -> None:
    """Show current configuration."""
    configure_logging()
    
    table = Table(title="Configuration Settings")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # Environment settings
    table.add_row("Environment", settings.environment.value)
    table.add_row("Debug Mode", str(settings.debug))
    table.add_row("App Version", settings.app_version)
    
    # API settings
    table.add_row(
        "WeatherAPI Key",
        (
            "***" + settings.api.weatherapi_key[-4:]
            if settings.api.weatherapi_key and len(settings.api.weatherapi_key) >= 4
            else "***" if settings.api.weatherapi_key else "Not Set"
        )
    )
    table.add_row("OpenWeather Key", "Set" if settings.api.openweather_api_key else "Not Set")
    table.add_row("Rate Limit", str(settings.api.rate_limit_requests))
    
    # Database settings
    table.add_row("Database Host", settings.database.host)
    table.add_row("Database Port", str(settings.database.port))
    table.add_row("Database Name", settings.database.name)
    
    # Logging settings
    table.add_row("Log Level", settings.logging.level.value)
    table.add_row("Log Format", settings.logging.format)
    
    console.print(table)


@app.command()
def init(
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Data directory path"
    ),
) -> None:
    """Initialize the weather pipeline environment."""
    configure_logging()
    logger = get_logger("cli")
    
    with logger.bind(command="init"):
        logger.info("Initializing weather pipeline environment")
        
        # Create data directory
        actual_data_dir = data_dir or settings.data_dir
        actual_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created data directory", path=str(actual_data_dir))
        
        # Create subdirectories
        subdirs = ["raw", "processed", "cache", "exports"]
        for subdir in subdirs:
            subdir_path = actual_data_dir / subdir
            subdir_path.mkdir(exist_ok=True)
            logger.info("Created subdirectory", subdir=subdir, path=str(subdir_path))
        
        # Initialize dependency container
        container = get_container()
        logger.info("Initialized dependency injection container")
        
        console.print("[green]✓[/green] Weather pipeline environment initialized successfully!")
        console.print(f"[blue]Data directory:[/blue] {actual_data_dir}")


@app.command()
def check() -> None:
    """Check system health and dependencies."""
    configure_logging()
    logger = get_logger("cli")
    
    console.print("[yellow]Checking system health...[/yellow]")
    
    health_table = Table(title="System Health Check")
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Status", style="green")
    health_table.add_column("Details")
    
    # Check configuration
    try:
        config_status = "✓ OK"
        config_details = f"Environment: {settings.environment.value}"
        health_table.add_row("Configuration", config_status, config_details)
    except Exception as e:
        health_table.add_row("Configuration", "✗ ERROR", str(e))
    
    # Check logging
    try:
        logger.info("Testing logging configuration")
        health_table.add_row("Logging", "✓ OK", f"Level: {settings.logging.level.value}")
    except Exception as e:
        health_table.add_row("Logging", "✗ ERROR", str(e))
    
    # Check data directory
    try:
        if settings.data_dir.exists():
            health_table.add_row("Data Directory", "✓ OK", str(settings.data_dir))
        else:
            health_table.add_row("Data Directory", "⚠ MISSING", f"Run 'weather-pipeline init' to create")
    except Exception as e:
        health_table.add_row("Data Directory", "✗ ERROR", str(e))
    
    # Check dependency injection
    try:
        container = get_container()
        health_table.add_row("DI Container", "✓ OK", "Dependency injection ready")
    except Exception as e:
        health_table.add_row("DI Container", "✗ ERROR", str(e))
    
    console.print(health_table)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
