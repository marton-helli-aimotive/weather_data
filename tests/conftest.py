"""Test configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_coordinates():
    """Sample coordinates for testing."""
    return {
        "budapest": {"latitude": 47.4979, "longitude": 19.0402},
        "london": {"latitude": 51.5074, "longitude": -0.1278},
        "invalid_lat": {"latitude": 91.0, "longitude": 0.0},
        "invalid_lon": {"latitude": 0.0, "longitude": 181.0},
    }
