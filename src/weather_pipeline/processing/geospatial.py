"""Geospatial analysis capabilities for weather data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

from ..models.weather import WeatherDataPoint, Coordinates


class GeospatialAnalyzer:
    """Advanced geospatial analysis for weather data."""
    
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        
    def cluster_weather_patterns(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        features: Optional[List[str]] = None,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        include_coordinates: bool = True
    ) -> Dict[str, Any]:
        """
        Cluster weather patterns based on meteorological variables.
        
        Args:
            data: Weather data with coordinates
            features: Features to use for clustering
            method: Clustering method ('kmeans', 'dbscan')
            n_clusters: Number of clusters (for kmeans)
            include_coordinates: Whether to include lat/lon in clustering
            
        Returns:
            Dictionary with clustering results
        """
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
            
        if features is None:
            features = ["temperature", "humidity", "pressure", "wind_speed"]
            
        # Add coordinates if requested
        if include_coordinates:
            if "latitude" in data.columns and "longitude" in data.columns:
                features.extend(["latitude", "longitude"])
            else:
                print("Warning: Coordinates not found, clustering without location")
                
        # Filter available features
        available_features = [f for f in features if f in data.columns]
        if not available_features:
            return {"error": "No valid features found for clustering"}
            
        # Prepare clustering data
        cluster_data = data[available_features].dropna()
        
        if len(cluster_data) < 3:
            return {"error": "Insufficient data points for clustering"}
            
        # Scale features
        scaled_data = self.scaler.fit_transform(cluster_data)
        
        results = {
            "method": method,
            "features_used": available_features,
            "data_points": len(cluster_data)
        }
        
        if method == "kmeans":
            if n_clusters is None:
                # Use elbow method to find optimal clusters
                n_clusters = self._find_optimal_kmeans_clusters(scaled_data)
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            results.update({
                "n_clusters": n_clusters,
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "inertia": float(kmeans.inertia_)
            })
            
        elif method == "dbscan":
            # Auto-tune DBSCAN parameters
            eps, min_samples = self._tune_dbscan_params(scaled_data)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(scaled_data)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            results.update({
                "n_clusters": n_clusters,
                "cluster_labels": cluster_labels.tolist(),
                "eps": eps,
                "min_samples": min_samples,
                "n_noise_points": n_noise
            })
            
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_stats(cluster_data, cluster_labels)
        results["cluster_statistics"] = cluster_stats
        
        return results
        
    def spatial_interpolation(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        target_column: str,
        grid_resolution: float = 0.1,
        method: str = "idw"
    ) -> Dict[str, Any]:
        """
        Perform spatial interpolation of weather variables.
        
        Args:
            data: Weather data with coordinates
            target_column: Variable to interpolate
            grid_resolution: Grid resolution in degrees
            method: Interpolation method ('idw', 'kriging')
            
        Returns:
            Interpolated data and metadata
        """
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
            
        # Ensure we have coordinates and target variable
        required_cols = ["latitude", "longitude", target_column]
        if not all(col in data.columns for col in required_cols):
            return {"error": f"Missing required columns: {required_cols}"}
            
        clean_data = data[required_cols].dropna()
        
        if len(clean_data) < 3:
            return {"error": "Insufficient data points for interpolation"}
            
        # Create interpolation grid
        lat_min, lat_max = clean_data["latitude"].min(), clean_data["latitude"].max()
        lon_min, lon_max = clean_data["longitude"].min(), clean_data["longitude"].max()
        
        lat_range = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
        lon_range = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
        
        grid_lat, grid_lon = np.meshgrid(lat_range, lon_range)
        grid_points = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
        
        results = {
            "method": method,
            "grid_shape": grid_lat.shape,
            "n_observations": len(clean_data),
            "bounds": {
                "lat_min": lat_min, "lat_max": lat_max,
                "lon_min": lon_min, "lon_max": lon_max
            }
        }
        
        if method == "idw":
            # Inverse Distance Weighting
            interpolated = self._idw_interpolation(
                clean_data[["latitude", "longitude"]].values,
                clean_data[target_column].values,
                grid_points,
                power=2
            )
            
        elif method == "kriging":
            # Simple kriging implementation
            interpolated = self._simple_kriging(
                clean_data[["latitude", "longitude"]].values,
                clean_data[target_column].values,
                grid_points
            )
            
        else:
            return {"error": f"Unknown interpolation method: {method}"}
            
        results.update({
            "interpolated_grid": interpolated.reshape(grid_lat.shape).tolist(),
            "grid_coordinates": {
                "latitude": lat_range.tolist(),
                "longitude": lon_range.tolist()
            }
        })
        
        return results
        
    def calculate_distances(
        self,
        point1: Tuple[float, float],
        point2: Union[Tuple[float, float], List[Tuple[float, float]]],
        method: str = "haversine"
    ) -> Union[float, List[float]]:
        """
        Calculate distances between geographic points.
        
        Args:
            point1: Reference point (lat, lon)
            point2: Target point(s) (lat, lon)
            method: Distance calculation method
            
        Returns:
            Distance(s) in kilometers
        """
        if isinstance(point2, tuple):
            point2 = [point2]
            
        distances = []
        
        for p2 in point2:
            if method == "haversine":
                dist = self._haversine_distance(point1, p2)
            elif method == "euclidean":
                dist = self._euclidean_distance(point1, p2)
            else:
                raise ValueError(f"Unknown distance method: {method}")
                
            distances.append(dist)
            
        return distances[0] if len(distances) == 1 else distances
        
    def find_nearest_stations(
        self,
        target_point: Tuple[float, float],
        station_data: Union[pd.DataFrame, pl.DataFrame],
        n_nearest: int = 5
    ) -> Dict[str, Any]:
        """
        Find nearest weather stations to a target point.
        
        Args:
            target_point: Target coordinates (lat, lon)
            station_data: Station data with coordinates
            n_nearest: Number of nearest stations to return
            
        Returns:
            Nearest stations with distances
        """
        if isinstance(station_data, pl.DataFrame):
            station_data = station_data.to_pandas()
            
        if "latitude" not in station_data.columns or "longitude" not in station_data.columns:
            return {"error": "Station data must have latitude and longitude columns"}
            
        # Calculate distances to all stations
        stations = station_data.dropna(subset=["latitude", "longitude"]).copy()
        
        distances = []
        for _, station in stations.iterrows():
            dist = self._haversine_distance(
                target_point,
                (station["latitude"], station["longitude"])
            )
            distances.append(dist)
            
        stations["distance_km"] = distances
        nearest = stations.nsmallest(n_nearest, "distance_km")
        
        return {
            "target_point": target_point,
            "nearest_stations": nearest.to_dict("records"),
            "mean_distance": float(nearest["distance_km"].mean()),
            "min_distance": float(nearest["distance_km"].min()),
            "max_distance": float(nearest["distance_km"].max())
        }
        
    def create_spatial_grid(
        self,
        bounds: Dict[str, float],
        resolution: float = 0.1
    ) -> Dict[str, Any]:
        """
        Create a spatial grid for analysis.
        
        Args:
            bounds: Geographic bounds (lat_min, lat_max, lon_min, lon_max)
            resolution: Grid resolution in degrees
            
        Returns:
            Spatial grid information
        """
        lat_range = np.arange(
            bounds["lat_min"], 
            bounds["lat_max"] + resolution, 
            resolution
        )
        lon_range = np.arange(
            bounds["lon_min"], 
            bounds["lon_max"] + resolution, 
            resolution
        )
        
        grid_lat, grid_lon = np.meshgrid(lat_range, lon_range)
        
        return {
            "grid_shape": grid_lat.shape,
            "n_points": grid_lat.size,
            "latitude_range": lat_range.tolist(),
            "longitude_range": lon_range.tolist(),
            "resolution": resolution,
            "bounds": bounds
        }
        
    def _find_optimal_kmeans_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        inertias = []
        k_range = range(1, min(max_k + 1, len(data)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            
        # Simple elbow detection
        if len(inertias) < 3:
            return 3
            
        # Calculate second derivatives to find elbow
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_derivatives.append(inertias[i-1] - 2*inertias[i] + inertias[i+1])
            
        if second_derivatives:
            elbow_idx = np.argmax(second_derivatives) + 1
            return k_range[elbow_idx]
            
        return 3  # Default
        
    def _tune_dbscan_params(self, data: np.ndarray) -> Tuple[float, int]:
        """Auto-tune DBSCAN parameters."""
        from sklearn.neighbors import NearestNeighbors
        
        # Use k-distance graph to estimate eps
        k = min(5, len(data) // 2)
        nbrs = NearestNeighbors(n_neighbors=k).fit(data)
        distances, indices = nbrs.kneighbors(data)
        
        # Sort distances and find knee point
        k_distances = np.sort(distances[:, k-1])
        eps = np.percentile(k_distances, 95)  # Conservative estimate
        
        # Estimate min_samples
        min_samples = max(3, len(data) // 50)
        
        return eps, min_samples
        
    def _calculate_cluster_stats(
        self, 
        data: pd.DataFrame, 
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate statistics for each cluster."""
        stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points in DBSCAN
                continue
                
            cluster_data = data[labels == label]
            cluster_stats = {}
            
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    cluster_stats[col] = {
                        "mean": float(cluster_data[col].mean()),
                        "std": float(cluster_data[col].std()),
                        "min": float(cluster_data[col].min()),
                        "max": float(cluster_data[col].max()),
                        "count": int(len(cluster_data))
                    }
                    
            stats[f"cluster_{label}"] = cluster_stats
            
        return stats
        
    def _haversine_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two points."""
        lat1, lon1 = np.radians(point1)
        lat2, lon2 = np.radians(point2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        # Earth's radius in kilometers
        r = 6371
        return r * c
        
    def _euclidean_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) * 111.32  # Convert to km
        
    def _idw_interpolation(
        self,
        points: np.ndarray,
        values: np.ndarray,
        grid_points: np.ndarray,
        power: float = 2
    ) -> np.ndarray:
        """Inverse Distance Weighting interpolation."""
        interpolated = np.zeros(len(grid_points))
        
        for i, grid_point in enumerate(grid_points):
            distances = np.sqrt(np.sum((points - grid_point)**2, axis=1))
            
            # Avoid division by zero
            if np.any(distances == 0):
                zero_idx = np.where(distances == 0)[0][0]
                interpolated[i] = values[zero_idx]
            else:
                weights = 1 / (distances ** power)
                interpolated[i] = np.sum(weights * values) / np.sum(weights)
                
        return interpolated
        
    def _simple_kriging(
        self,
        points: np.ndarray,
        values: np.ndarray,
        grid_points: np.ndarray
    ) -> np.ndarray:
        """Simple kriging interpolation."""
        # Simplified kriging using RBF-like approach
        from scipy.spatial.distance import cdist
        
        # Calculate distance matrix
        distances = cdist(grid_points, points)
        
        # Use Gaussian RBF kernel
        sigma = np.mean(distances) * 0.1
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        
        # Normalize weights
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        # Interpolate
        interpolated = np.sum(weights * values, axis=1)
        
        return interpolated
