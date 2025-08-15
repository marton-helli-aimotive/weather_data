"""Time series analysis capabilities for weather data."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from ..models.weather import WeatherDataPoint


class TimeSeriesAnalyzer:
    """Advanced time series analysis for weather data."""
    
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        
    def detect_trends(
        self, 
        data: Union[pd.DataFrame, pl.DataFrame],
        column: str = "temperature",
        window_size: int = 24
    ) -> Dict[str, Any]:
        """
        Detect trends in time series data using various methods.
        
        Args:
            data: Time series data with datetime index
            column: Column to analyze for trends
            window_size: Window size for trend detection
            
        Returns:
            Dictionary containing trend analysis results
        """
        if isinstance(data, pl.DataFrame):
            # Convert Polars to Pandas for statsmodels compatibility
            data = data.to_pandas()
            
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                raise ValueError("Data must have datetime index or timestamp column")
                
        # Ensure we have the column
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
            
        series = data[column].dropna()
        
        results = {}
        
        # 1. Linear trend using regression
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
        results["linear_trend"] = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        }
        
        # 2. Rolling trend analysis
        rolling_slopes = []
        for i in range(window_size, len(series)):
            window_data = series.iloc[i-window_size:i]
            window_x = np.arange(len(window_data))
            window_slope, _, _, _, _ = stats.linregress(window_x, window_data.values)
            rolling_slopes.append(window_slope)
            
        results["rolling_trend"] = {
            "mean_slope": np.mean(rolling_slopes),
            "slope_volatility": np.std(rolling_slopes),
            "positive_trend_periods": sum(1 for s in rolling_slopes if s > 0),
            "negative_trend_periods": sum(1 for s in rolling_slopes if s < 0)
        }
        
        # 3. Stationarity test (Augmented Dickey-Fuller)
        adf_result = adfuller(series.values)
        results["stationarity"] = {
            "adf_statistic": adf_result[0],
            "p_value": adf_result[1],
            "is_stationary": bool(adf_result[1] < 0.05),
            "critical_values": adf_result[4]
        }
        
        return results
        
    def analyze_seasonality(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        column: str = "temperature",
        period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in weather data.
        
        Args:
            data: Time series data
            column: Column to analyze
            period: Seasonal period (auto-detected if None)
            
        Returns:
            Dictionary with seasonality analysis results
        """
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
            
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                raise ValueError("Data must have datetime index or timestamp column")
                
        series = data[column].dropna()
        
        if len(series) < 50:  # Minimum data points for decomposition
            return {"error": "Insufficient data for seasonal analysis"}
            
        results = {}
        
        # Auto-detect period if not provided
        if period is None:
            # For hourly data, try daily (24) and weekly (168) patterns
            # For daily data, try weekly (7) and monthly (30) patterns
            freq = pd.infer_freq(series.index)
            if freq and 'H' in freq:  # Hourly data
                period = 24  # Daily seasonality
            elif freq and 'D' in freq:  # Daily data
                period = 7   # Weekly seasonality
            else:
                period = min(len(series) // 4, 24)  # Conservative default
                
        try:
            # Seasonal decomposition
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decomposition = seasonal_decompose(
                    series, 
                    model='additive', 
                    period=period,
                    extrapolate_trend='freq'
                )
            
            results["decomposition"] = {
                "period_used": period,
                "trend_strength": float(np.var(decomposition.trend.dropna()) / np.var(series)),
                "seasonal_strength": float(np.var(decomposition.seasonal) / np.var(series)),
                "residual_variance": float(np.var(decomposition.resid.dropna())),
                "seasonal_peaks": self._find_seasonal_peaks(decomposition.seasonal, period)
            }
            
            # Seasonal pattern analysis
            if period <= 24:  # Hourly/sub-daily analysis
                hourly_pattern = series.groupby(series.index.hour).mean()
                results["hourly_pattern"] = {
                    "peak_hour": int(hourly_pattern.idxmax()),
                    "min_hour": int(hourly_pattern.idxmin()),
                    "range": float(hourly_pattern.max() - hourly_pattern.min())
                }
                
            # Monthly pattern analysis
            monthly_pattern = series.groupby(series.index.month).mean()
            results["monthly_pattern"] = {
                "peak_month": int(monthly_pattern.idxmax()),
                "min_month": int(monthly_pattern.idxmin()),
                "seasonal_amplitude": float(monthly_pattern.max() - monthly_pattern.min())
            }
            
        except Exception as e:
            results["error"] = f"Seasonal decomposition failed: {str(e)}"
            
        return results
        
    def detect_anomalies(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        column: str = "temperature",
        method: str = "zscore",
        threshold: float = 3.0,
        window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies in weather time series data.
        
        Args:
            data: Time series data
            column: Column to analyze for anomalies  
            method: Anomaly detection method ('zscore', 'iqr', 'isolation')
            threshold: Threshold for anomaly detection
            window_size: Window size for rolling anomaly detection
            
        Returns:
            Dictionary with anomaly detection results
        """
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
            
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
            
        series = data[column].dropna()
        results = {"method": method, "threshold": threshold}
        
        if method == "zscore":
            if window_size:
                # Rolling z-score
                rolling_mean = series.rolling(window=window_size).mean()
                rolling_std = series.rolling(window=window_size).std()
                z_scores = (series - rolling_mean) / rolling_std
            else:
                # Global z-score
                z_scores = stats.zscore(series)
                
            anomalies = np.abs(z_scores) > threshold
            results["anomaly_indices"] = series.index[anomalies].tolist()
            results["anomaly_values"] = series[anomalies].tolist()
            results["z_scores"] = z_scores[anomalies].tolist()
            
        elif method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            anomalies = (series < lower_bound) | (series > upper_bound)
            results["anomaly_indices"] = series.index[anomalies].tolist()
            results["anomaly_values"] = series[anomalies].tolist()
            results["bounds"] = {"lower": lower_bound, "upper": upper_bound}
            
        elif method == "isolation":
            from sklearn.ensemble import IsolationForest
            
            # Prepare features (value + time features)
            X = np.column_stack([
                series.values,
                series.index.hour if hasattr(series.index, 'hour') else np.zeros(len(series)),
                series.index.dayofyear if hasattr(series.index, 'dayofyear') else np.arange(len(series))
            ])
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(X) == -1
            
            results["anomaly_indices"] = series.index[anomalies].tolist()
            results["anomaly_values"] = series[anomalies].tolist()
            results["anomaly_scores"] = iso_forest.score_samples(X)[anomalies].tolist()
            
        # Summary statistics
        results["total_anomalies"] = int(np.sum(anomalies))
        results["anomaly_rate"] = float(np.sum(anomalies) / len(series))
        
        if len(series[anomalies]) > 0:
            results["anomaly_stats"] = {
                "mean": float(series[anomalies].mean()),
                "std": float(series[anomalies].std()),
                "min": float(series[anomalies].min()),
                "max": float(series[anomalies].max())
            }
            
        return results
        
    def forecast_simple(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        column: str = "temperature",
        steps: int = 24,
        method: str = "linear"
    ) -> Dict[str, Any]:
        """
        Simple forecasting using basic methods.
        
        Args:
            data: Historical time series data
            column: Column to forecast
            steps: Number of steps to forecast
            method: Forecasting method ('linear', 'seasonal', 'ewm')
            
        Returns:
            Dictionary with forecast results
        """
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
            
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                raise ValueError("Data must have datetime index or timestamp column")
                
        series = data[column].dropna()
        
        if len(series) < 10:
            return {"error": "Insufficient data for forecasting"}
            
        results = {"method": method, "steps": steps}
        
        # Generate future timestamps
        last_timestamp = series.index[-1]
        freq = pd.infer_freq(series.index)
        
        if freq is None:
            # Fallback to calculating frequency from the data
            if len(series.index) < 2:
                return {"error": "Insufficient data to infer frequency for forecasting"}
            time_diff = series.index[-1] - series.index[-2]
            future_index = pd.date_range(
                start=last_timestamp + time_diff, 
                periods=steps, 
                freq=time_diff
            )
        else:
            # Use the inferred frequency directly
            future_index = pd.date_range(
                start=last_timestamp, 
                periods=steps + 1, 
                freq=freq
            )[1:]  # Skip the first element which is the last_timestamp
        
        if method == "linear":
            # Linear trend extrapolation
            x = np.arange(len(series))
            slope, intercept, _, _, _ = stats.linregress(x, series.values)
            
            future_x = np.arange(len(series), len(series) + steps)
            forecast = slope * future_x + intercept
            
        elif method == "seasonal":
            # Seasonal naive forecast
            period = min(24, len(series) // 2)  # Daily pattern or half the data
            seasonal_pattern = []
            
            for i in range(steps):
                idx = (len(series) - period + (i % period)) if len(series) >= period else -1
                seasonal_pattern.append(series.iloc[idx])
                
            forecast = np.array(seasonal_pattern)
            
        elif method == "ewm":
            # Exponential weighted moving average
            alpha = 0.3
            ewm_forecast = series.ewm(alpha=alpha).mean().iloc[-1]
            forecast = np.full(steps, ewm_forecast)
            
        else:
            return {"error": f"Unknown forecasting method: {method}"}
            
        results["forecast_values"] = forecast.tolist()
        results["forecast_index"] = future_index.tolist()
        results["last_observed"] = float(series.iloc[-1])
        
        return results
        
    def _find_seasonal_peaks(self, seasonal_component: pd.Series, period: int) -> List[int]:
        """Find peaks in seasonal component."""
        # Reshape to identify peaks within each period
        if len(seasonal_component) >= period:
            reshaped = seasonal_component.values[:len(seasonal_component)//period * period].reshape(-1, period)
            mean_pattern = np.mean(reshaped, axis=0)
            peaks = []
            
            # Simple peak detection
            for i in range(1, len(mean_pattern) - 1):
                if mean_pattern[i] > mean_pattern[i-1] and mean_pattern[i] > mean_pattern[i+1]:
                    peaks.append(i)
                    
            return peaks
        return []
        
    def time_aggregations(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        column: str = "temperature",
        aggregations: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Perform various time-based aggregations.
        
        Args:
            data: Time series data
            column: Column to aggregate
            aggregations: List of aggregation periods
            
        Returns:
            Dictionary of aggregated DataFrames
        """
        if aggregations is None:
            aggregations = ["1H", "6H", "1D", "1W", "1M"]
            
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
            
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index('timestamp')
            else:
                raise ValueError("Data must have datetime index or timestamp column")
                
        results = {}
        
        for agg_period in aggregations:
            try:
                agg_data = data[column].resample(agg_period).agg([
                    'mean', 'min', 'max', 'std', 'count'
                ]).dropna()
                
                results[agg_period] = agg_data
                
            except Exception as e:
                results[agg_period] = {"error": str(e)}
                
        return results
