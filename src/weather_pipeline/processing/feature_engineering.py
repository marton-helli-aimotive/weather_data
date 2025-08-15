"""Feature engineering for weather data analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from ..models.weather import WeatherDataPoint


class FeatureEngineer:
    """Advanced feature engineering for weather data."""
    
    def __init__(self) -> None:
        pass
        
    def create_rolling_features(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        columns: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
        operations: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Create rolling window features.
        
        Args:
            data: Time series data
            columns: Columns to create rolling features for
            windows: Window sizes (in time periods)
            operations: Operations to apply ('mean', 'std', 'min', 'max', 'sum')
            
        Returns:
            DataFrame with additional rolling features
        """
        if columns is None:
            columns = ["temperature", "humidity", "pressure", "wind_speed"]
            
        if windows is None:
            windows = [3, 6, 12, 24]  # Hours for typical weather data
            
        if operations is None:
            operations = ["mean", "std", "min", "max"]
            
        # Handle both pandas and polars
        use_polars = isinstance(data, pl.DataFrame)
        
        if use_polars:
            df = data.clone()
            available_columns = [col for col in columns if col in df.columns]
        else:
            df = data.copy()
            available_columns = [col for col in columns if col in df.columns]
            
        # Create rolling features
        for col in available_columns:
            for window in windows:
                for op in operations:
                    feature_name = f"{col}_rolling_{window}_{op}"
                    
                    if use_polars:
                        # Polars rolling operations
                        if op == "mean":
                            df = df.with_columns(
                                pl.col(col).rolling_mean(window).alias(feature_name)
                            )
                        elif op == "std":
                            df = df.with_columns(
                                pl.col(col).rolling_std(window).alias(feature_name)
                            )
                        elif op == "min":
                            df = df.with_columns(
                                pl.col(col).rolling_min(window).alias(feature_name)
                            )
                        elif op == "max":
                            df = df.with_columns(
                                pl.col(col).rolling_max(window).alias(feature_name)
                            )
                        elif op == "sum":
                            df = df.with_columns(
                                pl.col(col).rolling_sum(window).alias(feature_name)
                            )
                    else:
                        # Pandas rolling operations
                        if op == "mean":
                            df[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                        elif op == "std":
                            df[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                        elif op == "min":
                            df[feature_name] = df[col].rolling(window=window, min_periods=1).min()
                        elif op == "max":
                            df[feature_name] = df[col].rolling(window=window, min_periods=1).max()
                        elif op == "sum":
                            df[feature_name] = df[col].rolling(window=window, min_periods=1).sum()
                            
        return df
        
    def create_lag_features(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        columns: Optional[List[str]] = None,
        lags: Optional[List[int]] = None
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Create lagged features for time series analysis.
        
        Args:
            data: Time series data
            columns: Columns to create lag features for
            lags: Lag periods to create
            
        Returns:
            DataFrame with lag features
        """
        if columns is None:
            columns = ["temperature", "humidity", "pressure", "wind_speed"]
            
        if lags is None:
            lags = [1, 3, 6, 12, 24]  # Hours
            
        use_polars = isinstance(data, pl.DataFrame)
        
        if use_polars:
            df = data.clone()
            available_columns = [col for col in columns if col in df.columns]
        else:
            df = data.copy()
            available_columns = [col for col in columns if col in df.columns]
            
        # Create lag features
        for col in available_columns:
            for lag in lags:
                feature_name = f"{col}_lag_{lag}"
                
                if use_polars:
                    df = df.with_columns(
                        pl.col(col).shift(lag).alias(feature_name)
                    )
                else:
                    df[feature_name] = df[col].shift(lag)
                    
        return df
        
    def create_derived_metrics(
        self,
        data: Union[pd.DataFrame, pl.DataFrame]
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Create derived meteorological metrics.
        
        Args:
            data: Weather data with basic variables
            
        Returns:
            DataFrame with derived metrics
        """
        use_polars = isinstance(data, pl.DataFrame)
        
        if use_polars:
            df = data.clone()
        else:
            df = data.copy()
            
        # Heat Index (simplified formula)
        if all(col in df.columns for col in ["temperature", "humidity"]):
            if use_polars:
                df = df.with_columns([
                    # Heat index approximation
                    (pl.col("temperature") + 0.5 * (pl.col("humidity") / 100.0) * 
                     (pl.col("temperature") - 14.0)).alias("heat_index")
                ])
            else:
                df["heat_index"] = (df["temperature"] + 0.5 * (df["humidity"] / 100.0) * 
                                  (df["temperature"] - 14.0))
                
        # Wind Chill (simplified formula for temperatures below 10°C)
        if all(col in df.columns for col in ["temperature", "wind_speed"]):
            if use_polars:
                df = df.with_columns([
                    pl.when(pl.col("temperature") < 10.0)
                    .then(13.12 + 0.6215 * pl.col("temperature") - 
                          11.37 * (pl.col("wind_speed") * 3.6).pow(0.16) +
                          0.3965 * pl.col("temperature") * (pl.col("wind_speed") * 3.6).pow(0.16))
                    .otherwise(pl.col("temperature"))
                    .alias("wind_chill")
                ])
            else:
                # Wind chill formula (convert m/s to km/h for formula)
                wind_kmh = df["wind_speed"] * 3.6
                wind_chill = np.where(
                    df["temperature"] < 10.0,
                    13.12 + 0.6215 * df["temperature"] - 11.37 * (wind_kmh**0.16) + 
                    0.3965 * df["temperature"] * (wind_kmh**0.16),
                    df["temperature"]
                )
                df["wind_chill"] = wind_chill
                
        # Dew Point approximation (Magnus formula)
        if all(col in df.columns for col in ["temperature", "humidity"]):
            if use_polars:
                df = df.with_columns([
                    (243.04 * (pl.col("humidity")/100.0).ln() + 17.625 * pl.col("temperature") / 
                     (243.04 + pl.col("temperature"))) /
                    (17.625 - (pl.col("humidity")/100.0).ln() - 
                     17.625 * pl.col("temperature") / (243.04 + pl.col("temperature")))
                    .alias("dew_point")
                ])
            else:
                a = 17.625
                b = 243.04
                alpha = np.log(df["humidity"]/100.0) + (a * df["temperature"]) / (b + df["temperature"])
                df["dew_point"] = (b * alpha) / (a - alpha)
                
        # Pressure tendency (change in pressure)
        if "pressure" in df.columns:
            if use_polars:
                df = df.with_columns([
                    (pl.col("pressure") - pl.col("pressure").shift(1)).alias("pressure_tendency")
                ])
            else:
                df["pressure_tendency"] = df["pressure"].diff()
                
        # Temperature range features
        if "temperature" in df.columns:
            if use_polars:
                df = df.with_columns([
                    (pl.col("temperature").rolling_max(24) - 
                     pl.col("temperature").rolling_min(24)).alias("temp_daily_range")
                ])
            else:
                df["temp_daily_range"] = (df["temperature"].rolling(24, min_periods=1).max() - 
                                        df["temperature"].rolling(24, min_periods=1).min())
                
        # Wind direction categories
        if "wind_direction" in df.columns:
            if use_polars:
                df = df.with_columns([
                    pl.when(pl.col("wind_direction").is_between(337.5, 360) | 
                           pl.col("wind_direction").is_between(0, 22.5))
                    .then(pl.lit("N"))
                    .when(pl.col("wind_direction").is_between(22.5, 67.5))
                    .then(pl.lit("NE"))
                    .when(pl.col("wind_direction").is_between(67.5, 112.5))
                    .then(pl.lit("E"))
                    .when(pl.col("wind_direction").is_between(112.5, 157.5))
                    .then(pl.lit("SE"))
                    .when(pl.col("wind_direction").is_between(157.5, 202.5))
                    .then(pl.lit("S"))
                    .when(pl.col("wind_direction").is_between(202.5, 247.5))
                    .then(pl.lit("SW"))
                    .when(pl.col("wind_direction").is_between(247.5, 292.5))
                    .then(pl.lit("W"))
                    .when(pl.col("wind_direction").is_between(292.5, 337.5))
                    .then(pl.lit("NW"))
                    .otherwise(pl.lit("Unknown"))
                    .alias("wind_direction_category")
                ])
            else:
                def categorize_wind_direction(degree):
                    if pd.isna(degree):
                        return "Unknown"
                    if 337.5 <= degree <= 360 or 0 <= degree < 22.5:
                        return "N"
                    elif 22.5 <= degree < 67.5:
                        return "NE"
                    elif 67.5 <= degree < 112.5:
                        return "E"
                    elif 112.5 <= degree < 157.5:
                        return "SE"
                    elif 157.5 <= degree < 202.5:
                        return "S"
                    elif 202.5 <= degree < 247.5:
                        return "SW"
                    elif 247.5 <= degree < 292.5:
                        return "W"
                    elif 292.5 <= degree < 337.5:
                        return "NW"
                    else:
                        return "Unknown"
                        
                df["wind_direction_category"] = df["wind_direction"].apply(categorize_wind_direction)
                
        return df
        
    def create_statistical_features(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        columns: Optional[List[str]] = None,
        window_size: int = 24
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Create statistical features like percentiles and z-scores.
        
        Args:
            data: Time series data
            columns: Columns to create statistical features for
            window_size: Window size for rolling statistics
            
        Returns:
            DataFrame with statistical features
        """
        if columns is None:
            columns = ["temperature", "humidity", "pressure", "wind_speed"]
            
        use_polars = isinstance(data, pl.DataFrame)
        
        if use_polars:
            df = data.clone()
            available_columns = [col for col in columns if col in df.columns]
        else:
            df = data.copy()
            available_columns = [col for col in columns if col in df.columns]
            
        for col in available_columns:
            if use_polars:
                # Rolling quantiles
                df = df.with_columns([
                    pl.col(col).rolling_quantile(0.25, window_size).alias(f"{col}_q25"),
                    pl.col(col).rolling_quantile(0.75, window_size).alias(f"{col}_q75"),
                    pl.col(col).rolling_quantile(0.90, window_size).alias(f"{col}_q90"),
                ])
                
                # Z-score (standardized values)
                rolling_mean = pl.col(col).rolling_mean(window_size)
                rolling_std = pl.col(col).rolling_std(window_size)
                df = df.with_columns([
                    ((pl.col(col) - rolling_mean) / rolling_std).alias(f"{col}_zscore")
                ])
                
            else:
                # Rolling quantiles
                rolling = df[col].rolling(window=window_size, min_periods=1)
                df[f"{col}_q25"] = rolling.quantile(0.25)
                df[f"{col}_q75"] = rolling.quantile(0.75)
                df[f"{col}_q90"] = rolling.quantile(0.90)
                
                # Z-score
                rolling_mean = rolling.mean()
                rolling_std = rolling.std()
                df[f"{col}_zscore"] = (df[col] - rolling_mean) / rolling_std
                
                # Percentile rank
                df[f"{col}_percentile_rank"] = df[col].rolling(
                    window=window_size, min_periods=1
                ).apply(lambda x: stats.percentileofscore(x, x.iloc[-1], kind='rank'))
                
        return df
        
    def create_temporal_features(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        timestamp_col: str = "timestamp"
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Create temporal features from timestamp.
        
        Args:
            data: Data with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with temporal features
        """
        use_polars = isinstance(data, pl.DataFrame)
        
        if use_polars:
            df = data.clone()
            
            if timestamp_col in df.columns:
                df = df.with_columns([
                    pl.col(timestamp_col).dt.hour().alias("hour"),
                    pl.col(timestamp_col).dt.day().alias("day"),
                    pl.col(timestamp_col).dt.month().alias("month"),
                    pl.col(timestamp_col).dt.year().alias("year"),
                    pl.col(timestamp_col).dt.weekday().alias("day_of_week"),
                    pl.col(timestamp_col).dt.ordinal_day().alias("day_of_year"),
                    pl.col(timestamp_col).dt.quarter().alias("quarter")
                ])
                
                # Cyclical encoding for circular features
                df = df.with_columns([
                    (2 * np.pi * pl.col("hour") / 24).sin().alias("hour_sin"),
                    (2 * np.pi * pl.col("hour") / 24).cos().alias("hour_cos"),
                    (2 * np.pi * pl.col("day_of_year") / 365).sin().alias("day_of_year_sin"),
                    (2 * np.pi * pl.col("day_of_year") / 365).cos().alias("day_of_year_cos"),
                    (2 * np.pi * pl.col("month") / 12).sin().alias("month_sin"),
                    (2 * np.pi * pl.col("month") / 12).cos().alias("month_cos")
                ])
                
        else:
            df = data.copy()
            
            if timestamp_col in df.columns:
                # Ensure datetime type
                if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    
                # Basic temporal features
                df["hour"] = df[timestamp_col].dt.hour
                df["day"] = df[timestamp_col].dt.day
                df["month"] = df[timestamp_col].dt.month
                df["year"] = df[timestamp_col].dt.year
                df["day_of_week"] = df[timestamp_col].dt.dayofweek
                df["day_of_year"] = df[timestamp_col].dt.dayofyear
                df["quarter"] = df[timestamp_col].dt.quarter
                
                # Week of year
                df["week_of_year"] = df[timestamp_col].dt.isocalendar().week
                
                # Season based on month
                df["season"] = df["month"].map({
                    12: "Winter", 1: "Winter", 2: "Winter",
                    3: "Spring", 4: "Spring", 5: "Spring", 
                    6: "Summer", 7: "Summer", 8: "Summer",
                    9: "Autumn", 10: "Autumn", 11: "Autumn"
                })
                
                # Cyclical encoding
                df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
                df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
                df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
                df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
                df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
                df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
                
                # Is weekend
                df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
                
        return df
        
    def create_interaction_features(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Create interaction features between variables.
        
        Args:
            data: Input data
            feature_pairs: Pairs of features to create interactions for
            
        Returns:
            DataFrame with interaction features
        """
        if feature_pairs is None:
            # Default weather variable interactions
            feature_pairs = [
                ("temperature", "humidity"),
                ("temperature", "pressure"), 
                ("wind_speed", "temperature"),
                ("pressure", "humidity")
            ]
            
        use_polars = isinstance(data, pl.DataFrame)
        
        if use_polars:
            df = data.clone()
        else:
            df = data.copy()
            
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplicative interaction
                interaction_name = f"{feat1}_x_{feat2}"
                
                if use_polars:
                    df = df.with_columns([
                        (pl.col(feat1) * pl.col(feat2)).alias(interaction_name)
                    ])
                else:
                    df[interaction_name] = df[feat1] * df[feat2]
                    
                # Ratio interaction (if no zeros)
                ratio_name = f"{feat1}_div_{feat2}"
                if use_polars:
                    df = df.with_columns([
                        pl.when(pl.col(feat2) != 0)
                        .then(pl.col(feat1) / pl.col(feat2))
                        .otherwise(None)
                        .alias(ratio_name)
                    ])
                else:
                    df[ratio_name] = np.where(df[feat2] != 0, df[feat1] / df[feat2], np.nan)
                    
        return df
        
    def create_change_features(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        columns: Optional[List[str]] = None,
        periods: Optional[List[int]] = None
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """
        Create rate of change features.
        
        Args:
            data: Time series data
            columns: Columns to create change features for
            periods: Periods to calculate changes over
            
        Returns:
            DataFrame with change features
        """
        if columns is None:
            columns = ["temperature", "pressure", "humidity", "wind_speed"]
            
        if periods is None:
            periods = [1, 3, 6, 12]  # Hours
            
        use_polars = isinstance(data, pl.DataFrame)
        
        if use_polars:
            df = data.clone()
            available_columns = [col for col in columns if col in df.columns]
        else:
            df = data.copy()
            available_columns = [col for col in columns if col in df.columns]
            
        for col in available_columns:
            for period in periods:
                # Absolute change
                change_name = f"{col}_change_{period}"
                # Percentage change
                pct_change_name = f"{col}_pct_change_{period}"
                
                if use_polars:
                    df = df.with_columns([
                        (pl.col(col) - pl.col(col).shift(period)).alias(change_name),
                        ((pl.col(col) - pl.col(col).shift(period)) / 
                         pl.col(col).shift(period) * 100).alias(pct_change_name)
                    ])
                else:
                    df[change_name] = df[col] - df[col].shift(period)
                    df[pct_change_name] = df[col].pct_change(periods=period) * 100
                    
        return df
