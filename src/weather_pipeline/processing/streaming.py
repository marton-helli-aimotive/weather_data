"""Streaming data processing capabilities for real-time weather analysis."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
import polars as pl
import logging

from ..models.weather import WeatherDataPoint


class StreamProcessor:
    """Real-time stream processing for weather data."""
    
    def __init__(self, buffer_size: int = 1000, window_size: int = 100) -> None:
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.data_buffer: List[Dict[str, Any]] = []
        self.processors: List[StreamDataProcessor] = []
        self.is_running = False
        
    async def add_data_point(self, data_point: Union[Dict[str, Any], WeatherDataPoint]) -> None:
        """
        Add a data point to the stream buffer.
        
        Args:
            data_point: New weather data point
        """
        if isinstance(data_point, WeatherDataPoint):
            data_point = data_point.model_dump()
            
        # Add timestamp if not present
        if "timestamp" not in data_point:
            data_point["timestamp"] = datetime.utcnow()
            
        self.data_buffer.append(data_point)
        
        # Maintain buffer size
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            
        # Process data if we have enough points
        if len(self.data_buffer) >= self.window_size:
            await self._process_window()
            
    async def add_processor(self, processor: "StreamDataProcessor") -> None:
        """Add a stream processor."""
        self.processors.append(processor)
        
    async def start_processing(self) -> None:
        """Start continuous stream processing."""
        self.is_running = True
        
        while self.is_running:
            if len(self.data_buffer) >= self.window_size:
                await self._process_window()
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
            
    def stop_processing(self) -> None:
        """Stop stream processing."""
        self.is_running = False
        
    async def _process_window(self) -> None:
        """Process current window of data."""
        if not self.processors:
            return
            
        # Get current window
        window_data = self.data_buffer[-self.window_size:]
        df = pd.DataFrame(window_data)
        
        # Apply all processors
        for processor in self.processors:
            try:
                await processor.process(df)
            except Exception as e:
                logging.error(f"Error in stream processor {processor.__class__.__name__}: {e}")
                
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self.data_buffer:
            return {"size": 0, "empty": True}
            
        df = pd.DataFrame(self.data_buffer)
        
        return {
            "size": len(self.data_buffer),
            "empty": False,
            "oldest_timestamp": df["timestamp"].min() if "timestamp" in df.columns else None,
            "newest_timestamp": df["timestamp"].max() if "timestamp" in df.columns else None,
            "columns": list(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        
    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        self.data_buffer.clear()


class StreamDataProcessor(ABC):
    """Abstract base class for stream data processors."""
    
    @abstractmethod
    async def process(self, data: pd.DataFrame) -> Any:
        """Process a window of streaming data."""
        pass


class RealTimeAnomalyDetector(StreamDataProcessor):
    """Real-time anomaly detection for streaming weather data."""
    
    def __init__(self, 
                 columns: Optional[List[str]] = None,
                 threshold: float = 3.0,
                 callback: Optional[Callable] = None) -> None:
        self.columns = columns or ["temperature", "humidity", "pressure"]
        self.threshold = threshold
        self.callback = callback
        self.historical_stats: Dict[str, Dict[str, float]] = {}
        
    async def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in real-time data."""
        results = {"anomalies": [], "timestamp": datetime.utcnow()}
        
        for col in self.columns:
            if col not in data.columns:
                continue
                
            # Update historical statistics
            self._update_stats(col, data[col])
            
            # Detect anomalies in latest data
            if col in self.historical_stats:
                stats = self.historical_stats[col]
                latest_values = data[col].tail(10)  # Check last 10 values
                
                z_scores = np.abs((latest_values - stats["mean"]) / stats["std"])
                anomalies = z_scores > self.threshold
                
                if anomalies.any():
                    anomaly_indices = latest_values[anomalies].index.tolist()
                    anomaly_values = latest_values[anomalies].tolist()
                    
                    results["anomalies"].append({
                        "column": col,
                        "indices": anomaly_indices,
                        "values": anomaly_values,
                        "z_scores": z_scores[anomalies].tolist()
                    })
                    
        # Call callback if anomalies detected
        if results["anomalies"] and self.callback:
            await self.callback(results)
            
        return results
        
    def _update_stats(self, column: str, values: pd.Series) -> None:
        """Update running statistics for a column."""
        if column not in self.historical_stats:
            self.historical_stats[column] = {
                "mean": values.mean(),
                "std": values.std(),
                "count": len(values)
            }
        else:
            # Update running statistics
            old_stats = self.historical_stats[column]
            new_count = old_stats["count"] + len(values)
            new_mean = (old_stats["mean"] * old_stats["count"] + values.sum()) / new_count
            
            # Update variance using online algorithm
            old_var = old_stats["std"] ** 2
            new_var = ((old_stats["count"] - 1) * old_var + 
                      values.var() * (len(values) - 1) +
                      (old_stats["count"] * len(values) * 
                      (old_stats["mean"] - values.mean()) ** 2) / new_count) / (new_count - 1)
            
            self.historical_stats[column] = {
                "mean": new_mean,
                "std": np.sqrt(max(new_var, 0)),
                "count": new_count
            }


class RealTimeTrendDetector(StreamDataProcessor):
    """Real-time trend detection for streaming weather data."""
    
    def __init__(self, 
                 columns: Optional[List[str]] = None,
                 window_size: int = 50,
                 callback: Optional[Callable] = None) -> None:
        self.columns = columns or ["temperature", "pressure"]
        self.window_size = window_size
        self.callback = callback
        
    async def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect trends in real-time data."""
        results = {"trends": [], "timestamp": datetime.utcnow()}
        
        for col in self.columns:
            if col not in data.columns or len(data) < self.window_size:
                continue
                
            # Get recent window
            window_data = data[col].tail(self.window_size)
            
            # Calculate trend using linear regression
            x = np.arange(len(window_data))
            slope, intercept = np.polyfit(x, window_data, 1)
            
            # Determine trend direction and strength
            trend_strength = abs(slope)
            trend_direction = "increasing" if slope > 0 else "decreasing"
            
            # Statistical significance test
            correlation = np.corrcoef(x, window_data)[0, 1]
            is_significant = abs(correlation) > 0.5  # Simple threshold
            
            trend_info = {
                "column": col,
                "slope": slope,
                "direction": trend_direction,
                "strength": trend_strength,
                "correlation": correlation,
                "is_significant": is_significant,
                "latest_value": float(window_data.iloc[-1]),
                "window_size": len(window_data)
            }
            
            results["trends"].append(trend_info)
            
        # Call callback if significant trends detected
        significant_trends = [t for t in results["trends"] if t["is_significant"]]
        if significant_trends and self.callback:
            await self.callback({"significant_trends": significant_trends})
            
        return results


class StreamAggregator(StreamDataProcessor):
    """Real-time aggregation of streaming weather data."""
    
    def __init__(self, 
                 aggregation_window: str = "5T",  # 5 minutes
                 operations: Optional[List[str]] = None) -> None:
        self.aggregation_window = aggregation_window
        self.operations = operations or ["mean", "min", "max", "std"]
        self.aggregated_data: List[Dict[str, Any]] = []
        
    async def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate streaming data."""
        if "timestamp" not in data.columns:
            return {"error": "Timestamp column required for aggregation"}
            
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            
        # Set timestamp as index for resampling
        data_indexed = data.set_index("timestamp")
        
        # Perform aggregation
        numeric_columns = data_indexed.select_dtypes(include=[np.number]).columns
        
        aggregated = {}
        for op in self.operations:
            if op == "mean":
                agg_data = data_indexed[numeric_columns].resample(self.aggregation_window).mean()
            elif op == "min":
                agg_data = data_indexed[numeric_columns].resample(self.aggregation_window).min()
            elif op == "max":
                agg_data = data_indexed[numeric_columns].resample(self.aggregation_window).max()
            elif op == "std":
                agg_data = data_indexed[numeric_columns].resample(self.aggregation_window).std()
            elif op == "count":
                agg_data = data_indexed[numeric_columns].resample(self.aggregation_window).count()
            else:
                continue
                
            # Store latest aggregated values
            if not agg_data.empty:
                latest_agg = agg_data.iloc[-1].to_dict()
                aggregated[op] = {
                    "timestamp": agg_data.index[-1],
                    "values": latest_agg
                }
                
        # Store aggregated data
        if aggregated:
            self.aggregated_data.append({
                "aggregation_timestamp": datetime.utcnow(),
                "window": self.aggregation_window,
                "data": aggregated
            })
            
        return aggregated
        
    def get_aggregated_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get aggregated data history."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            agg for agg in self.aggregated_data
            if agg["aggregation_timestamp"] >= cutoff_time
        ]


class EventGenerator(StreamDataProcessor):
    """Generate events based on streaming weather data conditions."""
    
    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rules = rules or self._default_rules()
        self.events: List[Dict[str, Any]] = []
        
    def _default_rules(self) -> List[Dict[str, Any]]:
        """Default event generation rules."""
        return [
            {
                "name": "extreme_temperature",
                "condition": lambda df: (df["temperature"] > 40) | (df["temperature"] < -30),
                "severity": "high",
                "message": "Extreme temperature detected"
            },
            {
                "name": "sudden_pressure_drop",
                "condition": lambda df: df["pressure"].diff().abs() > 10,
                "severity": "medium", 
                "message": "Sudden pressure change detected"
            },
            {
                "name": "high_wind_speed",
                "condition": lambda df: df["wind_speed"] > 25,
                "severity": "medium",
                "message": "High wind speed detected"
            }
        ]
        
    async def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate events based on data conditions."""
        new_events = []
        
        for rule in self.rules:
            try:
                # Check rule condition
                if rule["condition"](data).any():
                    # Find matching records
                    matching_indices = data[rule["condition"](data)].index.tolist()
                    
                    event = {
                        "rule_name": rule["name"],
                        "severity": rule["severity"],
                        "message": rule["message"],
                        "timestamp": datetime.utcnow(),
                        "matching_records": len(matching_indices),
                        "latest_values": data.loc[matching_indices[-1:]].to_dict("records") if matching_indices else []
                    }
                    
                    new_events.append(event)
                    self.events.append(event)
                    
            except Exception as e:
                print(f"Error processing rule {rule['name']}: {e}")
                
        return {"events": new_events, "total_events_generated": len(new_events)}
        
    def get_recent_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent events."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            event for event in self.events
            if event["timestamp"] >= cutoff_time
        ]
        
    def add_rule(self, rule: Dict[str, Any]) -> None:
        """Add a custom event rule."""
        required_fields = ["name", "condition", "severity", "message"]
        if all(field in rule for field in required_fields):
            self.rules.append(rule)
        else:
            raise ValueError(f"Rule must contain fields: {required_fields}")


class StreamingPipeline:
    """Complete streaming pipeline orchestrator."""
    
    def __init__(self) -> None:
        self.stream_processor = StreamProcessor()
        self.is_running = False
        
    async def setup_default_pipeline(self) -> None:
        """Set up default streaming pipeline with common processors."""
        
        # Anomaly detection
        anomaly_detector = RealTimeAnomalyDetector(
            callback=self._handle_anomalies
        )
        await self.stream_processor.add_processor(anomaly_detector)
        
        # Trend detection
        trend_detector = RealTimeTrendDetector(
            callback=self._handle_trends
        )
        await self.stream_processor.add_processor(trend_detector)
        
        # Aggregation
        aggregator = StreamAggregator()
        await self.stream_processor.add_processor(aggregator)
        
        # Event generation
        event_generator = EventGenerator()
        await self.stream_processor.add_processor(event_generator)
        
    async def start(self) -> None:
        """Start the streaming pipeline."""
        self.is_running = True
        await self.stream_processor.start_processing()
        
    def stop(self) -> None:
        """Stop the streaming pipeline."""
        self.is_running = False
        self.stream_processor.stop_processing()
        
    async def add_data(self, data_point: Union[Dict[str, Any], WeatherDataPoint]) -> None:
        """Add data to the streaming pipeline."""
        await self.stream_processor.add_data_point(data_point)
        
    async def _handle_anomalies(self, anomaly_result: Dict[str, Any]) -> None:
        """Handle detected anomalies."""
        print(f"Anomalies detected at {anomaly_result['timestamp']}: {len(anomaly_result['anomalies'])} anomalies")
        
    async def _handle_trends(self, trend_result: Dict[str, Any]) -> None:
        """Handle detected trends."""
        significant_count = len(trend_result.get("significant_trends", []))
        if significant_count > 0:
            print(f"Significant trends detected: {significant_count} trends")
            
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "is_running": self.is_running,
            "buffer_stats": self.stream_processor.get_buffer_stats(),
            "processor_count": len(self.stream_processor.processors),
            "processors": [p.__class__.__name__ for p in self.stream_processor.processors]
        }
