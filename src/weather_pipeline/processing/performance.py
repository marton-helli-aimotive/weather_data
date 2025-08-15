"""Performance comparison between pandas and Polars for weather data processing."""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import psutil
import gc

import numpy as np
import pandas as pd
import polars as pl

from ..models.weather import WeatherDataPoint


class PerformanceComparator:
    """Compare performance between pandas and Polars for weather data operations."""
    
    def __init__(self) -> None:
        self.benchmark_results: List[Dict[str, Any]] = []
        
    def benchmark_operation(self,
                           operation_name: str,
                           pandas_func: Callable[[pd.DataFrame], Any],
                           polars_func: Callable[[pl.DataFrame], Any],
                           data: Union[pd.DataFrame, pl.DataFrame],
                           iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark a specific operation on both pandas and Polars.
        
        Args:
            operation_name: Name of the operation being benchmarked
            pandas_func: Function to execute on pandas DataFrame
            polars_func: Function to execute on Polars DataFrame  
            data: Input data
            iterations: Number of iterations to average
            
        Returns:
            Performance comparison results
        """
        # Prepare data for both libraries
        if isinstance(data, pd.DataFrame):
            pandas_data = data.copy()
            polars_data = pl.from_pandas(data)
        else:
            polars_data = data.clone()
            pandas_data = data.to_pandas()
            
        # Benchmark pandas
        pandas_times = []
        pandas_memory_usage = []
        pandas_result = None
        
        for _ in range(iterations):
            gc.collect()  # Clean up before measurement

            # Measure memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Time the operation
            start_time = time.perf_counter()
            try:
                # Use a shallow copy (view) or reference for benchmarking
                pandas_result = pandas_func(pandas_data)
                pandas_success = True
            except Exception as e:
                pandas_success = False
                pandas_error = str(e)

            end_time = time.perf_counter()

            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB

            if pandas_success:
                pandas_times.append(end_time - start_time)
                pandas_memory_usage.append(memory_after - memory_before)
                
        # Benchmark Polars
        polars_times = []
        polars_memory_usage = []
        polars_result = None
        
        for _ in range(iterations):
            gc.collect()  # Clean up before measurement

            # Measure memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Time the operation
            start_time = time.perf_counter()
            try:
                # Use a reference for benchmarking (no clone)
                polars_result = polars_func(polars_data)
                polars_success = True
            except Exception as e:
                polars_success = False
                polars_error = str(e)

            end_time = time.perf_counter()

            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB

            if polars_success:
                polars_times.append(end_time - start_time)
                polars_memory_usage.append(memory_after - memory_before)
                
        # Calculate statistics
        results = {
            "operation": operation_name,
            "data_shape": data.shape,
            "data_size_mb": self._estimate_dataframe_size(data),
            "iterations": iterations
        }
        
        if pandas_times:
            results["pandas"] = {
                "success": pandas_success,
                "avg_time_seconds": np.mean(pandas_times),
                "std_time_seconds": np.std(pandas_times),
                "min_time_seconds": np.min(pandas_times),
                "max_time_seconds": np.max(pandas_times),
                "avg_memory_mb": np.mean(pandas_memory_usage),
                "result_type": str(type(pandas_result)),
                "result_shape": getattr(pandas_result, 'shape', None)
            }
        else:
            results["pandas"] = {
                "success": False,
                "error": pandas_error,
                "avg_time_seconds": float('inf')
            }
            
        if polars_times:
            results["polars"] = {
                "success": polars_success,
                "avg_time_seconds": np.mean(polars_times),
                "std_time_seconds": np.std(polars_times),
                "min_time_seconds": np.min(polars_times),
                "max_time_seconds": np.max(polars_times),
                "avg_memory_mb": np.mean(polars_memory_usage),
                "result_type": str(type(polars_result)),
                "result_shape": getattr(polars_result, 'shape', None)
            }
        else:
            results["polars"] = {
                "success": False,
                "error": polars_error,
                "avg_time_seconds": float('inf')
            }
            
        # Calculate performance ratio
        if pandas_times and polars_times:
            pandas_avg = np.mean(pandas_times)
            polars_avg = np.mean(polars_times)
            results["speedup_ratio"] = pandas_avg / polars_avg
            results["winner"] = "polars" if polars_avg < pandas_avg else "pandas"
        else:
            results["speedup_ratio"] = None
            results["winner"] = None
            
        # Store results
        self.benchmark_results.append(results)
        
        return results
        
    def comprehensive_benchmark(self, data: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
        """
        Run comprehensive benchmarks on common weather data operations.
        
        Args:
            data: Weather data to benchmark
            
        Returns:
            Complete benchmark results
        """
        benchmark_suite = [
            {
                "name": "basic_aggregation",
                "pandas": lambda df: df.groupby("city").agg({
                    "temperature": ["mean", "min", "max"],
                    "humidity": "mean",
                    "pressure": "mean"
                }),
                "polars": lambda df: df.group_by("city").agg([
                    pl.col("temperature").mean().alias("temp_mean"),
                    pl.col("temperature").min().alias("temp_min"), 
                    pl.col("temperature").max().alias("temp_max"),
                    pl.col("humidity").mean().alias("humidity_mean"),
                    pl.col("pressure").mean().alias("pressure_mean")
                ])
            },
            {
                "name": "filtering", 
                "pandas": lambda df: df[(df["temperature"] > 20) & (df["humidity"] < 80)],
                "polars": lambda df: df.filter(
                    (pl.col("temperature") > 20) & (pl.col("humidity") < 80)
                )
            },
            {
                "name": "sorting",
                "pandas": lambda df: df.sort_values(["city", "timestamp"]),
                "polars": lambda df: df.sort(["city", "timestamp"])
            },
            {
                "name": "rolling_mean",
                "pandas": lambda df: df.assign(
                    temp_rolling=df["temperature"].rolling(window=5, min_periods=1).mean()
                ),
                "polars": lambda df: df.with_columns([
                    pl.col("temperature").rolling_mean(window_size=5).alias("temp_rolling")
                ])
            },
            {
                "name": "column_operations",
                "pandas": lambda df: df.assign(
                    temp_celsius=df["temperature"],
                    temp_fahrenheit=df["temperature"] * 9/5 + 32,
                    temp_kelvin=df["temperature"] + 273.15
                ),
                "polars": lambda df: df.with_columns([
                    pl.col("temperature").alias("temp_celsius"),
                    (pl.col("temperature") * 9/5 + 32).alias("temp_fahrenheit"),
                    (pl.col("temperature") + 273.15).alias("temp_kelvin")
                ])
            },
            {
                "name": "datetime_operations",
                "pandas": lambda df: self._pandas_datetime_ops(df),
                "polars": lambda df: self._polars_datetime_ops(df)
            },
            {
                "name": "missing_value_handling",
                "pandas": lambda df: df.fillna(df.mean(numeric_only=True)),
                "polars": lambda df: df.fill_null(strategy="mean")
            },
            {
                "name": "quantile_calculation",
                "pandas": lambda df: df[["temperature", "humidity", "pressure"]].quantile([0.25, 0.5, 0.75]),
                "polars": lambda df: df.select([
                    pl.col("temperature").quantile(0.25).alias("temp_q25"),
                    pl.col("temperature").quantile(0.5).alias("temp_median"),
                    pl.col("temperature").quantile(0.75).alias("temp_q75"),
                    pl.col("humidity").quantile(0.25).alias("humidity_q25"),
                    pl.col("humidity").quantile(0.5).alias("humidity_median"),
                    pl.col("humidity").quantile(0.75).alias("humidity_q75")
                ])
            }
        ]
        
        # Ensure required columns exist
        required_columns = ["city", "temperature", "humidity", "pressure"]
        available_columns = list(data.columns)
        
        # Filter benchmarks based on available columns
        valid_benchmarks = []
        for benchmark in benchmark_suite:
            # Simple check - if benchmark name suggests it needs certain columns
            if benchmark["name"] in ["basic_aggregation", "filtering"] and "city" not in available_columns:
                continue
            if "datetime" in benchmark["name"] and "timestamp" not in available_columns:
                continue
            valid_benchmarks.append(benchmark)
            
        results = {
            "benchmark_timestamp": time.time(),
            "data_info": {
                "shape": data.shape,
                "columns": available_columns,
                "size_mb": self._estimate_dataframe_size(data)
            },
            "benchmarks": [],
            "summary": {}
        }
        
        # Run benchmarks
        pandas_wins = 0
        polars_wins = 0
        total_speedups = []
        
        for benchmark in valid_benchmarks:
            print(f"Running benchmark: {benchmark['name']}")
            
            try:
                result = self.benchmark_operation(
                    benchmark["name"],
                    benchmark["pandas"],
                    benchmark["polars"],
                    data
                )
                results["benchmarks"].append(result)
                
                if result["winner"] == "pandas":
                    pandas_wins += 1
                elif result["winner"] == "polars":
                    polars_wins += 1
                    
                if result["speedup_ratio"] is not None:
                    total_speedups.append(result["speedup_ratio"])
                    
            except Exception as e:
                print(f"Benchmark {benchmark['name']} failed: {e}")
                results["benchmarks"].append({
                    "operation": benchmark["name"],
                    "error": str(e),
                    "success": False
                })
                
        # Calculate summary statistics
        results["summary"] = {
            "total_benchmarks": len(valid_benchmarks),
            "pandas_wins": pandas_wins,
            "polars_wins": polars_wins,
            "average_speedup_ratio": np.mean(total_speedups) if total_speedups else None,
            "median_speedup_ratio": np.median(total_speedups) if total_speedups else None,
            "overall_winner": "polars" if polars_wins > pandas_wins else "pandas"
        }
        
        return results
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        if not self.benchmark_results:
            return {"error": "No benchmark results available"}
            
        # Aggregate results
        total_benchmarks = len(self.benchmark_results)
        pandas_wins = sum(1 for r in self.benchmark_results if r.get("winner") == "pandas")
        polars_wins = sum(1 for r in self.benchmark_results if r.get("winner") == "polars")
        
        speedup_ratios = [r["speedup_ratio"] for r in self.benchmark_results if r.get("speedup_ratio")]
        
        report = {
            "report_timestamp": time.time(),
            "total_benchmarks": total_benchmarks,
            "pandas_wins": pandas_wins,
            "polars_wins": polars_wins,
            "benchmark_results": self.benchmark_results,
            "summary_statistics": {
                "average_speedup": np.mean(speedup_ratios) if speedup_ratios else None,
                "median_speedup": np.median(speedup_ratios) if speedup_ratios else None,
                "max_speedup": np.max(speedup_ratios) if speedup_ratios else None,
                "min_speedup": np.min(speedup_ratios) if speedup_ratios else None
            },
            "recommendations": self._generate_performance_recommendations()
        }
        
        return report
        
    def benchmark_memory_usage(self, 
                              data_sizes: List[int],
                              operations: List[str]) -> Dict[str, Any]:
        """
        Benchmark memory usage for different data sizes.
        
        Args:
            data_sizes: List of row counts to test
            operations: Operations to benchmark
            
        Returns:
            Memory usage comparison results
        """
        results = {
            "timestamp": time.time(),
            "data_sizes": data_sizes,
            "operations": operations,
            "memory_benchmarks": []
        }
        
        for size in data_sizes:
            print(f"Testing data size: {size} rows")
            
            # Generate test data
            test_data = self._generate_test_data(size)
            
            size_results = {
                "data_size": size,
                "operations": {}
            }
            
            for operation in operations:
                if operation == "load_data":
                    # Test data loading memory
                    pandas_memory = self._measure_memory_pandas_load(test_data)
                    polars_memory = self._measure_memory_polars_load(test_data)
                    
                    size_results["operations"][operation] = {
                        "pandas_memory_mb": pandas_memory,
                        "polars_memory_mb": polars_memory,
                        "memory_ratio": pandas_memory / polars_memory if polars_memory > 0 else None
                    }
                    
            results["memory_benchmarks"].append(size_results)
            
        return results
        
    def clear_results(self) -> None:
        """Clear all benchmark results."""
        self.benchmark_results.clear()
        
    def _estimate_dataframe_size(self, df: Union[pd.DataFrame, pl.DataFrame]) -> float:
        """Estimate DataFrame size in MB."""
        if isinstance(df, pd.DataFrame):
            return df.memory_usage(deep=True).sum() / 1024 / 1024
        elif isinstance(df, pl.DataFrame):
            # Polars doesn't have direct memory usage, estimate
            return df.estimated_size("mb")
        return 0.0
        
    def _pandas_datetime_ops(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pandas datetime operations for benchmarking."""
        if "timestamp" not in df.columns:
            return df
            
        result = df.copy()
        result["hour"] = pd.to_datetime(result["timestamp"]).dt.hour
        result["day_of_week"] = pd.to_datetime(result["timestamp"]).dt.dayofweek
        result["month"] = pd.to_datetime(result["timestamp"]).dt.month
        return result
        
    def _polars_datetime_ops(self, df: pl.DataFrame) -> pl.DataFrame:
        """Polars datetime operations for benchmarking."""
        if "timestamp" not in df.columns:
            return df
            
        return df.with_columns([
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.weekday().alias("day_of_week"),
            pl.col("timestamp").dt.month().alias("month")
        ])
        
    def _generate_test_data(self, n_rows: int) -> pd.DataFrame:
        """Generate test weather data for benchmarking."""
        np.random.seed(42)  # For reproducibility
        
        cities = ["New York", "London", "Tokyo", "Sydney", "Paris"] * (n_rows // 5 + 1)
        cities = cities[:n_rows]
        
        data = {
            "city": cities,
            "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="H"),
            "temperature": np.random.normal(20, 10, n_rows),
            "humidity": np.random.randint(30, 90, n_rows),
            "pressure": np.random.normal(1013, 20, n_rows),
            "wind_speed": np.random.exponential(5, n_rows),
            "precipitation": np.random.exponential(0.5, n_rows)
        }
        
        return pd.DataFrame(data)
        
    def _measure_memory_pandas_load(self, data: pd.DataFrame) -> float:
        """Measure memory usage for pandas data loading."""
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Create copy to simulate loading
        _ = data.copy()
        
        memory_after = process.memory_info().rss / 1024 / 1024
        return memory_after - memory_before
        
    def _measure_memory_polars_load(self, data: pd.DataFrame) -> float:
        """Measure memory usage for polars data loading."""
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Convert to Polars to simulate loading
        _ = pl.from_pandas(data)
        
        memory_after = process.memory_info().rss / 1024 / 1024
        return memory_after - memory_before
        
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations based on benchmark results."""
        if not self.benchmark_results:
            return ["No benchmark data available for recommendations"]
            
        recommendations = []
        
        # Count wins
        pandas_wins = sum(1 for r in self.benchmark_results if r.get("winner") == "pandas")
        polars_wins = sum(1 for r in self.benchmark_results if r.get("winner") == "polars")
        
        if polars_wins > pandas_wins:
            recommendations.append(
                f"Polars outperformed pandas in {polars_wins}/{len(self.benchmark_results)} benchmarks. "
                "Consider using Polars for better performance."
            )
        elif pandas_wins > polars_wins:
            recommendations.append(
                f"Pandas outperformed Polars in {pandas_wins}/{len(self.benchmark_results)} benchmarks. "
                "Pandas may be more suitable for your workload."
            )
        else:
            recommendations.append("Performance is roughly equivalent between pandas and Polars.")
            
        # Specific operation recommendations
        operation_winners = {}
        for result in self.benchmark_results:
            if result.get("winner"):
                operation_winners[result["operation"]] = result["winner"]
                
        if operation_winners:
            polars_strong = [op for op, winner in operation_winners.items() if winner == "polars"]
            pandas_strong = [op for op, winner in operation_winners.items() if winner == "pandas"]
            
            if polars_strong:
                recommendations.append(f"Use Polars for: {', '.join(polars_strong)}")
            if pandas_strong:
                recommendations.append(f"Use pandas for: {', '.join(pandas_strong)}")
                
        # Memory recommendations
        memory_results = [r for r in self.benchmark_results if "pandas" in r and "polars" in r]
        if memory_results:
            avg_pandas_memory = np.mean([r["pandas"].get("avg_memory_mb", 0) for r in memory_results])
            avg_polars_memory = np.mean([r["polars"].get("avg_memory_mb", 0) for r in memory_results])
            
            if avg_polars_memory < avg_pandas_memory * 0.8:
                recommendations.append("Polars generally uses less memory - consider for memory-constrained environments")
                
        return recommendations
