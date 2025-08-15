"""Data quality monitoring and alerting for weather data."""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from ..models.weather import DataQualityMetrics, WeatherDataPoint


class DataQualityMonitor:
    """Comprehensive data quality monitoring for weather data."""
    
    def __init__(self, alerting_enabled: bool = True) -> None:
        self.alerting_enabled = alerting_enabled
        self.quality_thresholds = {
            "completeness_threshold": 0.8,
            "outlier_threshold": 0.05,  # 5% outliers max
            "consistency_threshold": 0.9,
            "freshness_threshold_hours": 2,
            "temperature_range": (-50, 60),  # Celsius
            "humidity_range": (0, 100),
            "pressure_range": (800, 1100),  # hPa
            "wind_speed_range": (0, 200),  # m/s (hurricane max ~85 m/s)
            "precipitation_range": (0, 500)  # mm/hour max
        }
        self.alerts: List[Dict[str, Any]] = []
        
    def assess_data_quality(
        self, 
        data: Union[pd.DataFrame, pl.DataFrame],
        check_types: Optional[List[str]] = None
    ) -> DataQualityMetrics:
        """
        Comprehensive data quality assessment.
        
        Args:
            data: Weather data to assess
            check_types: Types of quality checks to perform
            
        Returns:
            DataQualityMetrics with assessment results
        """
        if check_types is None:
            check_types = [
                "completeness", "validity", "consistency", 
                "outliers", "duplicates", "temporal"
            ]
            
        if isinstance(data, pl.DataFrame):
            data_pd = data.to_pandas()
        else:
            data_pd = data.copy()
            
        total_records = len(data_pd)
        if total_records == 0:
            return self._create_empty_metrics()
            
        # Initialize quality metrics
        quality_results = {
            "total_records": total_records,
            "valid_records": total_records,
            "missing_temperature": 0,
            "missing_humidity": 0,
            "outliers_detected": 0,
            "checks_performed": check_types
        }
        
        # Run quality checks
        for check_type in check_types:
            if check_type == "completeness":
                completeness_results = self._check_completeness(data_pd)
                quality_results.update(completeness_results)
                
            elif check_type == "validity":
                validity_results = self._check_validity(data_pd)
                quality_results.update(validity_results)
                
            elif check_type == "consistency":
                consistency_results = self._check_consistency(data_pd)
                quality_results.update(consistency_results)
                
            elif check_type == "outliers":
                outlier_results = self._detect_outliers(data_pd)
                quality_results.update(outlier_results)
                
            elif check_type == "duplicates":
                duplicate_results = self._check_duplicates(data_pd)
                quality_results.update(duplicate_results)
                
            elif check_type == "temporal":
                temporal_results = self._check_temporal_consistency(data_pd)
                quality_results.update(temporal_results)
                
        # Calculate overall scores
        completeness_score = self._calculate_completeness_score(quality_results)
        overall_quality_score = self._calculate_overall_quality_score(quality_results)
        
        # Create time range info
        time_range_start = None
        time_range_end = None
        
        if "timestamp" in data_pd.columns:
            time_range_start = data_pd["timestamp"].min()
            time_range_end = data_pd["timestamp"].max()
            
        # Create metrics object
        metrics = DataQualityMetrics(
            total_records=total_records,
            valid_records=quality_results.get("valid_records", total_records),
            missing_temperature=quality_results.get("missing_temperature", 0),
            missing_humidity=quality_results.get("missing_humidity", 0),
            outliers_detected=quality_results.get("outliers_detected", 0),
            completeness_score=completeness_score,
            quality_score=overall_quality_score,
            data_time_range_start=time_range_start,
            data_time_range_end=time_range_end
        )
        
        # Generate alerts if enabled
        if self.alerting_enabled:
            self._generate_alerts(quality_results, metrics)
            
        return metrics
        
    def check_data_freshness(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        timestamp_col: str = "timestamp"
    ) -> Dict[str, Any]:
        """
        Check data freshness and recency.
        
        Args:
            data: Weather data
            timestamp_col: Timestamp column name
            
        Returns:
            Freshness assessment results
        """
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
            
        if timestamp_col not in data.columns:
            return {"error": f"Timestamp column '{timestamp_col}' not found"}
            
        if len(data) == 0:
            return {"error": "No data provided"}
            
        current_time = datetime.utcnow()
        timestamps = pd.to_datetime(data[timestamp_col])
        latest_timestamp = timestamps.max()
        oldest_timestamp = timestamps.min()
        
        # Calculate freshness
        time_since_latest = current_time - latest_timestamp
        hours_since_latest = max(0, time_since_latest.total_seconds() / 3600)
        
        data_span_hours = (latest_timestamp - oldest_timestamp).total_seconds() / 3600
        
        results = {
            "latest_data_time": latest_timestamp,
            "oldest_data_time": oldest_timestamp,
            "hours_since_latest": hours_since_latest,
            "data_span_hours": data_span_hours,
            "is_fresh": hours_since_latest <= self.quality_thresholds["freshness_threshold_hours"],
            "freshness_status": self._get_freshness_status(hours_since_latest)
        }
        
        # Alert if data is stale
        if self.alerting_enabled and not results["is_fresh"]:
            self._add_alert({
                "type": "freshness",
                "severity": "warning",
                "message": f"Data is {hours_since_latest:.1f} hours old",
                "timestamp": current_time,
                "details": results
            })
            
        return results
        
    def validate_schema(
        self,
        data: Union[pd.DataFrame, pl.DataFrame],
        expected_columns: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate data schema against expected structure.
        
        Args:
            data: Data to validate
            expected_columns: Expected columns
            required_columns: Required columns that must be present
            
        Returns:
            Schema validation results
        """
        if isinstance(data, pl.DataFrame):
            actual_columns = data.columns
        else:
            actual_columns = list(data.columns)
            
        if expected_columns is None:
            expected_columns = [
                "timestamp", "temperature", "humidity", "pressure",
                "wind_speed", "wind_direction", "precipitation", "city"
            ]
            
        if required_columns is None:
            required_columns = ["timestamp", "temperature", "city"]
            
        results = {
            "expected_columns": expected_columns,
            "actual_columns": actual_columns,
            "required_columns": required_columns
        }
        
        # Check missing columns
        missing_columns = [col for col in expected_columns if col not in actual_columns]
        missing_required = [col for col in required_columns if col not in actual_columns]
        extra_columns = [col for col in actual_columns if col not in expected_columns]
        
        results.update({
            "missing_columns": missing_columns,
            "missing_required_columns": missing_required,
            "extra_columns": extra_columns,
            "schema_valid": len(missing_required) == 0,
            "completeness_score": 1 - len(missing_columns) / len(expected_columns)
        })
        
        # Generate alerts for schema issues
        if self.alerting_enabled and missing_required:
            self._add_alert({
                "type": "schema",
                "severity": "error", 
                "message": f"Missing required columns: {missing_required}",
                "timestamp": datetime.utcnow(),
                "details": results
            })
            
        return results
        
    def get_quality_report(
        self,
        data: Union[pd.DataFrame, pl.DataFrame]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            data: Weather data to analyze
            
        Returns:
            Comprehensive quality report
        """
        report = {
            "report_timestamp": datetime.utcnow(),
            "data_summary": self._get_data_summary(data),
            "quality_metrics": self.assess_data_quality(data),
            "freshness_check": self.check_data_freshness(data),
            "schema_validation": self.validate_schema(data),
            "alerts": self.get_recent_alerts()
        }
        
        # Add recommendations
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
        
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent quality alerts."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts 
            if alert.get("timestamp", datetime.min) >= cutoff_time
        ]
        
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
        
    def _check_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness."""
        results = {}
        
        key_columns = ["temperature", "humidity", "pressure", "wind_speed"]
        
        for col in key_columns:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                results[f"missing_{col}"] = int(missing_count)
                results[f"{col}_completeness"] = 1 - (missing_count / len(data))
                
        return results
        
    def _check_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check value validity against expected ranges."""
        results = {"validity_issues": []}
        invalid_count = 0
        
        range_checks = {
            "temperature": self.quality_thresholds["temperature_range"],
            "humidity": self.quality_thresholds["humidity_range"],
            "pressure": self.quality_thresholds["pressure_range"],
            "wind_speed": self.quality_thresholds["wind_speed_range"],
            "precipitation": self.quality_thresholds["precipitation_range"]
        }
        
        for col, (min_val, max_val) in range_checks.items():
            if col in data.columns:
                out_of_range = ((data[col] < min_val) | (data[col] > max_val)).sum()
                if out_of_range > 0:
                    invalid_count += out_of_range
                    results["validity_issues"].append({
                        "column": col,
                        "out_of_range_count": int(out_of_range),
                        "expected_range": [min_val, max_val],
                        "actual_range": [float(data[col].min()), float(data[col].max())]
                    })
                    
        results["invalid_values_total"] = invalid_count
        return results
        
    def _check_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check cross-variable consistency."""
        results = {"consistency_issues": []}
        inconsistent_count = 0
        
        # Temperature-humidity relationship checks
        if "temperature" in data.columns and "humidity" in data.columns:
            # Very high humidity with very low temperature is unusual
            unusual_combo = ((data["temperature"] < -20) & (data["humidity"] > 90)).sum()
            if unusual_combo > 0:
                inconsistent_count += unusual_combo
                results["consistency_issues"].append({
                    "type": "temperature_humidity_inconsistency",
                    "count": int(unusual_combo),
                    "description": "Very low temperature with very high humidity"
                })
                
        # Wind speed and direction consistency
        if "wind_speed" in data.columns and "wind_direction" in data.columns:
            # Wind direction without wind speed
            direction_no_speed = ((data["wind_direction"].notna()) & 
                                (data["wind_speed"] == 0)).sum()
            if direction_no_speed > 0:
                inconsistent_count += direction_no_speed
                results["consistency_issues"].append({
                    "type": "wind_direction_without_speed",
                    "count": int(direction_no_speed),
                    "description": "Wind direction specified but wind speed is zero"
                })
                
        results["inconsistent_records_total"] = inconsistent_count
        return results
        
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical outliers."""
        results = {"outliers_by_column": {}}
        total_outliers = 0
        
        numeric_columns = ["temperature", "humidity", "pressure", "wind_speed", "precipitation"]
        
        for col in numeric_columns:
            if col in data.columns and len(data[col].dropna()) > 0:
                # Use IQR method for outlier detection
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    total_outliers += outlier_count
                    results["outliers_by_column"][col] = {
                        "count": int(outlier_count),
                        "percentage": float(outlier_count / len(data) * 100),
                        "bounds": [float(lower_bound), float(upper_bound)],
                        "outlier_values": data[col][outliers].tolist()[:10]  # First 10
                    }
                    
        results["outliers_detected"] = total_outliers
        return results
        
    def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate records."""
        # Check for complete duplicates
        complete_duplicates = data.duplicated().sum()
        
        # Check for duplicates based on key columns
        key_cols = ["timestamp", "city"] if all(col in data.columns for col in ["timestamp", "city"]) else []
        key_duplicates = 0
        
        if key_cols:
            key_duplicates = data.duplicated(subset=key_cols).sum()
            
        return {
            "complete_duplicates": int(complete_duplicates),
            "key_column_duplicates": int(key_duplicates),
            "duplicate_check_columns": key_cols
        }
        
    def _check_temporal_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check temporal consistency."""
        if "timestamp" not in data.columns:
            return {"error": "No timestamp column for temporal checks"}
            
        timestamps = pd.to_datetime(data["timestamp"]).sort_values()
        
        if len(timestamps) < 2:
            return {"error": "Insufficient data for temporal consistency checks"}
            
        # Calculate time gaps
        time_diffs = timestamps.diff().dropna()
        median_gap = time_diffs.median()
        
        # Find large gaps (more than 2x median)
        large_gaps = time_diffs[time_diffs > 2 * median_gap]
        
        # Check for future timestamps
        current_time = datetime.utcnow().replace(tzinfo=timestamps.dt.tz)
        future_timestamps = (timestamps > current_time).sum()
        
        return {
            "median_time_gap": str(median_gap),
            "large_gaps_count": len(large_gaps),
            "max_gap": str(time_diffs.max()),
            "future_timestamps": int(future_timestamps),
            "temporal_ordering_valid": timestamps.is_monotonic_increasing
        }
        
    def _calculate_completeness_score(self, quality_results: Dict[str, Any]) -> float:
        """Calculate overall completeness score."""
        completeness_scores = [
            v for k, v in quality_results.items() 
            if k.endswith("_completeness") and isinstance(v, (int, float))
        ]
        
        if completeness_scores:
            return float(np.mean(completeness_scores))
        return 1.0
        
    def _calculate_overall_quality_score(self, quality_results: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        total_records = quality_results.get("total_records", 1)
        
        # Weight different quality aspects
        weights = {
            "completeness": 0.3,
            "validity": 0.3,
            "consistency": 0.2,
            "outliers": 0.1,
            "duplicates": 0.1
        }
        
        scores = {}
        
        # Completeness score
        scores["completeness"] = self._calculate_completeness_score(quality_results)
        
        # Validity score
        invalid_total = quality_results.get("invalid_values_total", 0)
        scores["validity"] = max(0, 1 - invalid_total / total_records)
        
        # Consistency score  
        inconsistent_total = quality_results.get("inconsistent_records_total", 0)
        scores["consistency"] = max(0, 1 - inconsistent_total / total_records)
        
        # Outlier score
        outliers_total = quality_results.get("outliers_detected", 0)
        scores["outliers"] = max(0, 1 - outliers_total / total_records)
        
        # Duplicate score
        duplicates_total = quality_results.get("complete_duplicates", 0)
        scores["duplicates"] = max(0, 1 - duplicates_total / total_records)
        
        # Calculate weighted average
        overall_score = sum(scores[aspect] * weights[aspect] for aspect in weights)
        return float(overall_score)
        
    def _generate_alerts(self, quality_results: Dict[str, Any], metrics: DataQualityMetrics) -> None:
        """Generate quality alerts based on thresholds."""
        current_time = datetime.utcnow()
        
        # Completeness alerts
        if metrics.completeness_score < self.quality_thresholds["completeness_threshold"]:
            self._add_alert({
                "type": "completeness",
                "severity": "warning",
                "message": f"Data completeness below threshold: {metrics.completeness_score:.2f}",
                "timestamp": current_time,
                "threshold": self.quality_thresholds["completeness_threshold"]
            })
            
        # Overall quality alerts
        if metrics.quality_score < 0.7:
            severity = "error" if metrics.quality_score < 0.5 else "warning"
            self._add_alert({
                "type": "quality",
                "severity": severity,
                "message": f"Overall quality score low: {metrics.quality_score:.2f}",
                "timestamp": current_time,
                "details": quality_results
            })
            
        # Outlier alerts
        outlier_rate = metrics.outliers_detected / metrics.total_records
        if outlier_rate > self.quality_thresholds["outlier_threshold"]:
            self._add_alert({
                "type": "outliers",
                "severity": "warning",
                "message": f"High outlier rate: {outlier_rate:.1%}",
                "timestamp": current_time,
                "outlier_count": metrics.outliers_detected
            })
            
    def _add_alert(self, alert: Dict[str, Any]) -> None:
        """Add alert to the alert list."""
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 7 days)
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        self.alerts = [
            a for a in self.alerts 
            if a.get("timestamp", datetime.min) >= cutoff_time
        ]
        
    def _get_freshness_status(self, hours_since_latest: float) -> str:
        """Get freshness status description."""
        if hours_since_latest <= 1:
            return "very_fresh"
        elif hours_since_latest <= 2:
            return "fresh"
        elif hours_since_latest <= 6:
            return "acceptable"
        elif hours_since_latest <= 24:
            return "stale"
        else:
            return "very_stale"
            
    def _get_data_summary(self, data: Union[pd.DataFrame, pl.DataFrame]) -> Dict[str, Any]:
        """Get basic data summary statistics."""
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
            
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        summary = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "numeric_columns": len(numeric_cols),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024**2
        }
        
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = data[numeric_cols].describe().to_dict()
            
        return summary
        
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        metrics = report.get("quality_metrics")
        if metrics and hasattr(metrics, 'quality_score'):
            if metrics.quality_score < 0.8:
                recommendations.append("Overall data quality is below recommended threshold (0.8)")
                
            if metrics.completeness_score < 0.9:
                recommendations.append("Consider improving data collection to reduce missing values")
                
        freshness = report.get("freshness_check", {})
        if not freshness.get("is_fresh", True):
            recommendations.append("Data freshness is below threshold - check data ingestion pipeline")
            
        schema = report.get("schema_validation", {})
        if schema.get("missing_required_columns"):
            recommendations.append("Add missing required columns to improve data schema compliance")
            
        alerts = report.get("alerts", [])
        high_priority_alerts = [a for a in alerts if a.get("severity") == "error"]
        if high_priority_alerts:
            recommendations.append(f"Address {len(high_priority_alerts)} critical data quality issues")
            
        if not recommendations:
            recommendations.append("Data quality appears to be in good condition")
            
        return recommendations
        
    def _create_empty_metrics(self) -> DataQualityMetrics:
        """Create empty quality metrics for empty dataset."""
        return DataQualityMetrics(
            total_records=0,
            valid_records=0,
            missing_temperature=0,
            missing_humidity=0,
            outliers_detected=0,
            completeness_score=0.0,
            quality_score=0.0
        )
