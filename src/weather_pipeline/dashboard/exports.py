"""Export functionality for dashboard reports and data."""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class ExportManager:
    """Manages data export functionality for the dashboard."""
    
    def __init__(self):
        """Initialize the export manager."""
        pass
    
    def export_excel_report(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Export weather data to Excel format.
        
        Args:
            data: List of weather data points
            
        Returns:
            Dash download data dictionary
        """
        if not data:
            logger.warning("No data available for Excel export")
            return {}
        
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            
            # Clean and format data
            df = self._prepare_data_for_export(df)
            
            # Create Excel file in memory
            output = io.BytesIO()
            
            # Use simpler Excel writer to avoid formatting issues
            try:
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Main data sheet
                    df.to_excel(writer, sheet_name='Weather Data', index=False)
                    
                    # Summary statistics sheet
                    summary_df = self._create_summary_statistics(df)
                    summary_df.to_excel(writer, sheet_name='Summary Statistics', index=True)
                    
                    # City comparison sheet
                    if 'city' in df.columns:
                        city_comparison = self._create_city_comparison(df)
                        city_comparison.to_excel(writer, sheet_name='City Comparison', index=True)
                
            except ImportError:
                # Fallback to xlsxwriter if openpyxl not available
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Main data sheet
                    df.to_excel(writer, sheet_name='Weather Data', index=False)
                    
                    # Summary statistics sheet
                    summary_df = self._create_summary_statistics(df)
                    summary_df.to_excel(writer, sheet_name='Summary Statistics', index=True)
                    
                    # City comparison sheet
                    if 'city' in df.columns:
                        city_comparison = self._create_city_comparison(df)
                        city_comparison.to_excel(writer, sheet_name='City Comparison', index=True)
            
            output.seek(0)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_data_report_{timestamp}.xlsx"
            
            # Encode binary data to base64 for JSON serialization
            content_b64 = base64.b64encode(output.getvalue()).decode('utf-8')
            
            return {
                "content": content_b64,
                "filename": filename,
                "base64": True,
                "type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }
            
        except Exception as e:
            logger.error(f"Failed to export Excel report: {e}", exc_info=True)
            # Return a simple fallback CSV instead of empty dict
            try:
                df = pd.DataFrame(data)
                csv_content = df.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"weather_data_fallback_{timestamp}.csv"
                
                return {
                    "content": csv_content,
                    "filename": filename,
                    "type": "text/csv"
                }
            except Exception as fallback_error:
                logger.error(f"Fallback CSV export also failed: {fallback_error}")
                return {}
    
    def export_pdf_report(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Export weather data to PDF format.
        
        Args:
            data: List of weather data points
            
        Returns:
            Dash download data dictionary
        """
        if not data:
            logger.warning("No data available for PDF export")
            return {}
        
        try:
            # For PDF export, we'll create a detailed HTML report and convert to PDF
            # In a full implementation, you'd use libraries like reportlab or weasyprint
            
            df = pd.DataFrame(data)
            df = self._prepare_data_for_export(df)
            
            # Create HTML report
            html_content = self._create_html_report(df)
            
            # For now, return HTML (in production, convert to PDF)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_data_report_{timestamp}.html"
            
            # Encode to base64 for JSON serialization
            content_b64 = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            
            return {
                "content": content_b64,
                "filename": filename,
                "base64": True,
                "type": "text/html"
            }
            
        except Exception as e:
            logger.error(f"Failed to export PDF report: {e}", exc_info=True)
            # Return a simple text report instead of empty dict
            try:
                df = pd.DataFrame(data)
                # Create simple text summary
                summary = f"""Weather Data Report (Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                
Data Summary:
- Total records: {len(df)}
- Cities: {', '.join(df['city'].unique()) if 'city' in df.columns else 'N/A'}
- Date range: {df['timestamp'].min()} to {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}

Temperature Statistics:
- Average: {df['temperature'].mean():.1f}°C
- Min: {df['temperature'].min():.1f}°C  
- Max: {df['temperature'].max():.1f}°C
                """
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"weather_summary_{timestamp}.txt"
                
                # Encode to base64 for JSON serialization
                content_b64 = base64.b64encode(summary.encode('utf-8')).decode('utf-8')
                
                return {
                    "content": content_b64,
                    "filename": filename,
                    "base64": True,
                    "type": "text/plain"
                }
            except Exception as fallback_error:
                logger.error(f"Fallback text export also failed: {fallback_error}")
                return {}
    
    def export_csv_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Export weather data to CSV format.
        
        Args:
            data: List of weather data points
            
        Returns:
            Dash download data dictionary
        """
        if not data:
            logger.warning("No data available for CSV export")
            return {}
        
        try:
            df = pd.DataFrame(data)
            df = self._prepare_data_for_export(df)
            
            # Convert to CSV
            csv_content = df.to_csv(index=False)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_data_{timestamp}.csv"
            
            return {
                "content": csv_content,
                "filename": filename,
                "type": "text/csv"
            }
            
        except Exception as e:
            logger.error(f"Failed to export CSV data: {e}")
            return {}
    
    def _prepare_data_for_export(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for export by cleaning and formatting.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Cleaned and formatted DataFrame
        """
        # Make a copy to avoid modifying original
        export_df = df.copy()
        
        # Format timestamp
        if 'timestamp' in export_df.columns:
            export_df['timestamp'] = pd.to_datetime(export_df['timestamp'])
            export_df['date'] = export_df['timestamp'].dt.date
            export_df['time'] = export_df['timestamp'].dt.time
        
        # Flatten coordinates if present
        if 'coordinates' in export_df.columns:
            coord_df = pd.json_normalize(export_df['coordinates'])
            export_df['latitude'] = coord_df.get('latitude')
            export_df['longitude'] = coord_df.get('longitude')
            export_df = export_df.drop('coordinates', axis=1)
        
        # Round numerical columns
        numeric_columns = export_df.select_dtypes(include=['float64', 'float32']).columns
        export_df[numeric_columns] = export_df[numeric_columns].round(2)
        
        # Reorder columns for better readability
        preferred_order = [
            'timestamp', 'date', 'time', 'city', 'country',
            'latitude', 'longitude', 'temperature', 'humidity',
            'pressure', 'wind_speed', 'wind_direction',
            'precipitation', 'visibility', 'cloud_cover',
            'uv_index', 'provider', 'is_forecast', 'confidence_score'
        ]
        
        # Keep only existing columns in preferred order
        existing_columns = [col for col in preferred_order if col in export_df.columns]
        remaining_columns = [col for col in export_df.columns if col not in existing_columns]
        
        column_order = existing_columns + remaining_columns
        export_df = export_df[column_order]
        
        return export_df
    
    def _create_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics for the data.
        
        Args:
            df: Weather data DataFrame
            
        Returns:
            Summary statistics DataFrame
        """
        numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        existing_numeric = [col for col in numeric_columns if col in df.columns]
        
        if not existing_numeric:
            return pd.DataFrame()
        
        summary = df[existing_numeric].describe()
        
        # Add additional statistics
        summary.loc['median'] = df[existing_numeric].median()
        summary.loc['mode'] = df[existing_numeric].mode().iloc[0] if len(df) > 0 else 0
        
        return summary
    
    def _create_city_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create city comparison table.
        
        Args:
            df: Weather data DataFrame
            
        Returns:
            City comparison DataFrame
        """
        if 'city' not in df.columns:
            return pd.DataFrame()
        
        numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed']
        existing_numeric = [col for col in numeric_columns if col in df.columns]
        
        if not existing_numeric:
            return pd.DataFrame()
        
        city_stats = df.groupby('city')[existing_numeric].agg(['mean', 'min', 'max', 'std']).round(2)
        
        return city_stats
    
    def _format_excel_worksheet(self, workbook, worksheet, df: pd.DataFrame) -> None:
        """Format Excel worksheet with colors and styles.
        
        Args:
            workbook: XlsxWriter workbook object
            worksheet: XlsxWriter worksheet object
            df: Data DataFrame
        """
        # Create formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Format headers
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Set column widths
        for i, col in enumerate(df.columns):
            # Calculate width based on column content
            max_len = max(
                df[col].astype(str).map(len).max(),  # Max length of values
                len(str(col))  # Length of column name
            )
            worksheet.set_column(i, i, min(max_len + 2, 50))
    
    def _create_html_report(self, df: pd.DataFrame) -> str:
        """Create an HTML report with charts and tables.
        
        Args:
            df: Weather data DataFrame
            
        Returns:
            HTML report string
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Weather Data Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                .chart-placeholder {{ 
                    background-color: #ecf0f1; 
                    height: 300px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    margin: 20px 0;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>Weather Data Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                {summary_metrics}
            </div>
            
            <div class="chart-placeholder">
                <p>Charts would be rendered here in a full PDF implementation</p>
            </div>
            
            <h2>Detailed Data</h2>
            {data_table}
            
            {city_comparison}
        </body>
        </html>
        """
        
        # Generate summary metrics
        summary_metrics = ""
        if not df.empty:
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns[:4]:  # Show first 4 metrics
                if col in df.columns:
                    avg_val = df[col].mean()
                    summary_metrics += f"""
                    <div class="metric">
                        <div class="metric-value">{avg_val:.1f}</div>
                        <div class="metric-label">Avg {col.replace('_', ' ').title()}</div>
                    </div>
                    """
        
        # Generate data table (show first 100 rows)
        display_df = df.head(100) if len(df) > 100 else df
        data_table = display_df.to_html(classes='data-table', table_id='weather-data')
        
        # Generate city comparison if available
        city_comparison = ""
        if 'city' in df.columns and len(df['city'].unique()) > 1:
            city_comp_df = self._create_city_comparison(df)
            if not city_comp_df.empty:
                city_comparison = f"""
                <h2>City Comparison</h2>
                {city_comp_df.to_html(classes='comparison-table')}
                """
        
        return html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary_metrics=summary_metrics,
            data_table=data_table,
            city_comparison=city_comparison
        )
