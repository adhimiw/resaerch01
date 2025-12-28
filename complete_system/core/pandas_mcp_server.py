"""
Comprehensive Pandas MCP Server with ALL Data Science Tools
Production-grade implementation with error handling
"""

import json
import sys
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PandasMCPServer:
    """
    Complete pandas MCP server with comprehensive data science capabilities
    """
    
    def __init__(self):
        self.dataframes = {}  # Store loaded dataframes
        self.history = []  # Track operations
        
    def log_operation(self, operation: str, status: str, details: Dict):
        """Log all operations for debugging"""
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'status': status,
            'details': details
        })
    
    # ==================== DATA LOADING ====================
    
    def read_csv(self, filepath: str, name: str = 'df', **kwargs) -> Dict:
        """
        Read CSV file with comprehensive error handling
        
        Args:
            filepath: Path to CSV file
            name: Name to store dataframe as
            **kwargs: Additional pandas read_csv arguments
        """
        try:
            df = pd.read_csv(filepath, **kwargs)
            self.dataframes[name] = df
            
            result = {
                'status': 'success',
                'name': name,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'head': df.head(3).to_dict('records')
            }
            
            self.log_operation('read_csv', 'success', result)
            return result
            
        except Exception as e:
            result = {'status': 'error', 'message': str(e), 'filepath': filepath}
            self.log_operation('read_csv', 'error', result)
            return result
    
    def read_excel(self, filepath: str, name: str = 'df', **kwargs) -> Dict:
        """Read Excel file"""
        try:
            df = pd.read_excel(filepath, **kwargs)
            self.dataframes[name] = df
            return {'status': 'success', 'name': name, 'shape': df.shape}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    # ==================== DATA EXPLORATION ====================
    
    def info(self, name: str) -> Dict:
        """Get comprehensive dataframe information"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        df = self.dataframes[name]
        
        return {
            'status': 'success',
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': int(df.duplicated().sum())
        }
    
    def describe(self, name: str, include='all') -> Dict:
        """Statistical summary of dataframe"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        df = self.dataframes[name]
        desc = df.describe(include=include)
        
        return {
            'status': 'success',
            'statistics': desc.to_dict()
        }
    
    def head(self, name: str, n: int = 5) -> Dict:
        """Get first n rows"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        df = self.dataframes[name]
        return {
            'status': 'success',
            'data': df.head(n).to_dict('records')
        }
    
    def tail(self, name: str, n: int = 5) -> Dict:
        """Get last n rows"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        df = self.dataframes[name]
        return {
            'status': 'success',
            'data': df.tail(n).to_dict('records')
        }
    
    def value_counts(self, name: str, column: str, top_n: int = 10) -> Dict:
        """Get value counts for a column"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        df = self.dataframes[name]
        if column not in df.columns:
            return {'status': 'error', 'message': f'Column {column} not found'}
        
        counts = df[column].value_counts().head(top_n)
        
        return {
            'status': 'success',
            'column': column,
            'value_counts': counts.to_dict(),
            'total_unique': int(df[column].nunique())
        }
    
    # ==================== DATA ANALYSIS ====================
    
    def correlation(self, name: str, method: str = 'pearson', columns: Optional[List[str]] = None) -> Dict:
        """Calculate correlation matrix"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        df = self.dataframes[name]
        
        # Select numeric columns
        if columns:
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'status': 'error', 'message': 'No numeric columns found'}
        
        corr = numeric_df.corr(method=method)
        
        return {
            'status': 'success',
            'method': method,
            'correlation_matrix': corr.to_dict(),
            'strong_correlations': self._find_strong_correlations(corr, threshold=0.7)
        }
    
    def _find_strong_correlations(self, corr_matrix, threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations"""
        strong_corrs = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        strong_corrs.append({
                            'col1': col1,
                            'col2': col2,
                            'correlation': float(corr_value)
                        })
        
        return sorted(strong_corrs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def groupby(self, name: str, by: Union[str, List[str]], agg: Dict) -> Dict:
        """Group by and aggregate"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        try:
            df = self.dataframes[name]
            grouped = df.groupby(by).agg(agg)
            
            return {
                'status': 'success',
                'grouped_by': by if isinstance(by, list) else [by],
                'aggregations': agg,
                'result': grouped.reset_index().to_dict('records'),
                'result_shape': grouped.shape
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def pivot_table(self, name: str, values: str, index: str, columns: str, 
                    aggfunc: str = 'mean') -> Dict:
        """Create pivot table"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        try:
            df = self.dataframes[name]
            pivot = pd.pivot_table(df, values=values, index=index, 
                                   columns=columns, aggfunc=aggfunc)
            
            return {
                'status': 'success',
                'pivot_table': pivot.to_dict()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    # ==================== DATA FILTERING ====================
    
    def query(self, name: str, query_string: str, result_name: Optional[str] = None) -> Dict:
        """Filter dataframe using query string"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        try:
            df = self.dataframes[name]
            result = df.query(query_string)
            
            if result_name:
                self.dataframes[result_name] = result
            
            return {
                'status': 'success',
                'query': query_string,
                'original_shape': df.shape,
                'result_shape': result.shape,
                'rows_filtered': len(result),
                'result_name': result_name,
                'sample_data': result.head(5).to_dict('records')
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'query': query_string}
    
    def filter_by_value(self, name: str, column: str, operator: str, value: Any,
                        result_name: Optional[str] = None) -> Dict:
        """Filter by column value"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        try:
            df = self.dataframes[name]
            
            if operator == '==':
                result = df[df[column] == value]
            elif operator == '>':
                result = df[df[column] > value]
            elif operator == '<':
                result = df[df[column] < value]
            elif operator == '>=':
                result = df[df[column] >= value]
            elif operator == '<=':
                result = df[df[column] <= value]
            elif operator == '!=':
                result = df[df[column] != value]
            elif operator == 'in':
                result = df[df[column].isin(value)]
            else:
                return {'status': 'error', 'message': f'Invalid operator: {operator}'}
            
            if result_name:
                self.dataframes[result_name] = result
            
            return {
                'status': 'success',
                'filter': f'{column} {operator} {value}',
                'rows_filtered': len(result),
                'result_name': result_name
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    # ==================== DATA CLEANING ====================
    
    def drop_duplicates(self, name: str, subset: Optional[List[str]] = None, 
                        inplace: bool = False) -> Dict:
        """Drop duplicate rows"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        df = self.dataframes[name]
        original_len = len(df)
        
        if inplace:
            df.drop_duplicates(subset=subset, inplace=True)
            new_len = len(df)
        else:
            new_df = df.drop_duplicates(subset=subset)
            new_len = len(new_df)
        
        return {
            'status': 'success',
            'original_rows': original_len,
            'new_rows': new_len,
            'duplicates_removed': original_len - new_len,
            'inplace': inplace
        }
    
    def drop_na(self, name: str, subset: Optional[List[str]] = None, 
                how: str = 'any', inplace: bool = False) -> Dict:
        """Drop rows with missing values"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        df = self.dataframes[name]
        original_len = len(df)
        
        if inplace:
            df.dropna(subset=subset, how=how, inplace=True)
            new_len = len(df)
        else:
            new_df = df.dropna(subset=subset, how=how)
            new_len = len(new_df)
        
        return {
            'status': 'success',
            'original_rows': original_len,
            'new_rows': new_len,
            'rows_removed': original_len - new_len,
            'inplace': inplace
        }
    
    def fillna(self, name: str, value: Any = None, method: Optional[str] = None,
               columns: Optional[List[str]] = None, inplace: bool = False) -> Dict:
        """Fill missing values"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        try:
            df = self.dataframes[name]
            
            if columns:
                if inplace:
                    df[columns].fillna(value=value, method=method, inplace=True)
                else:
                    new_df = df.copy()
                    new_df[columns].fillna(value=value, method=method, inplace=True)
            else:
                if inplace:
                    df.fillna(value=value, method=method, inplace=True)
                else:
                    new_df = df.fillna(value=value, method=method)
            
            return {
                'status': 'success',
                'method': method if method else 'value',
                'fill_value': value,
                'columns': columns if columns else 'all',
                'inplace': inplace
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    # ==================== TREND ANALYSIS ====================
    
    def find_trends(self, name: str, date_column: str, value_column: str,
                    period: str = 'M') -> Dict:
        """Analyze trends over time"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        try:
            df = self.dataframes[name].copy()
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Group by period
            trend = df.groupby(pd.Grouper(key=date_column, freq=period))[value_column].agg([
                'mean', 'median', 'min', 'max', 'count'
            ])
            
            return {
                'status': 'success',
                'date_column': date_column,
                'value_column': value_column,
                'period': period,
                'trend_data': trend.to_dict('index'),
                'overall_trend': 'increasing' if trend['mean'].is_monotonic_increasing else 
                                'decreasing' if trend['mean'].is_monotonic_decreasing else 'mixed'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def find_outliers(self, name: str, column: str, method: str = 'iqr') -> Dict:
        """Detect outliers in a column"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        try:
            df = self.dataframes[name]
            
            if column not in df.columns:
                return {'status': 'error', 'message': f'Column {column} not found'}
            
            data = df[column].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                
            elif method == 'zscore':
                mean = data.mean()
                std = data.std()
                z_scores = np.abs((data - mean) / std)
                outliers = df[z_scores > 3]
            
            else:
                return {'status': 'error', 'message': f'Invalid method: {method}'}
            
            return {
                'status': 'success',
                'column': column,
                'method': method,
                'total_outliers': len(outliers),
                'outlier_percentage': len(outliers) / len(df) * 100,
                'outlier_samples': outliers.head(10).to_dict('records')
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    # ==================== ADVANCED ANALYSIS ====================
    
    def top_performers(self, name: str, metric_column: str, group_by: Optional[str] = None,
                       top_n: int = 10, ascending: bool = False) -> Dict:
        """Find top performers"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        try:
            df = self.dataframes[name]
            
            if group_by:
                result = df.groupby(group_by)[metric_column].mean().sort_values(
                    ascending=ascending
                ).head(top_n)
            else:
                result = df.nlargest(top_n, metric_column) if not ascending else \
                         df.nsmallest(top_n, metric_column)
            
            return {
                'status': 'success',
                'metric': metric_column,
                'top_n': top_n,
                'direction': 'ascending' if ascending else 'descending',
                'results': result.to_dict() if isinstance(result, pd.Series) else \
                          result.to_dict('records')
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def compare_periods(self, name: str, date_column: str, metric_column: str,
                        period1_start: str, period1_end: str,
                        period2_start: str, period2_end: str) -> Dict:
        """Compare two time periods"""
        if name not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {name} not found'}
        
        try:
            df = self.dataframes[name].copy()
            df[date_column] = pd.to_datetime(df[date_column])
            
            period1 = df[(df[date_column] >= period1_start) & (df[date_column] <= period1_end)]
            period2 = df[(df[date_column] >= period2_start) & (df[date_column] <= period2_end)]
            
            p1_mean = period1[metric_column].mean()
            p2_mean = period2[metric_column].mean()
            
            change = ((p2_mean - p1_mean) / p1_mean * 100) if p1_mean != 0 else 0
            
            return {
                'status': 'success',
                'period1': {'start': period1_start, 'end': period1_end, 'mean': float(p1_mean)},
                'period2': {'start': period2_start, 'end': period2_end, 'mean': float(p2_mean)},
                'change_percentage': float(change),
                'trend': 'increasing' if change > 0 else 'decreasing' if change < 0 else 'stable'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    # ==================== UTILITY ====================
    
    def merge_dataframes(self, left: str, right: str, on: Union[str, List[str]],
                         how: str = 'inner', result_name: str = 'merged') -> Dict:
        """Merge two dataframes"""
        if left not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {left} not found'}
        if right not in self.dataframes:
            return {'status': 'error', 'message': f'DataFrame {right} not found'}
        
        try:
            df_left = self.dataframes[left]
            df_right = self.dataframes[right]
            
            merged = pd.merge(df_left, df_right, on=on, how=how)
            self.dataframes[result_name] = merged
            
            return {
                'status': 'success',
                'left_shape': df_left.shape,
                'right_shape': df_right.shape,
                'result_shape': merged.shape,
                'merge_keys': on if isinstance(on, list) else [on],
                'merge_type': how,
                'result_name': result_name
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_history(self) -> List[Dict]:
        """Get operation history"""
        return self.history
    
    def list_dataframes(self) -> Dict:
        """List all loaded dataframes"""
        return {
            'status': 'success',
            'dataframes': {
                name: {
                    'shape': df.shape,
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
                }
                for name, df in self.dataframes.items()
            }
        }


# MCP Server Interface (for MCP protocol)
def handle_mcp_request(request: Dict) -> Dict:
    """
    Handle MCP protocol requests
    """
    server = PandasMCPServer()
    
    method = request.get('method')
    params = request.get('params', {})
    
    # Map method to function
    method_map = {
        'read_csv': server.read_csv,
        'read_excel': server.read_excel,
        'info': server.info,
        'describe': server.describe,
        'head': server.head,
        'tail': server.tail,
        'value_counts': server.value_counts,
        'correlation': server.correlation,
        'groupby': server.groupby,
        'pivot_table': server.pivot_table,
        'query': server.query,
        'filter_by_value': server.filter_by_value,
        'drop_duplicates': server.drop_duplicates,
        'drop_na': server.drop_na,
        'fillna': server.fillna,
        'find_trends': server.find_trends,
        'find_outliers': server.find_outliers,
        'top_performers': server.top_performers,
        'compare_periods': server.compare_periods,
        'merge_dataframes': server.merge_dataframes,
        'list_dataframes': server.list_dataframes,
        'get_history': server.get_history
    }
    
    if method in method_map:
        return method_map[method](**params)
    else:
        return {'status': 'error', 'message': f'Unknown method: {method}'}


if __name__ == '__main__':
    # Test the server
    server = PandasMCPServer()
    
    # Example usage
    result = server.read_csv('/project/workspace/spotify_data-clean.csv', 'spotify')
    print(json.dumps(result, indent=2))
    
    info = server.info('spotify')
    print(json.dumps(info, indent=2))