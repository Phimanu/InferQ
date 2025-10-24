"""
Utility functions for quality metric computation.

This module provides helper functions for common operations
used in quality metric calculations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from scipy import stats


def detect_outliers_iqr(data: pd.Series, 
                        threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers using the IQR method.
    
    Args:
        data: Series of numeric values
        threshold: IQR multiplier (typically 1.5 or 3.0)
        
    Returns:
        Boolean series indicating outliers
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data: pd.Series, 
                           threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using the Z-score method.
    
    Args:
        data: Series of numeric values
        threshold: Z-score threshold
        
    Returns:
        Boolean series indicating outliers
    """
    z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
    return z_scores > threshold


def detect_outliers_modified_zscore(data: pd.Series, 
                                    threshold: float = 3.5) -> pd.Series:
    """
    Detect outliers using the modified Z-score (based on MAD).
    
    More robust to extreme outliers than standard Z-score.
    
    Args:
        data: Series of numeric values
        threshold: Modified Z-score threshold
        
    Returns:
        Boolean series indicating outliers
    """
    median = data.median()
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        return pd.Series(False, index=data.index)
    
    modified_z_scores = 0.6745 * (data - median) / mad
    return np.abs(modified_z_scores) > threshold


def validate_range_constraint(data: pd.Series,
                              min_val: Optional[float] = None,
                              max_val: Optional[float] = None) -> pd.Series:
    """
    Validate range constraints on data.
    
    Args:
        data: Series of numeric values
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Returns:
        Boolean series indicating valid values
    """
    valid = pd.Series(True, index=data.index)
    
    if min_val is not None:
        valid &= data >= min_val
    
    if max_val is not None:
        valid &= data <= max_val
    
    return valid


def validate_enum_constraint(data: pd.Series,
                             allowed_values: List[Any]) -> pd.Series:
    """
    Validate enum constraints on data.
    
    Args:
        data: Series of values
        allowed_values: List of allowed values
        
    Returns:
        Boolean series indicating valid values
    """
    return data.isin(allowed_values)


def compute_missing_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missing value patterns across the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        DataFrame with missing patterns and their frequencies
    """
    missing_mask = df.isna()
    pattern_counts = missing_mask.value_counts()
    
    result = pd.DataFrame({
        'pattern': [tuple(pattern) for pattern in pattern_counts.index],
        'count': pattern_counts.values,
        'percentage': pattern_counts.values / len(df) * 100
    })
    
    return result.sort_values('count', ascending=False)


def compute_correlation_matrix(df: pd.DataFrame, 
                               method: str = 'pearson') -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.
    
    Args:
        df: DataFrame to analyze
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Correlation matrix
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return pd.DataFrame()
    
    return numeric_df.corr(method=method)


def identify_constant_columns(df: pd.DataFrame,
                             threshold: float = 0.95) -> List[str]:
    """
    Identify columns with constant or near-constant values.
    
    Args:
        df: DataFrame to analyze
        threshold: Ratio threshold for considering a column constant
                  (e.g., 0.95 means 95% of values are the same)
        
    Returns:
        List of column names with constant values
    """
    constant_columns = []
    
    for column in df.columns:
        value_counts = df[column].value_counts(normalize=True)
        
        if len(value_counts) > 0 and value_counts.iloc[0] >= threshold:
            constant_columns.append(column)
    
    return constant_columns


def identify_high_cardinality_columns(df: pd.DataFrame,
                                     threshold: float = 0.9) -> List[str]:
    """
    Identify columns with very high cardinality.
    
    High cardinality columns may indicate potential quality issues
    or improper data modeling.
    
    Args:
        df: DataFrame to analyze
        threshold: Ratio threshold (unique values / total values)
        
    Returns:
        List of column names with high cardinality
    """
    high_cardinality_columns = []
    
    for column in df.columns:
        nunique = df[column].nunique()
        total = len(df[column].dropna())
        
        if total > 0 and (nunique / total) >= threshold:
            high_cardinality_columns.append(column)
    
    return high_cardinality_columns


def compute_value_frequency(data: pd.Series, 
                           top_n: int = 10) -> pd.DataFrame:
    """
    Compute value frequency distribution.
    
    Args:
        data: Series to analyze
        top_n: Number of top values to return
        
    Returns:
        DataFrame with value, count, and percentage
    """
    value_counts = data.value_counts()
    
    result = pd.DataFrame({
        'value': value_counts.index[:top_n],
        'count': value_counts.values[:top_n],
        'percentage': value_counts.values[:top_n] / len(data) * 100
    })
    
    return result


def normalize_numeric_data(data: pd.Series,
                          method: str = 'minmax') -> pd.Series:
    """
    Normalize numeric data using various methods.
    
    Args:
        data: Series of numeric values
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized series
    """
    if method == 'minmax':
        min_val = data.min()
        max_val = data.max()
        if max_val - min_val == 0:
            return pd.Series(0.5, index=data.index)
        return (data - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean = data.mean()
        std = data.std()
        if std == 0:
            return pd.Series(0.0, index=data.index)
        return (data - mean) / std
    
    elif method == 'robust':
        median = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(0.0, index=data.index)
        return (data - median) / iqr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_data_profile(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute a comprehensive data profile for each column.
    
    Args:
        df: DataFrame to profile
        
    Returns:
        Dictionary mapping column names to their profile statistics
    """
    profile = {}
    
    for column in df.columns:
        col_data = df[column]
        
        col_profile = {
            'dtype': str(col_data.dtype),
            'count': len(col_data),
            'missing': col_data.isna().sum(),
            'missing_pct': col_data.isna().sum() / len(col_data) * 100,
            'unique': col_data.nunique(),
            'unique_pct': col_data.nunique() / len(col_data) * 100 if len(col_data) > 0 else 0,
        }
        
        # Add numeric statistics
        if pd.api.types.is_numeric_dtype(col_data):
            col_profile.update({
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'skew': col_data.skew(),
                'kurtosis': col_data.kurtosis(),
            })
        
        # Add categorical statistics
        else:
            value_counts = col_data.value_counts()
            if len(value_counts) > 0:
                col_profile.update({
                    'most_common': value_counts.index[0],
                    'most_common_freq': value_counts.iloc[0],
                    'most_common_pct': value_counts.iloc[0] / len(col_data) * 100,
                })
        
        profile[column] = col_profile
    
    return profile


def detect_data_drift(reference_data: pd.DataFrame,
                     current_data: pd.DataFrame,
                     method: str = 'ks',
                     alpha: float = 0.05) -> Dict[str, Tuple[float, bool]]:
    """
    Detect data drift between reference and current datasets.
    
    Args:
        reference_data: Reference DataFrame (historical data)
        current_data: Current DataFrame (new data)
        method: Statistical test method ('ks' for Kolmogorov-Smirnov, 
                'chi2' for Chi-squared)
        alpha: Significance level for hypothesis testing
        
    Returns:
        Dictionary mapping column names to (p_value, has_drift) tuples
    """
    drift_results = {}
    
    common_columns = reference_data.columns.intersection(current_data.columns)
    
    for column in common_columns:
        ref_col = reference_data[column].dropna()
        cur_col = current_data[column].dropna()
        
        if len(ref_col) == 0 or len(cur_col) == 0:
            continue
        
        # For numeric columns, use KS test
        if pd.api.types.is_numeric_dtype(ref_col) and method == 'ks':
            statistic, p_value = stats.ks_2samp(ref_col, cur_col)
            has_drift = p_value < alpha
            drift_results[column] = (p_value, has_drift)
        
        # For categorical columns, use Chi-squared test
        elif method == 'chi2':
            # Get value counts for both datasets
            ref_counts = ref_col.value_counts()
            cur_counts = cur_col.value_counts()
            
            # Align the categories
            all_categories = ref_counts.index.union(cur_counts.index)
            ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
            cur_aligned = cur_counts.reindex(all_categories, fill_value=0)
            
            # Perform chi-squared test
            try:
                statistic, p_value = stats.chisquare(cur_aligned, ref_aligned)
                has_drift = p_value < alpha
                drift_results[column] = (p_value, has_drift)
            except Exception:
                continue
    
    return drift_results


def find_correlated_missing_values(df: pd.DataFrame,
                                   threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """
    Find pairs of columns with correlated missing values.
    
    This can indicate systematic data collection issues.
    
    Args:
        df: DataFrame to analyze
        threshold: Correlation threshold for reporting
        
    Returns:
        List of (col1, col2, correlation) tuples
    """
    missing_mask = df.isna().astype(int)
    corr_matrix = missing_mask.corr()
    
    correlated_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) >= threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                correlated_pairs.append((col1, col2, corr_value))
    
    return sorted(correlated_pairs, key=lambda x: abs(x[2]), reverse=True)


def validate_dataframe(df: pd.DataFrame,
                      schema: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate DataFrame against a schema definition.
    
    Args:
        df: DataFrame to validate
        schema: Schema dictionary with column specifications
                Example: {
                    'age': {'type': 'int', 'min': 0, 'max': 120},
                    'name': {'type': 'str', 'required': True}
                }
        
    Returns:
        Dictionary with validation errors by category
    """
    errors = {
        'missing_columns': [],
        'type_errors': [],
        'range_errors': [],
        'missing_required': []
    }
    
    # Check for missing columns
    expected_columns = set(schema.keys())
    actual_columns = set(df.columns)
    missing = expected_columns - actual_columns
    errors['missing_columns'] = list(missing)
    
    # Validate each column
    for col, spec in schema.items():
        if col not in df.columns:
            continue
        
        col_data = df[col]
        
        # Check required columns
        if spec.get('required', False):
            if col_data.isna().any():
                errors['missing_required'].append(col)
        
        # Check types
        expected_type = spec.get('type')
        if expected_type == 'int':
            if not pd.api.types.is_integer_dtype(col_data):
                errors['type_errors'].append(f"{col}: expected int, got {col_data.dtype}")
        elif expected_type == 'float':
            if not pd.api.types.is_float_dtype(col_data):
                errors['type_errors'].append(f"{col}: expected float, got {col_data.dtype}")
        elif expected_type == 'str':
            if not pd.api.types.is_object_dtype(col_data):
                errors['type_errors'].append(f"{col}: expected str, got {col_data.dtype}")
        
        # Check range constraints
        if 'min' in spec:
            if (col_data < spec['min']).any():
                errors['range_errors'].append(f"{col}: values below minimum {spec['min']}")
        
        if 'max' in spec:
            if (col_data > spec['max']).any():
                errors['range_errors'].append(f"{col}: values above maximum {spec['max']}")
    
    return errors
