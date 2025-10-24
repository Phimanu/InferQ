"""
Quality Metrics Module

This module provides a comprehensive library of quality metric functions
for data quality assessment. Each metric function accepts a data subset
(pandas DataFrame) and returns a scalar quality score.

WP 1: Foundation - Quality Metric Framework
"""

import numpy as np
import pandas as pd
from typing import Union, Callable, Dict, List, Optional, Any
from scipy import stats


# ==============================================================================
# COMPLETENESS METRICS
# ==============================================================================

def compute_completeness(data_subset: pd.DataFrame) -> float:
    """
    Compute the completeness metric for a data subset.
    
    Completeness measures the ratio of non-missing values to total values.
    
    Args:
        data_subset: pandas DataFrame slice
        
    Returns:
        float: Completeness score in [0, 1], where 1 means no missing values
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, None], 'b': [4, 5, 6]})
        >>> compute_completeness(df)
        0.8333333333333334
    """
    if data_subset.empty:
        return 1.0
    
    total_cells = data_subset.size
    non_null_cells = data_subset.notna().sum().sum()
    
    return non_null_cells / total_cells if total_cells > 0 else 1.0


def compute_column_completeness(data_subset: pd.DataFrame, column: str) -> float:
    """
    Compute completeness for a specific column.
    
    Args:
        data_subset: pandas DataFrame slice
        column: Name of the column to check
        
    Returns:
        float: Completeness score for the column in [0, 1]
    """
    if data_subset.empty or column not in data_subset.columns:
        return 1.0
    
    total_rows = len(data_subset)
    non_null_rows = data_subset[column].notna().sum()
    
    return non_null_rows / total_rows if total_rows > 0 else 1.0


# ==============================================================================
# OUTLIER DETECTION METRICS
# ==============================================================================

def compute_outlier_rate(data_subset: pd.DataFrame, 
                         method: str = 'iqr',
                         threshold: float = 1.5) -> float:
    """
    Compute the outlier rate using various detection methods.
    
    Args:
        data_subset: pandas DataFrame slice
        method: Detection method - 'iqr', 'zscore', or 'modified_zscore'
        threshold: Threshold for outlier detection
                  - IQR: typically 1.5 or 3.0
                  - Z-score: typically 3.0
                  
    Returns:
        float: Outlier rate in [0, 1], where 0 means no outliers
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, 3, 100], 'b': [4, 5, 6, 7]})
        >>> compute_outlier_rate(df, method='iqr')
        0.125
    """
    if data_subset.empty:
        return 0.0
    
    # Select only numeric columns
    numeric_data = data_subset.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return 0.0
    
    total_values = numeric_data.size
    outlier_count = 0
    
    for column in numeric_data.columns:
        col_data = numeric_data[column].dropna()
        
        if len(col_data) < 4:  # Need at least 4 points for meaningful outlier detection
            continue
        
        if method == 'iqr':
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(col_data))
            outliers = (z_scores > threshold).sum()
            
        elif method == 'modified_zscore':
            median = np.median(col_data)
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * (col_data - median) / mad if mad != 0 else np.zeros_like(col_data)
            outliers = (np.abs(modified_z_scores) > threshold).sum()
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        outlier_count += outliers
    
    return outlier_count / total_values if total_values > 0 else 0.0


# ==============================================================================
# DUPLICATE DETECTION METRICS
# ==============================================================================

def compute_duplicate_rate(data_subset: pd.DataFrame, 
                           subset: Optional[List[str]] = None) -> float:
    """
    Compute the rate of duplicate rows in the data subset.
    
    Args:
        data_subset: pandas DataFrame slice
        subset: List of columns to consider for duplicate detection.
                If None, all columns are used.
                
    Returns:
        float: Duplicate rate in [0, 1], where 0 means no duplicates
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, 2], 'b': [4, 5, 5]})
        >>> compute_duplicate_rate(df)
        0.3333333333333333
    """
    if data_subset.empty or len(data_subset) == 1:
        return 0.0
    
    total_rows = len(data_subset)
    duplicate_rows = data_subset.duplicated(subset=subset, keep='first').sum()
    
    return duplicate_rows / total_rows if total_rows > 0 else 0.0


def compute_key_uniqueness(data_subset: pd.DataFrame, 
                           key_columns: List[str]) -> float:
    """
    Compute uniqueness ratio for specified key columns.
    
    Args:
        data_subset: pandas DataFrame slice
        key_columns: List of columns that should form a unique key
        
    Returns:
        float: Uniqueness score in [0, 1], where 1 means all keys are unique
    """
    if data_subset.empty or not key_columns:
        return 1.0
    
    # Check if all key columns exist
    missing_cols = [col for col in key_columns if col not in data_subset.columns]
    if missing_cols:
        return 0.0
    
    total_rows = len(data_subset)
    unique_rows = data_subset[key_columns].drop_duplicates().shape[0]
    
    return unique_rows / total_rows if total_rows > 0 else 1.0


# ==============================================================================
# CONSISTENCY METRICS
# ==============================================================================

def compute_format_consistency(data_subset: pd.DataFrame, 
                               column: str,
                               pattern: Optional[str] = None) -> float:
    """
    Compute format consistency for a column.
    
    Args:
        data_subset: pandas DataFrame slice
        column: Column name to check
        pattern: Regex pattern to match (if None, checks for format variety)
        
    Returns:
        float: Consistency score in [0, 1], where 1 means all values match
    """
    if data_subset.empty or column not in data_subset.columns:
        return 1.0
    
    col_data = data_subset[column].dropna()
    
    if len(col_data) == 0:
        return 1.0
    
    if pattern is not None:
        # Check against specific pattern
        import re
        matches = col_data.astype(str).str.match(pattern).sum()
        return matches / len(col_data)
    else:
        # Check format consistency (e.g., date formats, phone formats)
        # For strings, check if they have similar structure
        if col_data.dtype == object:
            # Simple heuristic: check length consistency
            str_lengths = col_data.astype(str).str.len()
            length_variance = str_lengths.var()
            # Normalize: lower variance = higher consistency
            # Use coefficient of variation
            mean_length = str_lengths.mean()
            if mean_length > 0:
                cv = np.sqrt(length_variance) / mean_length
                consistency = 1.0 / (1.0 + cv)
                return consistency
            return 1.0
        else:
            return 1.0


def compute_referential_integrity(data_subset: pd.DataFrame,
                                  foreign_key: str,
                                  reference_values: set) -> float:
    """
    Compute referential integrity for foreign key relationships.
    
    Args:
        data_subset: pandas DataFrame slice
        foreign_key: Column name of the foreign key
        reference_values: Set of valid reference values
        
    Returns:
        float: Integrity score in [0, 1], where 1 means all references are valid
    """
    if data_subset.empty or foreign_key not in data_subset.columns:
        return 1.0
    
    fk_values = data_subset[foreign_key].dropna()
    
    if len(fk_values) == 0:
        return 1.0
    
    valid_references = fk_values.isin(reference_values).sum()
    
    return valid_references / len(fk_values)


# ==============================================================================
# CONSTRAINT VIOLATION METRICS
# ==============================================================================

def compute_constraint_violation(data_subset: pd.DataFrame,
                                constraints: List[Dict[str, Any]]) -> float:
    """
    Compute constraint violation rate for a set of data constraints.
    
    Args:
        data_subset: pandas DataFrame slice
        constraints: List of constraint dictionaries, each with:
                    - 'type': 'range', 'enum', 'custom'
                    - 'column': column name
                    - 'min'/'max': for range constraints
                    - 'values': for enum constraints
                    - 'func': for custom validation function
                    
    Returns:
        float: Violation rate in [0, 1], where 0 means no violations
        
    Examples:
        >>> df = pd.DataFrame({'age': [25, 30, -5, 150]})
        >>> constraints = [{'type': 'range', 'column': 'age', 'min': 0, 'max': 120}]
        >>> compute_constraint_violation(df, constraints)
        0.5
    """
    if data_subset.empty or not constraints:
        return 0.0
    
    total_checks = 0
    violations = 0
    
    for constraint in constraints:
        constraint_type = constraint.get('type')
        column = constraint.get('column')
        
        if column not in data_subset.columns:
            continue
        
        col_data = data_subset[column].dropna()
        
        if len(col_data) == 0:
            continue
        
        if constraint_type == 'range':
            min_val = constraint.get('min', -np.inf)
            max_val = constraint.get('max', np.inf)
            violations += ((col_data < min_val) | (col_data > max_val)).sum()
            total_checks += len(col_data)
            
        elif constraint_type == 'enum':
            valid_values = set(constraint.get('values', []))
            violations += (~col_data.isin(valid_values)).sum()
            total_checks += len(col_data)
            
        elif constraint_type == 'custom':
            func = constraint.get('func')
            if func is not None and callable(func):
                # Custom function should return boolean array
                try:
                    invalid_mask = ~col_data.apply(func)
                    violations += invalid_mask.sum()
                    total_checks += len(col_data)
                except Exception:
                    pass
    
    return violations / total_checks if total_checks > 0 else 0.0


def compute_type_validity(data_subset: pd.DataFrame, 
                         expected_types: Dict[str, type]) -> float:
    """
    Compute type validity score based on expected data types.
    
    Args:
        data_subset: pandas DataFrame slice
        expected_types: Dictionary mapping column names to expected Python types
        
    Returns:
        float: Validity score in [0, 1], where 1 means all types are correct
    """
    if data_subset.empty or not expected_types:
        return 1.0
    
    total_checks = 0
    valid_types = 0
    
    for column, expected_type in expected_types.items():
        if column not in data_subset.columns:
            continue
        
        col_data = data_subset[column].dropna()
        
        if len(col_data) == 0:
            continue
        
        # Check if values can be cast to expected type
        try:
            if expected_type in (int, float):
                valid = pd.to_numeric(col_data, errors='coerce').notna().sum()
            elif expected_type == str:
                valid = len(col_data)  # Anything can be a string
            elif expected_type == bool:
                valid = col_data.isin([True, False, 0, 1]).sum()
            else:
                # Try direct isinstance check
                valid = sum(isinstance(val, expected_type) for val in col_data)
            
            valid_types += valid
            total_checks += len(col_data)
        except Exception:
            total_checks += len(col_data)
    
    return valid_types / total_checks if total_checks > 0 else 1.0


# ==============================================================================
# DISTRIBUTION METRICS
# ==============================================================================

def compute_distribution_skewness(data_subset: pd.DataFrame) -> float:
    """
    Compute average skewness across numeric columns.
    
    High skewness may indicate data quality issues or biased sampling.
    
    Args:
        data_subset: pandas DataFrame slice
        
    Returns:
        float: Average absolute skewness across numeric columns
    """
    if data_subset.empty:
        return 0.0
    
    numeric_data = data_subset.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return 0.0
    
    skewness_values = []
    
    for column in numeric_data.columns:
        col_data = numeric_data[column].dropna()
        
        if len(col_data) >= 3:  # Need at least 3 points for skewness
            skew = stats.skew(col_data)
            if not np.isnan(skew):
                skewness_values.append(abs(skew))
    
    return np.mean(skewness_values) if skewness_values else 0.0


def compute_distribution_kurtosis(data_subset: pd.DataFrame) -> float:
    """
    Compute average kurtosis across numeric columns.
    
    High kurtosis may indicate outliers or heavy tails in the distribution.
    
    Args:
        data_subset: pandas DataFrame slice
        
    Returns:
        float: Average absolute excess kurtosis across numeric columns
    """
    if data_subset.empty:
        return 0.0
    
    numeric_data = data_subset.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        return 0.0
    
    kurtosis_values = []
    
    for column in numeric_data.columns:
        col_data = numeric_data[column].dropna()
        
        if len(col_data) >= 4:  # Need at least 4 points for kurtosis
            kurt = stats.kurtosis(col_data)
            if not np.isnan(kurt):
                kurtosis_values.append(abs(kurt))
    
    return np.mean(kurtosis_values) if kurtosis_values else 0.0


# ==============================================================================
# TIMELINESS METRICS
# ==============================================================================

def compute_freshness(data_subset: pd.DataFrame,
                     timestamp_column: str,
                     current_time: Optional[pd.Timestamp] = None) -> float:
    """
    Compute data freshness based on timestamps.
    
    Args:
        data_subset: pandas DataFrame slice
        timestamp_column: Name of the timestamp column
        current_time: Reference time (default: now)
        
    Returns:
        float: Freshness score in [0, 1], where 1 means very fresh data
    """
    if data_subset.empty or timestamp_column not in data_subset.columns:
        return 0.0
    
    if current_time is None:
        current_time = pd.Timestamp.now()
    
    timestamps = pd.to_datetime(data_subset[timestamp_column], errors='coerce').dropna()
    
    if len(timestamps) == 0:
        return 0.0
    
    # Calculate average age in days
    ages = (current_time - timestamps).dt.total_seconds() / (24 * 3600)
    avg_age = ages.mean()
    
    # Normalize using exponential decay: freshness = exp(-age/30)
    # Data older than 30 days has low freshness
    freshness = np.exp(-avg_age / 30.0)
    
    return freshness


# ==============================================================================
# ACCURACY METRICS
# ==============================================================================

def compute_value_accuracy(data_subset: pd.DataFrame,
                          column: str,
                          ground_truth: pd.Series) -> float:
    """
    Compute accuracy by comparing with ground truth values.
    
    Args:
        data_subset: pandas DataFrame slice
        column: Column to check
        ground_truth: Series with correct values (same index as data_subset)
        
    Returns:
        float: Accuracy score in [0, 1], where 1 means perfect accuracy
    """
    if data_subset.empty or column not in data_subset.columns:
        return 0.0
    
    # Align indices
    common_idx = data_subset.index.intersection(ground_truth.index)
    
    if len(common_idx) == 0:
        return 0.0
    
    actual = data_subset.loc[common_idx, column]
    expected = ground_truth.loc[common_idx]
    
    # For numeric data, use relative error
    if pd.api.types.is_numeric_dtype(actual) and pd.api.types.is_numeric_dtype(expected):
        valid_mask = actual.notna() & expected.notna() & (expected != 0)
        if valid_mask.sum() == 0:
            return 1.0
        
        relative_errors = np.abs((actual[valid_mask] - expected[valid_mask]) / expected[valid_mask])
        # Convert to accuracy: 1 - mean relative error (capped at 0)
        accuracy = max(0.0, 1.0 - relative_errors.mean())
        return accuracy
    else:
        # For categorical data, use exact match
        matches = (actual == expected).sum()
        return matches / len(common_idx)


# ==============================================================================
# COMPOSITE METRICS
# ==============================================================================

def compute_overall_quality(data_subset: pd.DataFrame,
                           metric_weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute overall quality score as weighted combination of multiple metrics.
    
    Args:
        data_subset: pandas DataFrame slice
        metric_weights: Dictionary mapping metric names to weights.
                       If None, uses equal weights for completeness and outlier rate.
                       
    Returns:
        float: Overall quality score in [0, 1]
    """
    if metric_weights is None:
        metric_weights = {
            'completeness': 0.5,
            'outlier_rate': 0.5
        }
    
    scores = {}
    
    # Compute available metrics
    if 'completeness' in metric_weights:
        scores['completeness'] = compute_completeness(data_subset)
    
    if 'outlier_rate' in metric_weights:
        outlier_rate = compute_outlier_rate(data_subset)
        scores['outlier_rate'] = 1.0 - outlier_rate  # Invert so higher is better
    
    if 'duplicate_rate' in metric_weights:
        duplicate_rate = compute_duplicate_rate(data_subset)
        scores['duplicate_rate'] = 1.0 - duplicate_rate
    
    # Compute weighted average
    total_weight = sum(metric_weights.get(k, 0) for k in scores.keys())
    
    if total_weight == 0:
        return 1.0
    
    weighted_sum = sum(scores[k] * metric_weights.get(k, 0) for k in scores.keys())
    
    return weighted_sum / total_weight


# ==============================================================================
# QUALITY METRIC REGISTRY
# ==============================================================================

class QualityMetric:
    """
    Wrapper class for quality metric functions with metadata.
    """
    
    def __init__(self, 
                 name: str,
                 func: Callable[[pd.DataFrame], float],
                 description: str = "",
                 category: str = "general",
                 higher_is_better: bool = True,
                 requires_config: bool = False):
        """
        Initialize a quality metric.
        
        Args:
            name: Unique identifier for the metric
            func: Function that computes the metric
            description: Human-readable description
            category: Category (e.g., 'completeness', 'outliers', 'consistency')
            higher_is_better: Whether higher values indicate better quality
            requires_config: Whether the metric requires additional configuration
        """
        self.name = name
        self.func = func
        self.description = description
        self.category = category
        self.higher_is_better = higher_is_better
        self.requires_config = requires_config
    
    def compute(self, data_subset: pd.DataFrame, **kwargs) -> float:
        """
        Compute the metric on a data subset.
        
        Args:
            data_subset: pandas DataFrame slice
            **kwargs: Additional arguments for the metric function
            
        Returns:
            float: Quality score
        """
        try:
            if self.requires_config:
                return self.func(data_subset, **kwargs)
            else:
                return self.func(data_subset)
        except Exception as e:
            print(f"Warning: Failed to compute metric '{self.name}': {e}")
            return 0.0 if self.higher_is_better else 1.0
    
    def __repr__(self):
        return f"QualityMetric(name='{self.name}', category='{self.category}')"


class QualityMetricRegistry:
    """
    Central registry for managing quality metrics.
    
    This class provides a unified interface for registering, accessing,
    and computing quality metrics across the system.
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._metrics: Dict[str, QualityMetric] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(self, metric: QualityMetric) -> None:
        """
        Register a quality metric.
        
        Args:
            metric: QualityMetric instance to register
            
        Raises:
            ValueError: If a metric with the same name already exists
        """
        if metric.name in self._metrics:
            raise ValueError(f"Metric '{metric.name}' is already registered")
        
        self._metrics[metric.name] = metric
        
        # Update category index
        if metric.category not in self._categories:
            self._categories[metric.category] = []
        self._categories[metric.category].append(metric.name)
    
    def register_function(self,
                         name: str,
                         func: Callable[[pd.DataFrame], float],
                         description: str = "",
                         category: str = "general",
                         higher_is_better: bool = True,
                         requires_config: bool = False) -> None:
        """
        Register a metric function directly.
        
        Args:
            name: Unique identifier for the metric
            func: Function that computes the metric
            description: Human-readable description
            category: Category of the metric
            higher_is_better: Whether higher values indicate better quality
            requires_config: Whether the metric requires additional configuration
        """
        metric = QualityMetric(
            name=name,
            func=func,
            description=description,
            category=category,
            higher_is_better=higher_is_better,
            requires_config=requires_config
        )
        self.register(metric)
    
    def get(self, name: str) -> Optional[QualityMetric]:
        """
        Get a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            QualityMetric instance or None if not found
        """
        return self._metrics.get(name)
    
    def get_by_category(self, category: str) -> List[QualityMetric]:
        """
        Get all metrics in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of QualityMetric instances
        """
        metric_names = self._categories.get(category, [])
        return [self._metrics[name] for name in metric_names]
    
    def list_metrics(self) -> List[str]:
        """
        Get list of all registered metric names.
        
        Returns:
            List of metric names
        """
        return list(self._metrics.keys())
    
    def list_categories(self) -> List[str]:
        """
        Get list of all categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def compute(self, 
                name: str, 
                data_subset: pd.DataFrame,
                **kwargs) -> float:
        """
        Compute a metric by name.
        
        Args:
            name: Metric name
            data_subset: pandas DataFrame slice
            **kwargs: Additional arguments for the metric
            
        Returns:
            float: Quality score
            
        Raises:
            ValueError: If metric not found
        """
        metric = self.get(name)
        if metric is None:
            raise ValueError(f"Metric '{name}' not found in registry")
        
        return metric.compute(data_subset, **kwargs)
    
    def compute_all(self,
                   data_subset: pd.DataFrame,
                   metric_configs: Optional[Dict[str, Dict]] = None) -> Dict[str, float]:
        """
        Compute all registered metrics that don't require configuration.
        
        Args:
            data_subset: pandas DataFrame slice
            metric_configs: Optional dictionary mapping metric names to their configs
            
        Returns:
            Dictionary mapping metric names to their scores
        """
        results = {}
        metric_configs = metric_configs or {}
        
        for name, metric in self._metrics.items():
            if metric.requires_config and name not in metric_configs:
                continue
            
            config = metric_configs.get(name, {})
            results[name] = metric.compute(data_subset, **config)
        
        return results
    
    def compute_category(self,
                        category: str,
                        data_subset: pd.DataFrame,
                        metric_configs: Optional[Dict[str, Dict]] = None) -> Dict[str, float]:
        """
        Compute all metrics in a category.
        
        Args:
            category: Category name
            data_subset: pandas DataFrame slice
            metric_configs: Optional dictionary mapping metric names to their configs
            
        Returns:
            Dictionary mapping metric names to their scores
        """
        results = {}
        metric_configs = metric_configs or {}
        
        for metric in self.get_by_category(category):
            if metric.requires_config and metric.name not in metric_configs:
                continue
            
            config = metric_configs.get(metric.name, {})
            results[metric.name] = metric.compute(data_subset, **config)
        
        return results
    
    def __len__(self):
        """Return number of registered metrics."""
        return len(self._metrics)
    
    def __contains__(self, name: str):
        """Check if a metric is registered."""
        return name in self._metrics
    
    def __repr__(self):
        return f"QualityMetricRegistry(metrics={len(self._metrics)}, categories={len(self._categories)})"


def get_default_registry() -> QualityMetricRegistry:
    """
    Create and return a registry with all default quality metrics.
    
    Returns:
        QualityMetricRegistry with all built-in metrics registered
    """
    registry = QualityMetricRegistry()
    
    # Completeness metrics
    registry.register_function(
        name="completeness",
        func=compute_completeness,
        description="Ratio of non-missing values to total values",
        category="completeness",
        higher_is_better=True,
        requires_config=False
    )
    
    registry.register_function(
        name="column_completeness",
        func=compute_column_completeness,
        description="Completeness for a specific column",
        category="completeness",
        higher_is_better=True,
        requires_config=True
    )
    
    # Outlier metrics
    registry.register_function(
        name="outlier_rate",
        func=compute_outlier_rate,
        description="Rate of outliers in numeric columns",
        category="outliers",
        higher_is_better=False,
        requires_config=False
    )
    
    # Duplicate metrics
    registry.register_function(
        name="duplicate_rate",
        func=compute_duplicate_rate,
        description="Rate of duplicate rows",
        category="duplicates",
        higher_is_better=False,
        requires_config=False
    )
    
    registry.register_function(
        name="key_uniqueness",
        func=compute_key_uniqueness,
        description="Uniqueness ratio for specified key columns",
        category="duplicates",
        higher_is_better=True,
        requires_config=True
    )
    
    # Consistency metrics
    registry.register_function(
        name="format_consistency",
        func=compute_format_consistency,
        description="Format consistency within a column",
        category="consistency",
        higher_is_better=True,
        requires_config=True
    )
    
    registry.register_function(
        name="referential_integrity",
        func=compute_referential_integrity,
        description="Referential integrity for foreign keys",
        category="consistency",
        higher_is_better=True,
        requires_config=True
    )
    
    # Constraint metrics
    registry.register_function(
        name="constraint_violation",
        func=compute_constraint_violation,
        description="Rate of constraint violations",
        category="constraints",
        higher_is_better=False,
        requires_config=True
    )
    
    registry.register_function(
        name="type_validity",
        func=compute_type_validity,
        description="Type validity based on expected data types",
        category="constraints",
        higher_is_better=True,
        requires_config=True
    )
    
    # Distribution metrics
    registry.register_function(
        name="distribution_skewness",
        func=compute_distribution_skewness,
        description="Average skewness across numeric columns",
        category="distribution",
        higher_is_better=False,
        requires_config=False
    )
    
    registry.register_function(
        name="distribution_kurtosis",
        func=compute_distribution_kurtosis,
        description="Average excess kurtosis across numeric columns",
        category="distribution",
        higher_is_better=False,
        requires_config=False
    )
    
    # Timeliness metrics
    registry.register_function(
        name="freshness",
        func=compute_freshness,
        description="Data freshness based on timestamps",
        category="timeliness",
        higher_is_better=True,
        requires_config=True
    )
    
    # Accuracy metrics
    registry.register_function(
        name="value_accuracy",
        func=compute_value_accuracy,
        description="Accuracy by comparing with ground truth",
        category="accuracy",
        higher_is_better=True,
        requires_config=True
    )
    
    # Composite metrics
    registry.register_function(
        name="overall_quality",
        func=compute_overall_quality,
        description="Overall quality as weighted combination",
        category="composite",
        higher_is_better=True,
        requires_config=False
    )
    
    return registry


# Create a global default registry instance
_default_registry = None


def get_global_registry() -> QualityMetricRegistry:
    """
    Get the global default registry instance (singleton pattern).
    
    Returns:
        Global QualityMetricRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = get_default_registry()
    return _default_registry
