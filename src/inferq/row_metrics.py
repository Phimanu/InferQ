"""
Row-Level Quality Metrics for InferQ

These metrics are meaningful at the individual row level and will have variance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class RowQualityMetric:
    """Base class for row-level quality metrics."""
    name: str
    requires_config: bool = False
    
    def compute(self, row: pd.DataFrame) -> float:
        """Compute metric for a single row."""
        raise NotImplementedError


class RowCompleteness(RowQualityMetric):
    """
    Measures what fraction of values in the row are non-null.
    
    Range: [0, 1]
    - 1.0 = all values present
    - 0.0 = all values missing
    
    This WILL vary across rows in real datasets!
    """
    
    def __init__(self):
        super().__init__("row_completeness", requires_config=False)
    
    def compute(self, row: pd.DataFrame) -> float:
        if len(row) == 0:
            return 0.0
        # Count non-null values
        non_null = row.notna().sum().sum()
        total = row.shape[0] * row.shape[1]
        return float(non_null / total) if total > 0 else 0.0


class RowRangeConformance(RowQualityMetric):
    """
    Measures what fraction of numeric values fall within expected ranges.
    
    Uses dataset statistics to define "expected" ranges:
    - Expected range: [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (standard outlier detection)
    
    Range: [0, 1]
    - 1.0 = all values within expected ranges
    - 0.0 = all values are outliers
    
    Computed per-row based on how many of its values are outliers.
    """
    
    def __init__(self):
        super().__init__("row_range_conformance", requires_config=False)
        self.ranges = {}  # Will be set during training
    
    def fit(self, data: pd.DataFrame):
        """Learn expected ranges from training data."""
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.ranges[col] = (lower, upper)
    
    def compute(self, row: pd.DataFrame) -> float:
        if len(row) == 0 or not self.ranges:
            return 1.0
        
        numeric_cols = row.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 1.0
        
        conforming = 0
        total = 0
        
        for col in numeric_cols:
            if col in self.ranges and pd.notna(row[col].iloc[0]):
                value = row[col].iloc[0]
                lower, upper = self.ranges[col]
                if lower <= value <= upper:
                    conforming += 1
                total += 1
        
        return float(conforming / total) if total > 0 else 1.0


class RowConsistency(RowQualityMetric):
    """
    Measures how consistent the row's values are with learned patterns.
    
    Uses coefficient of variation (CV) as a proxy for consistency:
    - CV = std / mean for the row's numeric values
    - Normalized to [0, 1] where 1.0 = highly consistent
    
    Interpretation:
    - High consistency: Values have similar relative magnitudes
    - Low consistency: Values vary wildly (potential data quality issue)
    
    This captures intra-row patterns that vary across different rows.
    """
    
    def __init__(self):
        super().__init__("row_consistency", requires_config=False)
        self.typical_cv = 1.0  # Will be learned from training data
    
    def fit(self, data: pd.DataFrame):
        """Learn typical coefficient of variation."""
        cvs = []
        for idx in range(min(1000, len(data))):
            row = data.iloc[idx:idx+1]
            numeric_vals = row.select_dtypes(include=[np.number]).values.flatten()
            numeric_vals = numeric_vals[~np.isnan(numeric_vals)]
            if len(numeric_vals) > 1 and np.mean(numeric_vals) != 0:
                cv = np.std(numeric_vals) / abs(np.mean(numeric_vals))
                cvs.append(cv)
        
        if cvs:
            self.typical_cv = np.median(cvs)
    
    def compute(self, row: pd.DataFrame) -> float:
        if len(row) == 0:
            return 0.0
        
        numeric_vals = row.select_dtypes(include=[np.number]).values.flatten()
        numeric_vals = numeric_vals[~np.isnan(numeric_vals)]
        
        if len(numeric_vals) < 2:
            return 1.0  # Can't compute consistency with < 2 values
        
        mean_val = np.mean(numeric_vals)
        if abs(mean_val) < 1e-10:
            return 1.0  # Avoid division by zero
        
        cv = np.std(numeric_vals) / abs(mean_val)
        
        # Normalize: closer to typical_cv = higher consistency
        # Use exponential decay: consistency = exp(-|cv - typical_cv| / typical_cv)
        if self.typical_cv > 0:
            consistency = np.exp(-abs(cv - self.typical_cv) / self.typical_cv)
        else:
            consistency = 1.0
        
        return float(np.clip(consistency, 0.0, 1.0))


def get_row_level_registry():
    """Get registry with row-level metrics."""
    from inferq.quality_metrics import QualityMetricRegistry, QualityMetric
    
    registry = QualityMetricRegistry()
    
    # Wrap our row-level metrics in QualityMetric interface
    completeness_metric = RowCompleteness()
    registry.register(QualityMetric(
        name="row_completeness",
        func=completeness_metric.compute,
        description="Fraction of non-null values in the row",
        category="completeness",
        higher_is_better=True,
        requires_config=False
    ))
    
    range_metric = RowRangeConformance()
    registry.register(QualityMetric(
        name="row_range_conformance",
        func=range_metric.compute,
        description="Fraction of values within expected ranges",
        category="conformance",
        higher_is_better=True,
        requires_config=False
    ))
    
    consistency_metric = RowConsistency()
    registry.register(QualityMetric(
        name="row_consistency",
        func=consistency_metric.compute,
        description="Consistency of value patterns within the row",
        category="consistency",
        higher_is_better=True,
        requires_config=False
    ))
    
    return registry, {
        'row_completeness': completeness_metric,
        'row_range_conformance': range_metric,
        'row_consistency': consistency_metric
    }


def fit_row_metrics(data: pd.DataFrame, metrics_dict: Dict):
    """Fit row-level metrics on training data."""
    if 'row_range_conformance' in metrics_dict:
        metrics_dict['row_range_conformance'].fit(data)
    if 'row_consistency' in metrics_dict:
        metrics_dict['row_consistency'].fit(data)
