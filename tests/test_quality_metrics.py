"""
Unit tests for quality metrics.

Tests all quality metric functions with various scenarios and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from inferq.quality_metrics import (
    compute_completeness,
    compute_column_completeness,
    compute_outlier_rate,
    compute_duplicate_rate,
    compute_key_uniqueness,
    compute_format_consistency,
    compute_referential_integrity,
    compute_constraint_violation,
    compute_type_validity,
    compute_distribution_skewness,
    compute_distribution_kurtosis,
    compute_freshness,
    compute_value_accuracy,
    compute_overall_quality,
    QualityMetric,
    QualityMetricRegistry,
    get_default_registry,
)


class TestCompletenessMetrics:
    """Tests for completeness metrics."""
    
    def test_complete_data(self):
        """Test with completely filled data."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert compute_completeness(df) == 1.0
    
    def test_partial_missing(self):
        """Test with some missing values."""
        df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, 6]})
        assert compute_completeness(df) == pytest.approx(5/6, rel=1e-5)
    
    def test_all_missing(self):
        """Test with all missing values."""
        df = pd.DataFrame({'a': [None, None, None], 'b': [None, None, None]})
        assert compute_completeness(df) == 0.0
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        assert compute_completeness(df) == 1.0
    
    def test_column_completeness(self):
        """Test column-specific completeness."""
        df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, 6]})
        assert compute_column_completeness(df, 'a') == pytest.approx(2/3, rel=1e-5)
        assert compute_column_completeness(df, 'b') == 1.0
    
    def test_column_completeness_missing_column(self):
        """Test column completeness with non-existent column."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        assert compute_column_completeness(df, 'nonexistent') == 1.0


class TestOutlierMetrics:
    """Tests for outlier detection metrics."""
    
    def test_no_outliers(self):
        """Test with no outliers."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        assert compute_outlier_rate(df, method='iqr') == 0.0
    
    def test_with_outliers_iqr(self):
        """Test with clear outliers using IQR method."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 100]})
        rate = compute_outlier_rate(df, method='iqr', threshold=1.5)
        assert rate > 0.0
    
    def test_with_outliers_zscore(self):
        """Test with outliers using Z-score method."""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 100]})
        rate = compute_outlier_rate(df, method='zscore', threshold=2.0)
        assert rate > 0.0
    
    def test_multiple_columns(self):
        """Test outlier detection across multiple columns."""
        df = pd.DataFrame({'a': [1, 2, 3, 100], 'b': [1, 2, 3, 4]})
        rate = compute_outlier_rate(df, method='iqr')
        assert 0.0 < rate < 1.0
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        assert compute_outlier_rate(df) == 0.0
    
    def test_non_numeric_data(self):
        """Test with non-numeric data."""
        df = pd.DataFrame({'a': ['x', 'y', 'z']})
        assert compute_outlier_rate(df) == 0.0


class TestDuplicateMetrics:
    """Tests for duplicate detection metrics."""
    
    def test_no_duplicates(self):
        """Test with no duplicate rows."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert compute_duplicate_rate(df) == 0.0
    
    def test_with_duplicates(self):
        """Test with duplicate rows."""
        df = pd.DataFrame({'a': [1, 2, 2], 'b': [4, 5, 5]})
        assert compute_duplicate_rate(df) == pytest.approx(1/3, rel=1e-5)
    
    def test_all_duplicates(self):
        """Test with all duplicate rows."""
        df = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2]})
        assert compute_duplicate_rate(df) == pytest.approx(2/3, rel=1e-5)
    
    def test_subset_columns(self):
        """Test duplicate detection on subset of columns."""
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [4, 5, 6]})
        assert compute_duplicate_rate(df, subset=['a']) == pytest.approx(1/3, rel=1e-5)
    
    def test_key_uniqueness(self):
        """Test key uniqueness metric."""
        df = pd.DataFrame({'id': [1, 2, 2, 3], 'value': [10, 20, 30, 40]})
        assert compute_key_uniqueness(df, ['id']) == 0.75
    
    def test_key_uniqueness_all_unique(self):
        """Test key uniqueness with all unique keys."""
        df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        assert compute_key_uniqueness(df, ['id']) == 1.0


class TestConsistencyMetrics:
    """Tests for consistency metrics."""
    
    def test_format_consistency_uniform(self):
        """Test format consistency with uniform data."""
        df = pd.DataFrame({'phone': ['555-1234', '555-5678', '555-9012']})
        consistency = compute_format_consistency(df, 'phone')
        assert consistency > 0.9  # High consistency due to similar lengths
    
    def test_format_consistency_pattern(self):
        """Test format consistency with regex pattern."""
        df = pd.DataFrame({'phone': ['555-1234', '555-5678', 'invalid']})
        pattern = r'\d{3}-\d{4}'
        consistency = compute_format_consistency(df, 'phone', pattern=pattern)
        assert consistency == pytest.approx(2/3, rel=1e-5)
    
    def test_referential_integrity_valid(self):
        """Test referential integrity with all valid references."""
        df = pd.DataFrame({'country_id': [1, 2, 3]})
        reference_values = {1, 2, 3, 4, 5}
        assert compute_referential_integrity(df, 'country_id', reference_values) == 1.0
    
    def test_referential_integrity_invalid(self):
        """Test referential integrity with some invalid references."""
        df = pd.DataFrame({'country_id': [1, 2, 99]})
        reference_values = {1, 2, 3}
        assert compute_referential_integrity(df, 'country_id', reference_values) == pytest.approx(2/3, rel=1e-5)


class TestConstraintMetrics:
    """Tests for constraint violation metrics."""
    
    def test_range_constraint_valid(self):
        """Test range constraint with all valid values."""
        df = pd.DataFrame({'age': [25, 30, 35]})
        constraints = [{'type': 'range', 'column': 'age', 'min': 0, 'max': 120}]
        assert compute_constraint_violation(df, constraints) == 0.0
    
    def test_range_constraint_violations(self):
        """Test range constraint with violations."""
        df = pd.DataFrame({'age': [25, -5, 150, 30]})
        constraints = [{'type': 'range', 'column': 'age', 'min': 0, 'max': 120}]
        assert compute_constraint_violation(df, constraints) == 0.5
    
    def test_enum_constraint_valid(self):
        """Test enum constraint with all valid values."""
        df = pd.DataFrame({'status': ['active', 'inactive', 'active']})
        constraints = [{'type': 'enum', 'column': 'status', 'values': ['active', 'inactive', 'pending']}]
        assert compute_constraint_violation(df, constraints) == 0.0
    
    def test_enum_constraint_violations(self):
        """Test enum constraint with violations."""
        df = pd.DataFrame({'status': ['active', 'invalid', 'pending']})
        constraints = [{'type': 'enum', 'column': 'status', 'values': ['active', 'inactive', 'pending']}]
        assert compute_constraint_violation(df, constraints) == pytest.approx(1/3, rel=1e-5)
    
    def test_custom_constraint(self):
        """Test custom constraint function."""
        df = pd.DataFrame({'value': [10, 20, 5, 30]})
        constraints = [{'type': 'custom', 'column': 'value', 'func': lambda x: x >= 10}]
        assert compute_constraint_violation(df, constraints) == 0.25
    
    def test_type_validity(self):
        """Test type validity metric."""
        df = pd.DataFrame({'age': [25, 30, 35], 'name': ['Alice', 'Bob', 'Charlie']})
        expected_types = {'age': int, 'name': str}
        assert compute_type_validity(df, expected_types) == 1.0


class TestDistributionMetrics:
    """Tests for distribution metrics."""
    
    def test_skewness_normal(self):
        """Test skewness with normal distribution."""
        np.random.seed(42)
        df = pd.DataFrame({'a': np.random.normal(0, 1, 1000)})
        skewness = compute_distribution_skewness(df)
        assert skewness < 0.5  # Should be close to 0 for normal distribution
    
    def test_skewness_skewed(self):
        """Test skewness with skewed distribution."""
        df = pd.DataFrame({'a': [1, 1, 1, 2, 3, 10]})
        skewness = compute_distribution_skewness(df)
        assert skewness > 0.5  # Should be positive for right-skewed
    
    def test_kurtosis(self):
        """Test kurtosis metric."""
        np.random.seed(42)
        df = pd.DataFrame({'a': np.random.normal(0, 1, 1000)})
        kurtosis = compute_distribution_kurtosis(df)
        assert abs(kurtosis) < 1.0  # Should be close to 0 for normal distribution


class TestTimelinessMetrics:
    """Tests for timeliness metrics."""
    
    def test_freshness_recent_data(self):
        """Test freshness with recent data."""
        current = pd.Timestamp.now()
        df = pd.DataFrame({
            'timestamp': [current - timedelta(days=1), current - timedelta(days=2)]
        })
        freshness = compute_freshness(df, 'timestamp', current)
        assert freshness > 0.9  # Very fresh data
    
    def test_freshness_old_data(self):
        """Test freshness with old data."""
        current = pd.Timestamp.now()
        df = pd.DataFrame({
            'timestamp': [current - timedelta(days=365), current - timedelta(days=400)]
        })
        freshness = compute_freshness(df, 'timestamp', current)
        assert freshness < 0.1  # Very stale data


class TestAccuracyMetrics:
    """Tests for accuracy metrics."""
    
    def test_value_accuracy_perfect(self):
        """Test accuracy with perfect match."""
        df = pd.DataFrame({'value': [10, 20, 30]})
        ground_truth = pd.Series([10, 20, 30], index=df.index)
        assert compute_value_accuracy(df, 'value', ground_truth) == 1.0
    
    def test_value_accuracy_numeric(self):
        """Test accuracy with numeric values."""
        df = pd.DataFrame({'value': [10, 20, 30]})
        ground_truth = pd.Series([10, 22, 31], index=df.index)
        accuracy = compute_value_accuracy(df, 'value', ground_truth)
        assert 0.8 < accuracy < 1.0
    
    def test_value_accuracy_categorical(self):
        """Test accuracy with categorical values."""
        df = pd.DataFrame({'label': ['A', 'B', 'C', 'A']})
        ground_truth = pd.Series(['A', 'B', 'C', 'B'], index=df.index)
        assert compute_value_accuracy(df, 'label', ground_truth) == 0.75


class TestCompositeMetrics:
    """Tests for composite metrics."""
    
    def test_overall_quality_default(self):
        """Test overall quality with default weights."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        quality = compute_overall_quality(df)
        assert 0.0 <= quality <= 1.0
    
    def test_overall_quality_custom_weights(self):
        """Test overall quality with custom weights."""
        df = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, 6]})
        weights = {'completeness': 1.0}
        quality = compute_overall_quality(df, weights)
        assert quality == pytest.approx(5/6, rel=1e-5)


class TestQualityMetricClass:
    """Tests for QualityMetric class."""
    
    def test_metric_creation(self):
        """Test creating a quality metric."""
        metric = QualityMetric(
            name="test_metric",
            func=lambda df: 1.0,
            description="Test metric",
            category="test"
        )
        assert metric.name == "test_metric"
        assert metric.category == "test"
    
    def test_metric_compute(self):
        """Test computing a metric."""
        metric = QualityMetric(
            name="test_metric",
            func=lambda df: 0.5,
            description="Test metric"
        )
        df = pd.DataFrame({'a': [1, 2, 3]})
        assert metric.compute(df) == 0.5
    
    def test_metric_with_config(self):
        """Test metric that requires configuration."""
        def test_func(df, threshold=0.5):
            return threshold
        
        metric = QualityMetric(
            name="test_metric",
            func=test_func,
            requires_config=True
        )
        df = pd.DataFrame({'a': [1, 2, 3]})
        assert metric.compute(df, threshold=0.8) == 0.8


class TestQualityMetricRegistry:
    """Tests for QualityMetricRegistry class."""
    
    def test_registry_creation(self):
        """Test creating a registry."""
        registry = QualityMetricRegistry()
        assert len(registry) == 0
    
    def test_register_metric(self):
        """Test registering a metric."""
        registry = QualityMetricRegistry()
        metric = QualityMetric(
            name="test_metric",
            func=lambda df: 1.0,
            category="test"
        )
        registry.register(metric)
        assert len(registry) == 1
        assert "test_metric" in registry
    
    def test_register_duplicate(self):
        """Test registering duplicate metric raises error."""
        registry = QualityMetricRegistry()
        metric = QualityMetric(name="test", func=lambda df: 1.0)
        registry.register(metric)
        
        with pytest.raises(ValueError):
            registry.register(metric)
    
    def test_register_function(self):
        """Test registering a function directly."""
        registry = QualityMetricRegistry()
        registry.register_function(
            name="test_metric",
            func=lambda df: 1.0,
            category="test"
        )
        assert "test_metric" in registry
    
    def test_get_metric(self):
        """Test retrieving a metric."""
        registry = QualityMetricRegistry()
        registry.register_function(name="test", func=lambda df: 1.0)
        metric = registry.get("test")
        assert metric is not None
        assert metric.name == "test"
    
    def test_get_by_category(self):
        """Test retrieving metrics by category."""
        registry = QualityMetricRegistry()
        registry.register_function(name="test1", func=lambda df: 1.0, category="cat1")
        registry.register_function(name="test2", func=lambda df: 1.0, category="cat1")
        registry.register_function(name="test3", func=lambda df: 1.0, category="cat2")
        
        metrics = registry.get_by_category("cat1")
        assert len(metrics) == 2
    
    def test_compute_metric(self):
        """Test computing a metric through registry."""
        registry = QualityMetricRegistry()
        registry.register_function(name="test", func=lambda df: 0.75)
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = registry.compute("test", df)
        assert result == 0.75
    
    def test_compute_all(self):
        """Test computing all metrics."""
        registry = QualityMetricRegistry()
        registry.register_function(name="test1", func=lambda df: 0.5)
        registry.register_function(name="test2", func=lambda df: 0.8)
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        results = registry.compute_all(df)
        assert len(results) == 2
        assert results["test1"] == 0.5
        assert results["test2"] == 0.8
    
    def test_compute_category(self):
        """Test computing all metrics in a category."""
        registry = QualityMetricRegistry()
        registry.register_function(name="test1", func=lambda df: 0.5, category="cat1")
        registry.register_function(name="test2", func=lambda df: 0.8, category="cat1")
        registry.register_function(name="test3", func=lambda df: 0.9, category="cat2")
        
        df = pd.DataFrame({'a': [1, 2, 3]})
        results = registry.compute_category("cat1", df)
        assert len(results) == 2
        assert "test3" not in results


class TestDefaultRegistry:
    """Tests for default registry."""
    
    def test_get_default_registry(self):
        """Test getting default registry."""
        registry = get_default_registry()
        assert len(registry) > 0
    
    def test_default_registry_categories(self):
        """Test default registry has expected categories."""
        registry = get_default_registry()
        categories = registry.list_categories()
        assert "completeness" in categories
        assert "outliers" in categories
        assert "duplicates" in categories
    
    def test_default_registry_metrics(self):
        """Test default registry has expected metrics."""
        registry = get_default_registry()
        metrics = registry.list_metrics()
        assert "completeness" in metrics
        assert "outlier_rate" in metrics
        assert "duplicate_rate" in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
