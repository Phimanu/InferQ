"""
Unit tests for utility functions.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from inferq.utils import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    validate_range_constraint,
    validate_enum_constraint,
    compute_missing_pattern,
    identify_constant_columns,
    identify_high_cardinality_columns,
    normalize_numeric_data,
    compute_data_profile,
    validate_dataframe,
)


class TestOutlierDetection:
    """Tests for outlier detection utilities."""
    
    def test_iqr_no_outliers(self):
        """Test IQR method with no outliers."""
        data = pd.Series([1, 2, 3, 4, 5])
        outliers = detect_outliers_iqr(data)
        assert outliers.sum() == 0
    
    def test_iqr_with_outliers(self):
        """Test IQR method with clear outliers."""
        data = pd.Series([1, 2, 3, 4, 100])
        outliers = detect_outliers_iqr(data, threshold=1.5)
        assert outliers.sum() > 0
    
    def test_zscore_outliers(self):
        """Test Z-score method."""
        data = pd.Series([1, 2, 3, 4, 100])
        outliers = detect_outliers_zscore(data, threshold=2.0)
        assert outliers.sum() > 0


class TestConstraintValidation:
    """Tests for constraint validation utilities."""
    
    def test_range_constraint_valid(self):
        """Test range validation with valid values."""
        data = pd.Series([1, 2, 3, 4, 5])
        valid = validate_range_constraint(data, min_val=0, max_val=10)
        assert valid.all()
    
    def test_range_constraint_violations(self):
        """Test range validation with violations."""
        data = pd.Series([1, 2, -1, 11])
        valid = validate_range_constraint(data, min_val=0, max_val=10)
        assert valid.sum() == 2
    
    def test_enum_constraint_valid(self):
        """Test enum validation with valid values."""
        data = pd.Series(['a', 'b', 'c'])
        valid = validate_enum_constraint(data, ['a', 'b', 'c', 'd'])
        assert valid.all()
    
    def test_enum_constraint_violations(self):
        """Test enum validation with violations."""
        data = pd.Series(['a', 'b', 'x'])
        valid = validate_enum_constraint(data, ['a', 'b', 'c'])
        assert valid.sum() == 2


class TestDataProfiling:
    """Tests for data profiling utilities."""
    
    def test_missing_pattern(self):
        """Test missing pattern computation."""
        df = pd.DataFrame({
            'a': [1, None, 3],
            'b': [4, 5, None]
        })
        patterns = compute_missing_pattern(df)
        assert len(patterns) > 0
    
    def test_constant_columns(self):
        """Test constant column identification."""
        df = pd.DataFrame({
            'a': [1, 1, 1, 1],
            'b': [1, 2, 3, 4]
        })
        constant = identify_constant_columns(df, threshold=0.95)
        assert 'a' in constant
        assert 'b' not in constant
    
    def test_high_cardinality(self):
        """Test high cardinality identification."""
        df = pd.DataFrame({
            'id': range(100),
            'category': ['A'] * 50 + ['B'] * 50
        })
        high_card = identify_high_cardinality_columns(df, threshold=0.9)
        assert 'id' in high_card
        assert 'category' not in high_card


class TestNormalization:
    """Tests for normalization utilities."""
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = pd.Series([1, 2, 3, 4, 5])
        normalized = normalize_numeric_data(data, method='minmax')
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        data = pd.Series([1, 2, 3, 4, 5])
        normalized = normalize_numeric_data(data, method='zscore')
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1


class TestDataProfile:
    """Tests for data profiling."""
    
    def test_compute_profile(self):
        """Test computing data profile."""
        df = pd.DataFrame({
            'a': [1, 2, None, 4, 5],
            'b': ['x', 'y', 'z', 'x', 'y']
        })
        profile = compute_data_profile(df)
        
        assert 'a' in profile
        assert 'b' in profile
        assert profile['a']['missing'] == 1
        assert profile['b']['unique'] == 3


class TestValidateDataFrame:
    """Tests for DataFrame validation."""
    
    def test_validate_schema_valid(self):
        """Test validation with valid data."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'name': ['Alice', 'Bob', 'Charlie']
        })
        schema = {
            'age': {'type': 'int', 'min': 0, 'max': 120},
            'name': {'type': 'str', 'required': True}
        }
        errors = validate_dataframe(df, schema)
        assert len(errors['type_errors']) == 0
        assert len(errors['range_errors']) == 0
    
    def test_validate_schema_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({'age': [25, 30]})
        schema = {'age': {'type': 'int'}, 'name': {'type': 'str'}}
        errors = validate_dataframe(df, schema)
        assert 'name' in errors['missing_columns']


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')
    
    # Run tests manually
    test_outlier = TestOutlierDetection()
    test_outlier.test_iqr_no_outliers()
    test_outlier.test_iqr_with_outliers()
    print("✓ Outlier detection tests passed")
    
    test_constraint = TestConstraintValidation()
    test_constraint.test_range_constraint_valid()
    test_constraint.test_enum_constraint_valid()
    print("✓ Constraint validation tests passed")
    
    test_profile = TestDataProfiling()
    test_profile.test_constant_columns()
    test_profile.test_high_cardinality()
    print("✓ Data profiling tests passed")
    
    test_norm = TestNormalization()
    test_norm.test_minmax_normalization()
    test_norm.test_zscore_normalization()
    print("✓ Normalization tests passed")
    
    print("\n✅ All tests passed!")
