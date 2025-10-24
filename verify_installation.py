#!/usr/bin/env python3
"""
Quick verification script to test the InferQ installation and basic functionality.
"""

import sys
import os

# Add src to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from inferq import get_default_registry, QualityMetricRegistry
        from inferq.quality_metrics import (
            compute_completeness,
            compute_outlier_rate,
            compute_duplicate_rate,
        )
        from inferq.utils import (
            detect_outliers_iqr,
            validate_range_constraint,
            compute_data_profile,
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_metrics():
    """Test basic metric computation."""
    print("\nTesting basic metrics...")
    try:
        from inferq import get_default_registry
        
        # Create sample data
        df = pd.DataFrame({
            'a': [1, 2, 3, None, 5],
            'b': [10, 20, 30, 40, 50]
        })
        
        registry = get_default_registry()
        
        # Test completeness
        completeness = registry.compute('completeness', df)
        assert 0.0 <= completeness <= 1.0, "Completeness out of range"
        
        # Test outlier rate
        outlier_rate = registry.compute('outlier_rate', df)
        assert 0.0 <= outlier_rate <= 1.0, "Outlier rate out of range"
        
        # Test duplicate rate
        duplicate_rate = registry.compute('duplicate_rate', df)
        assert 0.0 <= duplicate_rate <= 1.0, "Duplicate rate out of range"
        
        print(f"✓ Basic metrics working (completeness: {completeness:.3f})")
        return True
    except Exception as e:
        print(f"✗ Basic metrics failed: {e}")
        return False


def test_registry():
    """Test registry functionality."""
    print("\nTesting registry...")
    try:
        from inferq import get_default_registry
        
        registry = get_default_registry()
        
        # Check registry has metrics
        assert len(registry) > 0, "Registry is empty"
        
        # Check categories exist
        categories = registry.list_categories()
        assert 'completeness' in categories, "Missing completeness category"
        assert 'outliers' in categories, "Missing outliers category"
        
        # Check specific metrics exist
        assert 'completeness' in registry, "Missing completeness metric"
        assert 'outlier_rate' in registry, "Missing outlier_rate metric"
        
        print(f"✓ Registry working ({len(registry)} metrics, {len(categories)} categories)")
        return True
    except Exception as e:
        print(f"✗ Registry failed: {e}")
        return False


def test_custom_metric():
    """Test custom metric registration."""
    print("\nTesting custom metrics...")
    try:
        from inferq.quality_metrics import QualityMetricRegistry
        
        registry = QualityMetricRegistry()
        
        # Create and register custom metric
        def my_metric(df):
            return 0.75
        
        registry.register_function(
            name='test_metric',
            func=my_metric,
            category='test'
        )
        
        # Test it
        df = pd.DataFrame({'a': [1, 2, 3]})
        result = registry.compute('test_metric', df)
        assert result == 0.75, "Custom metric returned wrong value"
        
        print("✓ Custom metrics working")
        return True
    except Exception as e:
        print(f"✗ Custom metrics failed: {e}")
        return False


def test_constraints():
    """Test constraint validation."""
    print("\nTesting constraint validation...")
    try:
        from inferq import get_default_registry
        
        df = pd.DataFrame({
            'age': [25, 30, -5, 150],
            'status': ['active', 'inactive', 'active', 'invalid']
        })
        
        registry = get_default_registry()
        
        # Test range constraint
        constraints = [
            {'type': 'range', 'column': 'age', 'min': 0, 'max': 120}
        ]
        violations = registry.compute('constraint_violation', df, constraints=constraints)
        assert violations > 0, "Should detect violations"
        
        # Test enum constraint
        constraints = [
            {'type': 'enum', 'column': 'status', 'values': ['active', 'inactive']}
        ]
        violations = registry.compute('constraint_violation', df, constraints=constraints)
        assert violations > 0, "Should detect violations"
        
        print("✓ Constraint validation working")
        return True
    except Exception as e:
        print(f"✗ Constraint validation failed: {e}")
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utilities...")
    try:
        from inferq.utils import (
            detect_outliers_iqr,
            validate_range_constraint,
            normalize_numeric_data,
        )
        
        # Test outlier detection
        data = pd.Series([1, 2, 3, 100])
        outliers = detect_outliers_iqr(data)
        assert outliers.any(), "Should detect outliers"
        
        # Test range validation
        valid = validate_range_constraint(data, min_val=0, max_val=50)
        assert not valid.all(), "Should detect invalid values"
        
        # Test normalization
        normalized = normalize_numeric_data(data, method='minmax')
        assert normalized.min() >= 0 and normalized.max() <= 1, "Normalization failed"
        
        print("✓ Utilities working")
        return True
    except Exception as e:
        print(f"✗ Utilities failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("InferQ Installation Verification")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_basic_metrics,
        test_registry,
        test_custom_metric,
        test_constraints,
        test_utils,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 70)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("=" * 70)
        print("\nInferQ is ready to use!")
        print("Run 'python examples/basic_usage.py' to see it in action.")
        return 0
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
