"""
Basic usage example for the InferQ Quality Metrics Framework.

This example demonstrates how to use the quality metrics library
to assess data quality on a sample dataset.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from inferq import get_default_registry


def create_sample_data():
    """Create a sample dataset with various quality issues."""
    np.random.seed(42)
    
    data = {
        'customer_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'name': ['Alice', 'Bob', None, 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
        'age': [25, 30, 35, -5, 150, 28, 32, 27, 29, 31],  # Quality issues: negative age, age > 120
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 
                  'invalid', 'eve@example.com', 'frank@example.com',
                  'grace@example.com', 'henry@example.com', 'ivy@example.com', 'jack@example.com'],
        'salary': [50000, 60000, 55000, 52000, 58000, 1000000, 54000, 51000, 53000, 56000],  # Outlier: 1M
        'status': ['active', 'active', 'inactive', 'active', 'invalid_status', 
                   'active', 'inactive', 'active', 'active', 'inactive']
    }
    
    return pd.DataFrame(data)


def main():
    """Main example function."""
    print("=" * 70)
    print("InferQ Quality Metrics Framework - Basic Example")
    print("=" * 70)
    
    # Create sample data
    df = create_sample_data()
    print("\n1. Sample Dataset:")
    print(df)
    print(f"\nDataset shape: {df.shape}")
    
    # Get the default registry
    registry = get_default_registry()
    print(f"\n2. Available Metrics: {len(registry)} metrics registered")
    print(f"   Categories: {', '.join(registry.list_categories())}")
    
    # Compute basic metrics (no configuration required)
    print("\n3. Basic Quality Metrics:")
    print("-" * 70)
    
    # Completeness
    completeness = registry.compute('completeness', df)
    print(f"   Overall Completeness: {completeness:.3f}")
    
    # Outlier rate
    outlier_rate = registry.compute('outlier_rate', df)
    print(f"   Outlier Rate: {outlier_rate:.3f}")
    
    # Duplicate rate
    duplicate_rate = registry.compute('duplicate_rate', df)
    print(f"   Duplicate Rate: {duplicate_rate:.3f}")
    
    # Distribution metrics
    skewness = registry.compute('distribution_skewness', df)
    kurtosis = registry.compute('distribution_kurtosis', df)
    print(f"   Distribution Skewness: {skewness:.3f}")
    print(f"   Distribution Kurtosis: {kurtosis:.3f}")
    
    # Compute metrics with configuration
    print("\n4. Metrics with Configuration:")
    print("-" * 70)
    
    # Column-specific completeness
    name_completeness = registry.compute('column_completeness', df, column='name')
    print(f"   Name Column Completeness: {name_completeness:.3f}")
    
    # Constraint violations
    age_constraints = [
        {'type': 'range', 'column': 'age', 'min': 0, 'max': 120}
    ]
    constraint_violations = registry.compute('constraint_violation', df, 
                                            constraints=age_constraints)
    print(f"   Age Constraint Violations: {constraint_violations:.3f}")
    
    # Enum constraint for status
    status_constraints = [
        {'type': 'enum', 'column': 'status', 'values': ['active', 'inactive', 'pending']}
    ]
    status_violations = registry.compute('constraint_violation', df, 
                                        constraints=status_constraints)
    print(f"   Status Constraint Violations: {status_violations:.3f}")
    
    # Key uniqueness
    key_uniqueness = registry.compute('key_uniqueness', df, key_columns=['customer_id'])
    print(f"   Customer ID Uniqueness: {key_uniqueness:.3f}")
    
    # Compute all metrics in a category
    print("\n5. Metrics by Category:")
    print("-" * 70)
    
    # Completeness category
    completeness_metrics = registry.compute_category('completeness', df, 
                                                     metric_configs={'column_completeness': {'column': 'name'}})
    print("   Completeness Metrics:")
    for metric_name, value in completeness_metrics.items():
        print(f"      {metric_name}: {value:.3f}")
    
    # Outlier category
    outlier_metrics = registry.compute_category('outliers', df)
    print("   Outlier Metrics:")
    for metric_name, value in outlier_metrics.items():
        print(f"      {metric_name}: {value:.3f}")
    
    # Custom metric configurations
    print("\n6. Advanced: Custom Metric Configuration:")
    print("-" * 70)
    
    # Create a custom metric config dictionary
    metric_configs = {
        'constraint_violation': {
            'constraints': [
                {'type': 'range', 'column': 'age', 'min': 0, 'max': 120},
                {'type': 'range', 'column': 'salary', 'min': 0, 'max': 500000},
                {'type': 'enum', 'column': 'status', 'values': ['active', 'inactive', 'pending']}
            ]
        },
        'key_uniqueness': {
            'key_columns': ['customer_id']
        }
    }
    
    # Compute all metrics with configurations
    all_results = registry.compute_all(df, metric_configs)
    print("   All Computed Metrics:")
    for metric_name, value in sorted(all_results.items()):
        print(f"      {metric_name}: {value:.3f}")
    
    # Overall quality score
    print("\n7. Overall Quality Assessment:")
    print("-" * 70)
    
    weights = {
        'completeness': 0.3,
        'outlier_rate': 0.3,
        'duplicate_rate': 0.2
    }
    overall = registry.compute('overall_quality', df, metric_weights=weights)
    print(f"   Overall Quality Score: {overall:.3f}")
    
    # Interpretation
    if overall >= 0.9:
        quality_level = "Excellent"
    elif overall >= 0.7:
        quality_level = "Good"
    elif overall >= 0.5:
        quality_level = "Fair"
    else:
        quality_level = "Poor"
    
    print(f"   Quality Level: {quality_level}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
