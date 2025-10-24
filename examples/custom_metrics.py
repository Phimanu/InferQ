"""
Advanced example: Creating custom quality metrics and registries.

This example shows how to:
1. Create custom quality metrics
2. Build a custom registry
3. Combine built-in and custom metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from inferq.quality_metrics import QualityMetric, QualityMetricRegistry, get_default_registry


# ============================================================================
# CUSTOM METRIC FUNCTIONS
# ============================================================================

def compute_email_validity(data_subset: pd.DataFrame, email_column: str = 'email') -> float:
    """
    Custom metric: Check email format validity.
    
    Args:
        data_subset: DataFrame to check
        email_column: Name of email column
        
    Returns:
        Ratio of valid emails
    """
    if data_subset.empty or email_column not in data_subset.columns:
        return 1.0
    
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    emails = data_subset[email_column].dropna()
    if len(emails) == 0:
        return 1.0
    
    valid_emails = emails.str.match(email_pattern).sum()
    return valid_emails / len(emails)


def compute_phone_validity(data_subset: pd.DataFrame, phone_column: str = 'phone') -> float:
    """
    Custom metric: Check phone format validity.
    
    Args:
        data_subset: DataFrame to check
        phone_column: Name of phone column
        
    Returns:
        Ratio of valid phone numbers
    """
    if data_subset.empty or phone_column not in data_subset.columns:
        return 1.0
    
    import re
    # Simple US phone pattern
    phone_pattern = r'^\d{3}-\d{3}-\d{4}$|^\d{10}$'
    
    phones = data_subset[phone_column].dropna()
    if len(phones) == 0:
        return 1.0
    
    valid_phones = phones.astype(str).str.match(phone_pattern).sum()
    return valid_phones / len(phones)


def compute_business_rule_compliance(data_subset: pd.DataFrame) -> float:
    """
    Custom metric: Check compliance with business rules.
    
    Example: Age must be >= 18 for active status.
    
    Args:
        data_subset: DataFrame to check
        
    Returns:
        Ratio of compliant rows
    """
    if data_subset.empty:
        return 1.0
    
    required_cols = {'age', 'status'}
    if not required_cols.issubset(data_subset.columns):
        return 1.0
    
    # Business rule: active users must be >= 18
    active_users = data_subset[data_subset['status'] == 'active']
    
    if len(active_users) == 0:
        return 1.0
    
    compliant = (active_users['age'] >= 18).sum()
    return compliant / len(active_users)


def compute_data_freshness_days(data_subset: pd.DataFrame, 
                               timestamp_column: str = 'last_update') -> float:
    """
    Custom metric: Average data age in days (lower is better).
    
    Args:
        data_subset: DataFrame to check
        timestamp_column: Name of timestamp column
        
    Returns:
        Average age in days (normalized to [0, 1], where 1 is fresh)
    """
    if data_subset.empty or timestamp_column not in data_subset.columns:
        return 0.0
    
    timestamps = pd.to_datetime(data_subset[timestamp_column], errors='coerce').dropna()
    
    if len(timestamps) == 0:
        return 0.0
    
    current_time = pd.Timestamp.now()
    ages_days = (current_time - timestamps).dt.total_seconds() / (24 * 3600)
    avg_age = ages_days.mean()
    
    # Normalize: data older than 30 days gets score close to 0
    freshness = max(0.0, 1.0 - (avg_age / 30.0))
    return freshness


# ============================================================================
# EXAMPLE 1: Creating a Custom Registry
# ============================================================================

def example_custom_registry():
    """Create and use a custom registry with domain-specific metrics."""
    print("\n" + "=" * 70)
    print("Example 1: Custom Registry")
    print("=" * 70)
    
    # Create a new registry
    registry = QualityMetricRegistry()
    
    # Register custom metrics
    registry.register_function(
        name='email_validity',
        func=compute_email_validity,
        description='Validates email format',
        category='format',
        higher_is_better=True,
        requires_config=True
    )
    
    registry.register_function(
        name='phone_validity',
        func=compute_phone_validity,
        description='Validates phone format',
        category='format',
        higher_is_better=True,
        requires_config=True
    )
    
    registry.register_function(
        name='business_rule_compliance',
        func=compute_business_rule_compliance,
        description='Checks business rule compliance',
        category='business_rules',
        higher_is_better=True,
        requires_config=False
    )
    
    # Create sample data
    df = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'email': ['valid@example.com', 'invalid', 'test@test.com', 'bad@', 'good@mail.com'],
        'phone': ['123-456-7890', '1234567890', 'invalid', '555-555-5555', '999-999-9999'],
        'age': [25, 30, 17, 35, 40],
        'status': ['active', 'active', 'active', 'inactive', 'active']
    })
    
    print("\nSample Data:")
    print(df)
    
    # Compute custom metrics
    print("\nCustom Metrics:")
    email_validity = registry.compute('email_validity', df, email_column='email')
    print(f"   Email Validity: {email_validity:.3f}")
    
    phone_validity = registry.compute('phone_validity', df, phone_column='phone')
    print(f"   Phone Validity: {phone_validity:.3f}")
    
    business_compliance = registry.compute('business_rule_compliance', df)
    print(f"   Business Rule Compliance: {business_compliance:.3f}")


# ============================================================================
# EXAMPLE 2: Extending the Default Registry
# ============================================================================

def example_extend_default_registry():
    """Extend the default registry with custom metrics."""
    print("\n" + "=" * 70)
    print("Example 2: Extending Default Registry")
    print("=" * 70)
    
    # Get default registry
    registry = get_default_registry()
    
    print(f"\nDefault registry has {len(registry)} metrics")
    
    # Add custom metrics to it
    registry.register_function(
        name='email_validity',
        func=compute_email_validity,
        description='Validates email format',
        category='format',
        higher_is_better=True,
        requires_config=True
    )
    
    registry.register_function(
        name='data_freshness_days',
        func=compute_data_freshness_days,
        description='Data freshness in days',
        category='timeliness',
        higher_is_better=True,
        requires_config=True
    )
    
    print(f"Extended registry now has {len(registry)} metrics")
    print(f"Categories: {', '.join(registry.list_categories())}")
    
    # Create sample data
    df = pd.DataFrame({
        'email': ['valid@example.com', 'test@test.com', 'good@mail.com'],
        'last_update': [
            pd.Timestamp.now() - pd.Timedelta(days=1),
            pd.Timestamp.now() - pd.Timedelta(days=5),
            pd.Timestamp.now() - pd.Timedelta(days=10)
        ],
        'value': [1, 2, 3]
    })
    
    # Compute both built-in and custom metrics
    print("\nMetrics (built-in + custom):")
    
    # Built-in
    completeness = registry.compute('completeness', df)
    print(f"   Completeness (built-in): {completeness:.3f}")
    
    # Custom
    email_validity = registry.compute('email_validity', df, email_column='email')
    print(f"   Email Validity (custom): {email_validity:.3f}")
    
    freshness = registry.compute('data_freshness_days', df, timestamp_column='last_update')
    print(f"   Data Freshness (custom): {freshness:.3f}")


# ============================================================================
# EXAMPLE 3: Creating a Metric Object with Complex Logic
# ============================================================================

def example_complex_metric():
    """Create a complex metric using the QualityMetric class."""
    print("\n" + "=" * 70)
    print("Example 3: Complex Custom Metric")
    print("=" * 70)
    
    def compute_multi_column_consistency(data_subset: pd.DataFrame,
                                        column_pairs: list) -> float:
        """
        Check consistency across multiple column pairs.
        
        For example: if 'country' is 'US', then 'zip_code' should be 5 digits.
        """
        if data_subset.empty:
            return 1.0
        
        total_checks = 0
        consistent = 0
        
        for col1, col2, check_func in column_pairs:
            if col1 not in data_subset.columns or col2 not in data_subset.columns:
                continue
            
            for _, row in data_subset.iterrows():
                if pd.notna(row[col1]) and pd.notna(row[col2]):
                    total_checks += 1
                    if check_func(row[col1], row[col2]):
                        consistent += 1
        
        return consistent / total_checks if total_checks > 0 else 1.0
    
    # Create the metric
    consistency_metric = QualityMetric(
        name='multi_column_consistency',
        func=compute_multi_column_consistency,
        description='Checks consistency across multiple column pairs',
        category='consistency',
        higher_is_better=True,
        requires_config=True
    )
    
    # Create sample data
    df = pd.DataFrame({
        'country': ['US', 'US', 'CA', 'US', 'CA'],
        'zip_code': ['12345', '67890', 'A1B2C3', 'invalid', 'X1Y2Z3']
    })
    
    print("\nSample Data:")
    print(df)
    
    # Define check functions
    def us_zip_check(country, zip_code):
        return len(str(zip_code)) == 5 if country == 'US' else True
    
    # Compute the metric
    column_pairs = [('country', 'zip_code', us_zip_check)]
    consistency = consistency_metric.compute(df, column_pairs=column_pairs)
    
    print(f"\nMulti-Column Consistency: {consistency:.3f}")


# ============================================================================
# EXAMPLE 4: Using Metrics for Data Profiling
# ============================================================================

def example_data_profiling():
    """Use metrics to create a comprehensive data quality profile."""
    print("\n" + "=" * 70)
    print("Example 4: Comprehensive Data Quality Profiling")
    print("=" * 70)
    
    # Get registry with custom metrics
    registry = get_default_registry()
    registry.register_function(
        name='email_validity',
        func=compute_email_validity,
        category='format',
        requires_config=True
    )
    
    # Create sample data with multiple issues
    np.random.seed(42)
    df = pd.DataFrame({
        'customer_id': range(1, 101),
        'email': ['valid@test.com' if i % 5 != 0 else 'invalid' 
                  for i in range(100)],
        'age': [np.random.randint(18, 80) if i % 10 != 0 else -1 
                for i in range(100)],
        'salary': [np.random.randint(30000, 100000) if i % 20 != 0 else None 
                   for i in range(100)],
        'status': [np.random.choice(['active', 'inactive']) if i % 15 != 0 else 'invalid' 
                   for i in range(100)]
    })
    
    print(f"\nDataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Compute comprehensive quality profile
    print("\nQuality Profile:")
    print("-" * 70)
    
    # Basic metrics
    completeness = registry.compute('completeness', df)
    print(f"   Overall Completeness: {completeness:.3f}")
    
    outlier_rate = registry.compute('outlier_rate', df)
    print(f"   Outlier Rate: {outlier_rate:.3f}")
    
    duplicate_rate = registry.compute('duplicate_rate', df)
    print(f"   Duplicate Rate: {duplicate_rate:.3f}")
    
    # Custom metric
    email_validity = registry.compute('email_validity', df, email_column='email')
    print(f"   Email Validity: {email_validity:.3f}")
    
    # Constraint violations
    constraints = [
        {'type': 'range', 'column': 'age', 'min': 0, 'max': 120},
        {'type': 'enum', 'column': 'status', 'values': ['active', 'inactive']}
    ]
    violations = registry.compute('constraint_violation', df, constraints=constraints)
    print(f"   Constraint Violations: {violations:.3f}")
    
    # Calculate quality score
    weights = {
        'completeness': 0.25,
        'outlier_rate': 0.25,
        'duplicate_rate': 0.25,
    }
    overall = registry.compute('overall_quality', df, metric_weights=weights)
    print(f"\n   Overall Quality Score: {overall:.3f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("InferQ - Advanced Custom Metrics Examples")
    print("=" * 70)
    
    example_custom_registry()
    example_extend_default_registry()
    example_complex_metric()
    example_data_profiling()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
