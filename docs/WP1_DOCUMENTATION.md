# WP 1: Quality Metric Framework - Detailed Documentation

## Overview

The Quality Metric Framework is the foundation of the InferQ system. It provides a comprehensive library of quality metric functions and a central registry system for managing and computing quality metrics on data subsets.

## Architecture

The framework consists of three main components:

1. **Quality Metric Functions**: Individual functions that compute specific quality metrics
2. **QualityMetric Class**: A wrapper class that adds metadata and behavior to metric functions
3. **QualityMetricRegistry**: A central registry for managing and accessing all metrics

## Quality Metrics Library

### Completeness Metrics

#### `compute_completeness(data_subset: pd.DataFrame) -> float`
Computes the overall completeness of the dataset as the ratio of non-missing values to total values.

**Returns**: Score in [0, 1], where 1 means no missing values

**Example**:
```python
df = pd.DataFrame({'a': [1, 2, None], 'b': [4, 5, 6]})
score = compute_completeness(df)  # Returns 0.833
```

#### `compute_column_completeness(data_subset: pd.DataFrame, column: str) -> float`
Computes completeness for a specific column.

**Parameters**:
- `column`: Name of the column to check

**Returns**: Score in [0, 1]

### Outlier Detection Metrics

#### `compute_outlier_rate(data_subset: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> float`
Detects outliers using various statistical methods.

**Parameters**:
- `method`: Detection method - 'iqr', 'zscore', or 'modified_zscore'
- `threshold`: Threshold for outlier detection
  - IQR: typically 1.5 or 3.0
  - Z-score: typically 3.0

**Returns**: Outlier rate in [0, 1], where 0 means no outliers

**Supported Methods**:
- **IQR (Interquartile Range)**: Classic box-plot method
- **Z-score**: Based on standard deviations from mean
- **Modified Z-score**: More robust to extreme outliers, uses MAD

### Duplicate Detection Metrics

#### `compute_duplicate_rate(data_subset: pd.DataFrame, subset: Optional[List[str]] = None) -> float`
Computes the rate of duplicate rows.

**Parameters**:
- `subset`: List of columns to consider (if None, all columns used)

**Returns**: Duplicate rate in [0, 1], where 0 means no duplicates

#### `compute_key_uniqueness(data_subset: pd.DataFrame, key_columns: List[str]) -> float`
Checks uniqueness of specified key columns.

**Parameters**:
- `key_columns`: List of columns that should form a unique key

**Returns**: Uniqueness score in [0, 1], where 1 means all keys are unique

### Consistency Metrics

#### `compute_format_consistency(data_subset: pd.DataFrame, column: str, pattern: Optional[str] = None) -> float`
Checks format consistency within a column.

**Parameters**:
- `column`: Column name to check
- `pattern`: Optional regex pattern to match

**Returns**: Consistency score in [0, 1]

#### `compute_referential_integrity(data_subset: pd.DataFrame, foreign_key: str, reference_values: set) -> float`
Validates foreign key relationships.

**Parameters**:
- `foreign_key`: Column name of the foreign key
- `reference_values`: Set of valid reference values

**Returns**: Integrity score in [0, 1]

### Constraint Violation Metrics

#### `compute_constraint_violation(data_subset: pd.DataFrame, constraints: List[Dict[str, Any]]) -> float`
Checks violations of data constraints.

**Parameters**:
- `constraints`: List of constraint dictionaries with:
  - `type`: 'range', 'enum', or 'custom'
  - `column`: column name
  - `min`/`max`: for range constraints
  - `values`: for enum constraints
  - `func`: for custom validation

**Returns**: Violation rate in [0, 1], where 0 means no violations

**Example**:
```python
constraints = [
    {'type': 'range', 'column': 'age', 'min': 0, 'max': 120},
    {'type': 'enum', 'column': 'status', 'values': ['active', 'inactive']},
    {'type': 'custom', 'column': 'email', 'func': lambda x: '@' in str(x)}
]
violation_rate = compute_constraint_violation(df, constraints)
```

#### `compute_type_validity(data_subset: pd.DataFrame, expected_types: Dict[str, type]) -> float`
Validates data types against expected types.

**Parameters**:
- `expected_types`: Dictionary mapping column names to expected Python types

**Returns**: Validity score in [0, 1]

### Distribution Metrics

#### `compute_distribution_skewness(data_subset: pd.DataFrame) -> float`
Computes average skewness across numeric columns.

**Returns**: Average absolute skewness

#### `compute_distribution_kurtosis(data_subset: pd.DataFrame) -> float`
Computes average excess kurtosis across numeric columns.

**Returns**: Average absolute excess kurtosis

### Timeliness Metrics

#### `compute_freshness(data_subset: pd.DataFrame, timestamp_column: str, current_time: Optional[pd.Timestamp] = None) -> float`
Measures data freshness based on timestamps.

**Parameters**:
- `timestamp_column`: Name of the timestamp column
- `current_time`: Reference time (default: now)

**Returns**: Freshness score in [0, 1], using exponential decay with 30-day half-life

### Accuracy Metrics

#### `compute_value_accuracy(data_subset: pd.DataFrame, column: str, ground_truth: pd.Series) -> float`
Compares values against ground truth.

**Parameters**:
- `column`: Column to check
- `ground_truth`: Series with correct values

**Returns**: Accuracy score in [0, 1]

### Composite Metrics

#### `compute_overall_quality(data_subset: pd.DataFrame, metric_weights: Optional[Dict[str, float]] = None) -> float`
Computes overall quality as weighted combination of metrics.

**Parameters**:
- `metric_weights`: Dictionary mapping metric names to weights

**Returns**: Overall quality score in [0, 1]

## Quality Metric Registry System

### QualityMetric Class

The `QualityMetric` class wraps metric functions with metadata:

```python
metric = QualityMetric(
    name="completeness",
    func=compute_completeness,
    description="Ratio of non-missing values",
    category="completeness",
    higher_is_better=True,
    requires_config=False
)
```

**Attributes**:
- `name`: Unique identifier
- `func`: The metric computation function
- `description`: Human-readable description
- `category`: Category for organization
- `higher_is_better`: Whether higher values indicate better quality
- `requires_config`: Whether the metric needs additional configuration

### QualityMetricRegistry Class

The registry manages all quality metrics:

#### Creating a Registry

```python
from inferq import get_default_registry

# Get default registry with all built-in metrics
registry = get_default_registry()

# Or create a custom empty registry
from inferq.quality_metrics import QualityMetricRegistry
custom_registry = QualityMetricRegistry()
```

#### Registering Metrics

```python
# Register a function directly
registry.register_function(
    name='my_metric',
    func=my_metric_function,
    description='Custom metric',
    category='custom',
    higher_is_better=True,
    requires_config=False
)

# Or register a QualityMetric object
metric = QualityMetric(name='my_metric', func=my_metric_function)
registry.register(metric)
```

#### Computing Metrics

```python
# Compute a single metric
score = registry.compute('completeness', df)

# Compute with configuration
score = registry.compute('column_completeness', df, column='age')

# Compute all metrics (that don't require config)
scores = registry.compute_all(df)

# Compute all metrics with configurations
metric_configs = {
    'constraint_violation': {
        'constraints': [{'type': 'range', 'column': 'age', 'min': 0, 'max': 120}]
    }
}
scores = registry.compute_all(df, metric_configs)

# Compute all metrics in a category
scores = registry.compute_category('completeness', df)
```

#### Querying the Registry

```python
# List all metrics
metric_names = registry.list_metrics()

# List all categories
categories = registry.list_categories()

# Get a specific metric
metric = registry.get('completeness')

# Get all metrics in a category
metrics = registry.get_by_category('outliers')

# Check if metric exists
if 'completeness' in registry:
    print("Metric exists")
```

## Usage Patterns

### Pattern 1: Basic Quality Assessment

```python
from inferq import get_default_registry
import pandas as pd

df = pd.read_csv('data.csv')
registry = get_default_registry()

# Compute basic metrics
completeness = registry.compute('completeness', df)
outlier_rate = registry.compute('outlier_rate', df)
duplicate_rate = registry.compute('duplicate_rate', df)

print(f"Completeness: {completeness:.2%}")
print(f"Outlier Rate: {outlier_rate:.2%}")
print(f"Duplicate Rate: {duplicate_rate:.2%}")
```

### Pattern 2: Custom Metric Development

```python
def compute_domain_specific_metric(data_subset: pd.DataFrame) -> float:
    """Custom metric for domain-specific logic."""
    # Your custom logic here
    return score

# Register it
registry.register_function(
    name='domain_metric',
    func=compute_domain_specific_metric,
    category='domain',
    higher_is_better=True
)

# Use it
score = registry.compute('domain_metric', df)
```

### Pattern 3: Comprehensive Data Profiling

```python
# Define all configurations
metric_configs = {
    'constraint_violation': {
        'constraints': [
            {'type': 'range', 'column': 'age', 'min': 0, 'max': 120},
            {'type': 'enum', 'column': 'status', 'values': ['active', 'inactive']}
        ]
    },
    'key_uniqueness': {'key_columns': ['id']},
    'column_completeness': {'column': 'critical_field'}
}

# Compute all metrics
all_scores = registry.compute_all(df, metric_configs)

# Analyze results
for metric_name, score in sorted(all_scores.items()):
    print(f"{metric_name}: {score:.3f}")
```

### Pattern 4: Quality Monitoring Pipeline

```python
def monitor_data_quality(df: pd.DataFrame, 
                        thresholds: dict) -> dict:
    """Monitor data quality and generate alerts."""
    registry = get_default_registry()
    
    results = {
        'passed': [],
        'failed': [],
        'scores': {}
    }
    
    for metric_name, threshold in thresholds.items():
        score = registry.compute(metric_name, df)
        results['scores'][metric_name] = score
        
        metric = registry.get(metric_name)
        if metric.higher_is_better:
            if score >= threshold:
                results['passed'].append(metric_name)
            else:
                results['failed'].append(metric_name)
        else:
            if score <= threshold:
                results['passed'].append(metric_name)
            else:
                results['failed'].append(metric_name)
    
    return results

# Usage
thresholds = {
    'completeness': 0.95,
    'outlier_rate': 0.05,
    'duplicate_rate': 0.01
}

results = monitor_data_quality(df, thresholds)
print(f"Passed: {len(results['passed'])}")
print(f"Failed: {len(results['failed'])}")
```

## Built-in Metrics Summary

| Metric Name | Category | Requires Config | Higher is Better | Description |
|-------------|----------|-----------------|------------------|-------------|
| completeness | completeness | No | Yes | Overall data completeness |
| column_completeness | completeness | Yes | Yes | Column-specific completeness |
| outlier_rate | outliers | No | No | Rate of statistical outliers |
| duplicate_rate | duplicates | No | No | Rate of duplicate rows |
| key_uniqueness | duplicates | Yes | Yes | Uniqueness of key columns |
| format_consistency | consistency | Yes | Yes | Format consistency in column |
| referential_integrity | consistency | Yes | Yes | Foreign key validity |
| constraint_violation | constraints | Yes | No | Constraint violation rate |
| type_validity | constraints | Yes | Yes | Data type validity |
| distribution_skewness | distribution | No | No | Distribution skewness |
| distribution_kurtosis | distribution | No | No | Distribution kurtosis |
| freshness | timeliness | Yes | Yes | Data freshness by timestamp |
| value_accuracy | accuracy | Yes | Yes | Accuracy vs ground truth |
| overall_quality | composite | No | Yes | Weighted overall quality |

## Best Practices

1. **Use the Default Registry**: Start with `get_default_registry()` which includes all built-in metrics
2. **Choose Appropriate Metrics**: Select metrics that align with your data quality requirements
3. **Configure Properly**: Provide necessary configurations for metrics that require them
4. **Interpret Results**: Remember which metrics are "higher is better" vs "lower is better"
5. **Combine Metrics**: Use `overall_quality` or create custom composites for holistic assessment
6. **Monitor Trends**: Track metrics over time to detect quality degradation
7. **Set Thresholds**: Define acceptable quality thresholds for your use case
8. **Document Custom Metrics**: When creating custom metrics, document their purpose and interpretation

## Extension Points

The framework is designed to be extensible:

1. **Custom Metrics**: Create domain-specific metric functions
2. **Custom Categories**: Organize metrics into custom categories
3. **Custom Registries**: Build specialized registries for different data types or domains
4. **Metric Composition**: Combine multiple metrics into new composite metrics
5. **Integration**: Integrate with data quality monitoring and alerting systems

## Next Steps

After implementing WP 1, the framework will serve as the foundation for:
- **WP 2**: Multi-Target Quality-aware Discretization (MTQD)
- **WP 3**: Feature Selection based on quality importance
- **WP 4**: Index Construction for real-time quality prediction

The quality metrics defined here will be used throughout the pipeline to:
1. Annotate data bins with quality scores
2. Optimize discretization for multiple quality dimensions
3. Train quality prediction models
4. Monitor incoming data in real-time
