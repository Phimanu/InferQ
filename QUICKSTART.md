# InferQ Quick Start Guide

## Installation

### Step 1: Clone or navigate to the repository

```bash
cd /sc/home/philipp.hildebrandt/InferQ
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Quick Start

### 1. Basic Usage

```python
import pandas as pd
from inferq import get_default_registry

# Load your data
df = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', None, 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 55000, 52000, 58000]
})

# Get the quality metrics registry
registry = get_default_registry()

# Compute basic metrics
completeness = registry.compute('completeness', df)
outlier_rate = registry.compute('outlier_rate', df)
duplicate_rate = registry.compute('duplicate_rate', df)

print(f"Completeness: {completeness:.2%}")
print(f"Outlier Rate: {outlier_rate:.2%}")
print(f"Duplicate Rate: {duplicate_rate:.2%}")
```

### 2. Run Examples

```bash
# Basic usage example
python examples/basic_usage.py

# Advanced custom metrics example
python examples/custom_metrics.py
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=inferq --cov-report=html

# Run specific test file
pytest tests/test_quality_metrics.py -v
```

## Available Quality Metrics

### Completeness
- `completeness`: Overall data completeness
- `column_completeness`: Column-specific completeness (requires config)

### Outliers
- `outlier_rate`: Statistical outlier detection

### Duplicates
- `duplicate_rate`: Duplicate row detection
- `key_uniqueness`: Key column uniqueness (requires config)

### Consistency
- `format_consistency`: Format consistency in columns (requires config)
- `referential_integrity`: Foreign key validation (requires config)

### Constraints
- `constraint_violation`: Constraint violation detection (requires config)
- `type_validity`: Data type validation (requires config)

### Distribution
- `distribution_skewness`: Distribution skewness analysis
- `distribution_kurtosis`: Distribution kurtosis analysis

### Timeliness
- `freshness`: Data freshness based on timestamps (requires config)

### Accuracy
- `value_accuracy`: Accuracy vs ground truth (requires config)

### Composite
- `overall_quality`: Weighted overall quality score

## Key Features

### 1. Easy Metric Computation

```python
# Single metric
score = registry.compute('completeness', df)

# Metric with configuration
score = registry.compute('constraint_violation', df, 
                        constraints=[{'type': 'range', 'column': 'age', 'min': 0, 'max': 120}])

# All metrics at once
scores = registry.compute_all(df)

# Category-specific metrics
scores = registry.compute_category('completeness', df)
```

### 2. Custom Metrics

```python
def my_custom_metric(data_subset: pd.DataFrame) -> float:
    """Your custom quality metric."""
    # Your logic here
    return score

# Register it
registry.register_function(
    name='my_metric',
    func=my_custom_metric,
    category='custom',
    higher_is_better=True
)

# Use it
score = registry.compute('my_metric', df)
```

### 3. Quality Monitoring

```python
# Define quality thresholds
thresholds = {
    'completeness': 0.95,
    'outlier_rate': 0.05,
    'duplicate_rate': 0.01
}

# Monitor quality
for metric_name, threshold in thresholds.items():
    score = registry.compute(metric_name, df)
    metric = registry.get(metric_name)
    
    if metric.higher_is_better:
        status = "✓" if score >= threshold else "✗"
    else:
        status = "✓" if score <= threshold else "✗"
    
    print(f"{status} {metric_name}: {score:.3f} (threshold: {threshold})")
```

## Project Structure

```
InferQ/
├── src/
│   └── inferq/
│       ├── __init__.py           # Main package
│       ├── quality_metrics.py    # Core metrics and registry
│       └── utils.py              # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_quality_metrics.py   # Metric tests
│   └── test_utils.py             # Utility tests
├── examples/
│   ├── __init__.py
│   ├── basic_usage.py            # Basic usage examples
│   └── custom_metrics.py         # Advanced custom metric examples
├── docs/
│   └── WP1_DOCUMENTATION.md      # Detailed documentation
├── README.md                     # Project overview
├── requirements.txt              # Dependencies
└── setup.py                      # Installation config
```

## Next Steps

1. **Explore Examples**: Run the example scripts to see the framework in action
2. **Read Documentation**: Check `docs/WP1_DOCUMENTATION.md` for detailed API reference
3. **Run Tests**: Verify everything works with `pytest tests/`
4. **Build Custom Metrics**: Create domain-specific quality metrics for your data
5. **Integrate**: Use the framework in your data quality monitoring pipeline

## Work Packages Roadmap

- ✅ **WP 1: Quality Metric Framework** - Complete
- ⏳ **WP 2: Multi-Target Quality-aware Discretization (MTQD)** - Coming next
- ⏳ **WP 3: Feature Selection** - Coming next
- ⏳ **WP 4: Index Construction** - Coming next

## Support

For questions or issues, refer to:
- `docs/WP1_DOCUMENTATION.md` for detailed API documentation
- Example scripts in `examples/` for usage patterns
- Test files in `tests/` for reference implementations

## Contributing

When adding new metrics:
1. Implement the metric function in `quality_metrics.py`
2. Register it in `get_default_registry()`
3. Add tests in `tests/test_quality_metrics.py`
4. Update documentation in `docs/WP1_DOCUMENTATION.md`
5. Add examples in `examples/` if needed
