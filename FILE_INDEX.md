# InferQ Project File Index

Quick reference guide to all project files and their purposes.

## 📚 Documentation

| File | Description |
|------|-------------|
| `README.md` | Project overview and quick start |
| `QUICKSTART.md` | Detailed quick start guide with examples |
| `WP1_SUMMARY.md` | Complete WP1 implementation summary |
| `docs/WP1_DOCUMENTATION.md` | Comprehensive API reference and documentation |

## 🔧 Configuration & Setup

| File | Description |
|------|-------------|
| `setup.py` | Package installation configuration |
| `requirements.txt` | Python dependencies |
| `verify_installation.py` | Installation verification script |

## 💻 Source Code

### Core Framework (`src/inferq/`)

| File | Lines | Description |
|------|-------|-------------|
| `__init__.py` | 20 | Package initialization and exports |
| `quality_metrics.py` | 900+ | Core quality metrics and registry system |
| `utils.py` | 450+ | Utility functions for quality analysis |

### Key Components in `quality_metrics.py`:
- 14 quality metric functions
- `QualityMetric` class (metric wrapper)
- `QualityMetricRegistry` class (central registry)
- `get_default_registry()` function
- `get_global_registry()` singleton

### Key Functions in `utils.py`:
- Outlier detection (IQR, Z-score, Modified Z-score)
- Constraint validation (range, enum)
- Data profiling (missing patterns, cardinality, correlation)
- Normalization (min-max, z-score, robust)
- Data drift detection (KS test, Chi-squared)
- DataFrame validation

## 🧪 Tests (`tests/`)

| File | Lines | Description |
|------|-------|-------------|
| `test_quality_metrics.py` | 600+ | Tests for all quality metrics (14 test classes, 50+ tests) |
| `test_utils.py` | 200+ | Tests for utility functions (5 test classes, 15+ tests) |

### Test Coverage:
- Completeness metrics
- Outlier detection
- Duplicate detection
- Consistency metrics
- Constraint validation
- Distribution metrics
- Timeliness metrics
- Accuracy metrics
- Registry system
- Custom metrics
- Utility functions

## 📖 Examples (`examples/`)

| File | Lines | Description |
|------|-------|-------------|
| `basic_usage.py` | 200+ | Basic usage with sample data and common patterns |
| `custom_metrics.py` | 400+ | Advanced examples: custom metrics, registry extension |

### Example Topics:
- Creating sample datasets
- Computing basic metrics
- Using the registry
- Metrics with configuration
- Category-based computation
- Custom metric functions
- Extending the default registry
- Complex multi-column metrics
- Comprehensive data profiling

## 📊 Quality Metrics Reference

### By Category

#### Completeness (2 metrics)
1. `completeness` - Overall data completeness
2. `column_completeness` - Column-specific completeness [config required]

#### Outliers (1 metric)
3. `outlier_rate` - Statistical outlier detection (IQR/Z-score/Modified Z-score)

#### Duplicates (2 metrics)
4. `duplicate_rate` - Duplicate row detection
5. `key_uniqueness` - Key column uniqueness [config required]

#### Consistency (2 metrics)
6. `format_consistency` - Format consistency validation [config required]
7. `referential_integrity` - Foreign key validation [config required]

#### Constraints (2 metrics)
8. `constraint_violation` - Multi-type constraint checking [config required]
9. `type_validity` - Data type validation [config required]

#### Distribution (2 metrics)
10. `distribution_skewness` - Statistical skewness
11. `distribution_kurtosis` - Statistical kurtosis

#### Timeliness (1 metric)
12. `freshness` - Timestamp-based data freshness [config required]

#### Accuracy (1 metric)
13. `value_accuracy` - Ground truth comparison [config required]

#### Composite (1 metric)
14. `overall_quality` - Weighted quality score

## 🚀 Quick Command Reference

```bash
# Verify installation
python verify_installation.py

# Run basic example
python examples/basic_usage.py

# Run advanced example
python examples/custom_metrics.py

# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=inferq --cov-report=html

# Run specific test file
pytest tests/test_quality_metrics.py -v

# Run specific test class
pytest tests/test_quality_metrics.py::TestCompletenessMetrics -v
```

## 📈 Project Statistics

- **Total Lines of Code**: 3,500+
- **Number of Metrics**: 14
- **Number of Categories**: 9
- **Test Classes**: 19
- **Test Cases**: 65+
- **Documentation Pages**: 4 major docs
- **Example Scripts**: 2 comprehensive examples

## 🔗 Navigation Guide

### New Users Start Here:
1. `README.md` - Overview
2. `QUICKSTART.md` - Quick start
3. `examples/basic_usage.py` - Basic example
4. `verify_installation.py` - Verify setup

### Advanced Users:
1. `docs/WP1_DOCUMENTATION.md` - Complete API reference
2. `examples/custom_metrics.py` - Advanced patterns
3. `src/inferq/quality_metrics.py` - Source code
4. `tests/` - Test examples

### Contributors:
1. `WP1_SUMMARY.md` - Implementation details
2. `src/inferq/` - Source code structure
3. `tests/` - Testing patterns
4. `setup.py` - Package configuration

## 📋 File Dependencies

```
README.md
├── QUICKSTART.md (detailed guide)
├── WP1_SUMMARY.md (implementation summary)
└── docs/WP1_DOCUMENTATION.md (API reference)

setup.py
└── requirements.txt (dependencies)

src/inferq/
├── __init__.py (exports)
├── quality_metrics.py (core)
└── utils.py (helpers)

examples/
├── basic_usage.py (imports from src/inferq)
└── custom_metrics.py (imports from src/inferq)

tests/
├── test_quality_metrics.py (imports from src/inferq)
└── test_utils.py (imports from src/inferq)
```

## 🎯 Use Case Index

### I want to...

**...compute basic quality metrics:**
- See: `examples/basic_usage.py`, `QUICKSTART.md`
- Use: `get_default_registry()`, `registry.compute()`

**...create custom metrics:**
- See: `examples/custom_metrics.py`, `docs/WP1_DOCUMENTATION.md`
- Use: `registry.register_function()`, `QualityMetric` class

**...validate constraints:**
- See: `examples/basic_usage.py` (section 4)
- Use: `registry.compute('constraint_violation', df, constraints=[...])`

**...detect outliers:**
- See: `docs/WP1_DOCUMENTATION.md`, `src/inferq/utils.py`
- Use: `registry.compute('outlier_rate', df)` or `detect_outliers_iqr()`

**...profile data quality:**
- See: `examples/custom_metrics.py` (example 4)
- Use: `registry.compute_all(df, metric_configs)`, `compute_data_profile()`

**...extend the framework:**
- See: `examples/custom_metrics.py`, `docs/WP1_DOCUMENTATION.md`
- Use: Custom metric functions + `registry.register_function()`

**...understand the code:**
- See: `src/inferq/quality_metrics.py`, `docs/WP1_DOCUMENTATION.md`
- Start with: Docstrings and inline comments

**...run tests:**
- See: `tests/test_quality_metrics.py`, `tests/test_utils.py`
- Run: `pytest tests/` or `python verify_installation.py`

## 📞 Support & Resources

- **API Documentation**: `docs/WP1_DOCUMENTATION.md`
- **Quick Start**: `QUICKSTART.md`
- **Examples**: `examples/` directory
- **Tests**: `tests/` directory (reference implementations)
- **Implementation Details**: `WP1_SUMMARY.md`

## ✅ Verification Checklist

Before using InferQ, verify:
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Verification passes (`python verify_installation.py`)
- [ ] Basic example works (`python examples/basic_usage.py`)
- [ ] Tests pass (optional: `pytest tests/`)

## 🔄 Version Information

- **Current Version**: 0.1.0
- **Work Package**: WP 1 (Complete)
- **Status**: Production Ready
- **Python Requirement**: >=3.8

---

**Last Updated**: WP 1 Implementation Complete
**Next**: WP 2 - Multi-Target Quality-aware Discretization (MTQD)
