# InferQ Scripts

Command-line tools for building and using quality-aware learned indexes.

## Scripts

### build_index.py

Build a quality index from historical data (offline construction).

**Usage:**
```bash
python scripts/build_index.py data.csv -o output/index -b 100
```

**Arguments:**
- `data`: Path to input CSV file
- `-o, --output`: Output path for index (default: `quality_index`)
- `-b, --budget`: Maximum total bins (default: 100)
- `-n, --n-estimators`: Number of trees in Random Forest (default: 100)
- `--max-depth`: Maximum tree depth (default: 20)
- `--initial-bins`: Initial bins for discretization (default: 30)
- `--test-size`: Fraction for testing (default: 0.2)
- `--quiet`: Suppress progress output

**Outputs:**
- `{output}.pkl`: Trained index
- `{output}_metadata.json`: Build metadata
- `{output}_training.npz`: Training data

**Example:**
```bash
# Build index with 60 bins budget
python scripts/build_index.py example_data/adult.csv \
    -o output/adult_index \
    -b 60 \
    --initial-bins 20

# Output:
# ✅ Index Construction Complete
# Model accuracy (R²): 0.9808
# Selected: 4/6 features
# Bins: 60/60 (100.0%)
```

### online_monitor.py

Monitor streaming data using a pre-trained index (online monitoring).

**Usage:**
```bash
python scripts/online_monitor.py index.pkl data.csv -b 200
```

**Arguments:**
- `index`: Path to trained index (.pkl file)
- `data`: Path to data file to monitor (CSV)
- `-b, --batch-size`: Tuples per batch (default: 100)
- `-d, --delay`: Delay between batches in seconds (default: 0.0)
- `-t, --thresholds`: Path to JSON file with custom alert thresholds
- `-l, --log`: Path to log file for alerts
- `-n, --max-tuples`: Maximum tuples to process (default: all)
- `--quiet`: Suppress progress output

**Alert Thresholds (default):**
```json
{
  "completeness": {"warning": 0.95, "critical": 0.90},
  "outlier_rate": {"warning": 0.05, "critical": 0.10},
  "duplicate_rate": {"warning": 0.05, "critical": 0.10},
  "consistency_score": {"warning": 0.95, "critical": 0.90},
  "overall_quality": {"warning": 0.90, "critical": 0.80}
}
```

**Example:**
```bash
# Monitor 1000 tuples in batches of 200
python scripts/online_monitor.py output/adult_index.pkl \
    example_data/adult.csv \
    -b 200 \
    -n 1000 \
    -l output/alerts.csv

# Output:
# ✅ Monitoring Complete
# Total tuples: 1000
# Throughput: 23 tuples/sec
# Total alerts: 952 (95.2%)
#   - Critical: 1
#   - Warning: 951
```

## Complete Workflow

### 1. Offline Construction

Build index from historical data:

```bash
# Prepare data
cat > data.csv << EOF
age,income,credit_score
25,50000,720
30,60000,680
...
EOF

# Build index
python scripts/build_index.py data.csv \
    -o models/my_index \
    -b 100 \
    --n-estimators 100
```

### 2. Online Monitoring

Use index to monitor new data:

```bash
# Monitor streaming data
python scripts/online_monitor.py models/my_index.pkl \
    new_data.csv \
    -b 100 \
    -l logs/alerts.csv
```

### 3. Custom Thresholds

Create custom alert thresholds:

```bash
# Create thresholds file
cat > thresholds.json << EOF
{
  "completeness": {"warning": 0.98, "critical": 0.95},
  "outlier_rate": {"warning": 0.03, "critical": 0.07}
}
EOF

# Use custom thresholds
python scripts/online_monitor.py models/my_index.pkl \
    new_data.csv \
    -t thresholds.json \
    -l logs/alerts.csv
```

## Output Files

### Index Files
- `{name}.pkl`: Serialized QualityIndex (model + bin dictionary)
- `{name}_metadata.json`: Build statistics and configuration
- `{name}_training.npz`: Training data (X and Y arrays)

### Alert Logs
CSV format with columns:
- `timestamp`: Unix timestamp
- `tuple_id`: Row identifier
- `metric`: Quality metric name
- `predicted_value`: Predicted quality value
- `threshold`: Alert threshold
- `severity`: "warning" or "critical"

## Performance

Typical throughput on modern hardware:
- **Index construction**: ~1000 tuples/sec (depends on features/bins)
- **Online monitoring**: ~100-1000 tuples/sec (depends on model size)

## Environment Setup

These scripts require the InferQ package to be installed or available via PYTHONPATH:

```bash
# Option 1: Add to PYTHONPATH
export PYTHONPATH=/path/to/InferQ/src:$PYTHONPATH

# Option 2: Install package (if setup.py available)
pip install -e .

# Then run scripts
python scripts/build_index.py data.csv
```

## Examples

See `examples/wp9_complete_pipeline.py` for a complete demonstration of both offline construction and online monitoring.
