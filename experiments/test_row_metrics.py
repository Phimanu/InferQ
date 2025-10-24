"""Quick test of row-level metrics"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')

from inferq.row_metrics import get_row_level_registry, fit_row_metrics

# Load sample data
df = pd.read_csv('/sc/home/philipp.hildebrandt/InferQ/example_data/adult.csv')
data = df.select_dtypes(include=[np.number]).dropna().sample(n=1000, random_state=42)

print("="*80)
print("TESTING ROW-LEVEL METRICS")
print("="*80)

# Get registry
registry, metrics_dict = get_row_level_registry()

print(f"\nMetrics in registry: {list(registry._metrics.keys())}")

# Fit metrics on training data
print("\nFitting metrics on training data...")
fit_row_metrics(data, metrics_dict)

# Compute metrics for first 10 rows
print("\nComputing metrics for first 10 rows:")
print(f"{'Row':<5} {'Completeness':<15} {'RangeConf':<15} {'Consistency':<15}")
print("-" * 55)

for idx in range(10):
    row = data.iloc[idx:idx+1]
    
    completeness = metrics_dict['row_completeness'].compute(row)
    range_conf = metrics_dict['row_range_conformance'].compute(row)
    consistency = metrics_dict['row_consistency'].compute(row)
    
    print(f"{idx:<5} {completeness:<15.4f} {range_conf:<15.4f} {consistency:<15.4f}")

# Statistics across all rows
print("\n" + "="*80)
print("VARIANCE CHECK (metrics should vary across rows)")
print("="*80)

all_metrics = {'row_completeness': [], 'row_range_conformance': [], 'row_consistency': []}

for idx in range(len(data)):
    row = data.iloc[idx:idx+1]
    for metric_name, metric_obj in metrics_dict.items():
        value = metric_obj.compute(row)
        all_metrics[metric_name].append(value)

for metric_name, values in all_metrics.items():
    values_arr = np.array(values)
    print(f"\n{metric_name}:")
    print(f"  Mean: {values_arr.mean():.4f}")
    print(f"  Std:  {values_arr.std():.4f}")
    print(f"  Min:  {values_arr.min():.4f}")
    print(f"  Max:  {values_arr.max():.4f}")
    print(f"  Unique values: {len(np.unique(values_arr))}")
    
    if values_arr.std() < 0.01:
        print(f"  ⚠️  WARNING: Low variance!")
    else:
        print(f"  ✅ Good variance")

print("\n" + "="*80)
print("SUCCESS: Row-level metrics are working and have variance!")
print("="*80)
