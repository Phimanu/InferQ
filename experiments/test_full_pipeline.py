"""Quick test of full pipeline with row-level metrics"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ')

from experiments.run_experiments import build_index_for_dataset, compute_ground_truth
from sklearn.metrics import r2_score, mean_absolute_error

# Load sample data
df = pd.read_csv('/sc/home/philipp.hildebrandt/InferQ/example_data/adult.csv')
data = df.select_dtypes(include=[np.number]).dropna().sample(n=1000, random_state=42)

print("="*80)
print("TESTING FULL PIPELINE WITH ROW-LEVEL METRICS")
print("="*80)

# Build index
print("\n1. Building index with row-level metrics...")
index, info = build_index_for_dataset(data, budget=30, initial_bins=20, use_row_metrics=True)

print(f"   Features selected: {info['n_features_selected']}")
print(f"   Bins used: {info['n_bins']}")
print(f"   Model test R²: {info['test_r2']:.4f}")

# Get registry
registry = info['registry']
metrics_dict = info['metrics_dict']

# Make predictions
print("\n2. Making predictions on test set (200 rows)...")
test_data = data.sample(n=200, random_state=99)
predictions = index.predict_batch(test_data)

print(f"   Predictions shape: {predictions.shape}")
print(f"   Prediction columns: {list(predictions.columns)}")

# Compute ground truth
print("\n3. Computing ground truth...")
ground_truth = compute_ground_truth(test_data, registry, metrics_dict)

print(f"   Ground truth shape: {ground_truth.shape}")
print(f"   Ground truth columns: {list(ground_truth.columns)}")

# Check variance
print("\n4. Checking ground truth variance...")
for col in ground_truth.columns:
    std = ground_truth[col].std()
    n_unique = ground_truth[col].nunique()
    status = "✅" if std > 0.01 else "⚠️"
    print(f"   {status} {col}: std={std:.4f}, unique={n_unique}")

# Compute accuracy
print("\n5. Computing accuracy...")
common_cols = [col for col in predictions.columns if col in ground_truth.columns]
print(f"   Common columns ({len(common_cols)}): {common_cols}")

pred_filtered = predictions[common_cols]
gt_filtered = ground_truth[common_cols]

r2 = r2_score(gt_filtered, pred_filtered)
mae = mean_absolute_error(gt_filtered, pred_filtered)

print(f"\n   Overall R²: {r2:.4f}")
print(f"   Overall MAE: {mae:.4f}")

print("\n   Per-metric R²:")
for col in common_cols:
    r2_metric = r2_score(gt_filtered[col], pred_filtered[col])
    print(f"      {col}: {r2_metric:.4f}")

print("\n" + "="*80)
if r2 > 0.5:
    print("✅ SUCCESS: Row-level metrics achieve good accuracy (R² > 0.5)!")
elif r2 > 0.3:
    print("⚠️  MODERATE: R² is {:.4f}, better than before but could be improved".format(r2))
else:
    print("❌ ISSUE: R² is still low ({:.4f})".format(r2))
print("="*80)
