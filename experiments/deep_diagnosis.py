"""Deep dive into why R² is stuck at 0.3333"""

import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ')

from inferq.quality_metrics import get_default_registry
from experiments.run_experiments import build_index_for_dataset, compute_ground_truth

# Load data
df = pd.read_csv('/sc/home/philipp.hildebrandt/InferQ/example_data/adult.csv')
data = df.sample(n=2000, random_state=42)
data = data.select_dtypes(include=[np.number]).dropna()

print("="*80)
print("INVESTIGATING WHY R² = 0.3333")
print("="*80)

registry = get_default_registry()

# Build index
print("\n1. Building index...")
index, info = build_index_for_dataset(data, budget=50, initial_bins=100)

print(f"   Features selected: {info['n_features_selected']}")
print(f"   Bins used: {info['n_bins']}")

# Make predictions
print("\n2. Making predictions on test set...")
test_data = data.sample(n=100, random_state=99)
predictions = index.predict_batch(test_data)

print(f"   Predictions shape: {predictions.shape}")
print(f"   Prediction columns: {list(predictions.columns)}")
print(f"\n   First 5 predictions:")
print(predictions.head())

print(f"\n   Prediction statistics:")
print(predictions.describe())

print(f"\n   Unique values per metric:")
for col in predictions.columns:
    n_unique = predictions[col].nunique()
    print(f"   {col}: {n_unique} unique values")

# Compute ground truth
print("\n3. Computing ground truth...")
ground_truth = compute_ground_truth(test_data, registry)

print(f"   Ground truth shape: {ground_truth.shape}")
print(f"   Ground truth columns: {list(ground_truth.columns)}")
print(f"\n   First 5 ground truth values:")
print(ground_truth.head())

print(f"\n   Ground truth statistics:")
print(ground_truth.describe())

print(f"\n   Unique values per metric:")
for col in ground_truth.columns:
    n_unique = ground_truth[col].nunique()
    print(f"   {col}: {n_unique} unique values")

# Compare
print("\n4. Comparing predictions vs ground truth...")
common_cols = [col for col in predictions.columns if col in ground_truth.columns]
print(f"   Common columns ({len(common_cols)}): {common_cols}")

pred_filtered = predictions[common_cols]
gt_filtered = ground_truth[common_cols]

from sklearn.metrics import r2_score, mean_absolute_error

print(f"\n   Overall R²: {r2_score(gt_filtered, pred_filtered):.4f}")
print(f"   Overall MAE: {mean_absolute_error(gt_filtered, pred_filtered):.4f}")

print(f"\n   Per-metric R²:")
for col in common_cols:
    try:
        r2 = r2_score(gt_filtered[[col]], pred_filtered[[col]])
        mae = mean_absolute_error(gt_filtered[[col]], pred_filtered[[col]])
        var_gt = gt_filtered[col].var()
        var_pred = pred_filtered[col].var()
        print(f"   {col:25s}: R²={r2:7.4f}, MAE={mae:7.4f}, Var(GT)={var_gt:8.4f}, Var(Pred)={var_pred:8.4f}")
    except Exception as e:
        print(f"   {col:25s}: ERROR - {e}")

# Check if model is actually working
print("\n5. Checking model internals...")
print(f"   Model test R²: {index.metrics.test_r2:.4f}")
print(f"   Model test MSE: {index.metrics.test_mse:.4f}")
print(f"   Model trained on {index.metrics.n_train} samples")

# Check training data quality
print("\n6. Diagnosing the issue...")

# Theory 1: Are all metrics constant?
print("\n   Theory 1: Are metrics constant in ground truth?")
for col in gt_filtered.columns:
    if gt_filtered[col].std() < 0.01:
        print(f"   ⚠️  {col} is nearly constant! std={gt_filtered[col].std():.6f}")
    else:
        print(f"   ✓  {col} varies. std={gt_filtered[col].std():.6f}")

# Theory 2: Are predictions constant?
print("\n   Theory 2: Are predictions constant?")
for col in pred_filtered.columns:
    if pred_filtered[col].std() < 0.01:
        print(f"   ⚠️  {col} predictions are nearly constant! std={pred_filtered[col].std():.6f}")
    else:
        print(f"   ✓  {col} predictions vary. std={pred_filtered[col].std():.6f}")

# Theory 3: Is the model outputting the same values?
print("\n   Theory 3: Are all predictions identical?")
print(f"   Number of unique prediction vectors: {pred_filtered.drop_duplicates().shape[0]}/{len(pred_filtered)}")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
