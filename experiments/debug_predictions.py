"""Debug why predictions are not matching ground truth"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ')

from experiments.run_experiments import build_index_for_dataset, compute_ground_truth

# Load sample data
df = pd.read_csv('/sc/home/philipp.hildebrandt/InferQ/example_data/adult.csv')
data = df.select_dtypes(include=[np.number]).dropna().sample(n=1000, random_state=42)

# Build index
index, info = build_index_for_dataset(data, budget=30, initial_bins=20, use_row_metrics=True)
registry = info['registry']
metrics_dict = info['metrics_dict']

# Test on same data used for training
print("Testing on TRAINING data:")
pred_train = index.predict_batch(data[:100])
gt_train = compute_ground_truth(data[:100], registry, metrics_dict)

print("\nFirst 5 predictions:")
print(pred_train.head())

print("\nFirst 5 ground truth:")
print(gt_train.head())

print("\nPrediction statistics:")
print(pred_train.describe())

print("\nGround truth statistics:")
print(gt_train.describe())

# Check if predictions are all the same
for col in pred_train.columns:
    n_unique_pred = pred_train[col].nunique()
    n_unique_gt = gt_train[col].nunique()
    print(f"\n{col}:")
    print(f"  Predictions: {n_unique_pred} unique values, range=[{pred_train[col].min():.4f}, {pred_train[col].max():.4f}]")
    print(f"  Ground truth: {n_unique_gt} unique values, range=[{gt_train[col].min():.4f}, {gt_train[col].max():.4f}]")
    
    if n_unique_pred == 1:
        print(f"  ⚠️  Predictions are constant!")
