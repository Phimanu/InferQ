"""
Example: WP 7 - Bin Dictionary and Training Data Generation

Demonstrates complete pipeline from data loading to training data generation
using the Adult dataset from UCI ML Repository.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')

from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.quality_assessment import AdaptiveMTQD
from inferq.feature_scoring import FeatureScorer, get_default_quality_thresholds
from inferq.feature_selection import GreedyFeatureSelector, summarize_selection
from inferq.bin_dictionary import (
    BinDictionary,
    generate_training_data,
    save_training_data
)


def main():
    print("="*70)
    print("WP 7: Bin Dictionary and Training Data Generation")
    print("="*70)
    
    # Load Adult dataset
    print(f"\n1. Loading Adult dataset...")
    print("="*70)
    
    try:
        data = pd.read_csv('/sc/home/philipp.hildebrandt/InferQ/example_data/adult.csv')
        print(f"   Loaded: {len(data)} rows, {len(data.columns)} columns")
    except FileNotFoundError:
        print("   Adult dataset not found, using synthetic data...")
        # Fallback to synthetic data
        np.random.seed(42)
        n = 1000
        data = pd.DataFrame({
            'age': np.random.randint(18, 80, n),
            'education_num': np.random.randint(1, 17, n),
            'hours_per_week': np.random.randint(1, 100, n),
            'capital_gain': np.random.randint(0, 100000, n),
            'fnlwgt': np.random.randint(10000, 1500000, n)
        })
    
    # Select numeric columns for analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith('>')][:6]  # Limit to 6 features
    
    data_numeric = data[numeric_cols].copy()
    
    # Add quality issues
    n = len(data_numeric)
    for col in numeric_cols[:3]:
        missing_mask = np.random.random(n) < 0.1
        data_numeric.loc[missing_mask, col] = np.nan
    
    print(f"\n   Analyzing {len(numeric_cols)} numeric features:")
    for col in numeric_cols:
        completeness = 1 - data_numeric[col].isna().mean()
        print(f"     {col}: {completeness:.1%} complete")
    
    # Stage 1: Discretization
    print(f"\n2. Stage 1: Discretization")
    print("="*70)
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=30)
    
    initial_partitions = {}
    for attr in numeric_cols:
        initial_partitions[attr] = partitioner.partition_attribute(data_numeric, attr)
    
    # Apply MTQD
    adaptive = AdaptiveMTQD(registry=registry, max_iterations=3)
    final_partitions = {}
    
    print(f"   {'Attribute':<20} {'Initial':>8} {'Final':>8} {'Reduction':>10}")
    print("   " + "-"*46)
    
    for attr, initial in initial_partitions.items():
        final, quality = adaptive.discretize_with_assessment(initial, data_numeric)
        final_partitions[attr] = final
        reduction = (1 - len(final.bins) / len(initial.bins)) * 100
        print(f"   {attr:<20} {len(initial.bins):>8} {len(final.bins):>8} {reduction:>9.1f}%")
    
    # Stage 2: Feature Selection
    print(f"\n3. Stage 2: Feature Selection")
    print("="*70)
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 0.4, 'outlier_rate': 0.3, 'duplicate_rate': 0.3},
        quality_thresholds=get_default_quality_thresholds(),
        alpha=0.7,
        beta=0.3
    )
    
    ranked_features = scorer.score_features(final_partitions, data_numeric)
    
    print(f"\n   Feature Ranking:")
    for rank, feature in enumerate(ranked_features[:8], 1):
        n_bins = len(final_partitions[feature.attribute].bins)
        print(f"     {rank}. {feature.attribute:<20} importance={feature.importance:.4f}, bins={n_bins}")
    
    # Select with budget
    budget = 80
    selector = GreedyFeatureSelector(budget=budget, registry=registry, min_bins_per_feature=5)
    result = selector.select_features(ranked_features, final_partitions, data_numeric)
    
    print(f"\n   {summarize_selection(result)}")
    
    # Stage 3: Bin Dictionary Creation
    print(f"\n4. Stage 3: Bin Dictionary Creation")
    print("="*70)
    
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    
    print(f"\n   {bin_dict.summary()}")
    
    # Test bin vector lookup
    print(f"\n5. Bin Vector Lookups")
    print("="*70)
    
    sample_rows = data_numeric.head(5)
    
    print(f"\n   Sample data rows and their bin vectors:")
    print(f"   {'Row':>4} {' | '.join(f'{attr[:10]:>10}' for attr in bin_dict.feature_order)} | Bin Vector")
    print("   " + "-" * (5 + 13 * len(bin_dict.feature_order) + 15))
    
    for i, (idx, row) in enumerate(sample_rows.iterrows()):
        bin_vector = bin_dict.get_bin_vector(row)
        value_str = ' | '.join(f'{row.get(attr, np.nan):>10.1f}' for attr in bin_dict.feature_order)
        bin_str = str(bin_vector).replace('\n', '')
        print(f"   {i:>4} {value_str} | {bin_str}")
    
    # Quality vector retrieval
    print(f"\n6. Quality Vector Retrieval")
    print("="*70)
    
    sample_row = data_numeric.iloc[0]
    bin_vector = bin_dict.get_bin_vector(sample_row)
    quality_vector = bin_dict.get_quality_vector(bin_vector)
    
    print(f"\n   Row: {dict((k, f'{v:.1f}' if not pd.isna(v) else 'NaN') for k, v in sample_row.items())}")
    print(f"   Bin Vector: {bin_vector}")
    print(f"\n   Quality Metrics:")
    for metric, value in quality_vector.items():
        print(f"     {metric:<25} {value:.4f}")
    
    # Training Data Generation
    print(f"\n7. Training Data Generation")
    print("="*70)
    
    training_data = generate_training_data(data_numeric, bin_dict)
    
    print(f"\n   Training Data Shape:")
    print(f"     X (bin vectors):     {training_data.X.shape}")
    print(f"     Y (quality vectors): {training_data.Y.shape}")
    print(f"\n   Dataset Info:")
    print(f"     Samples:  {training_data.n_samples}")
    print(f"     Features: {training_data.n_features} {training_data.feature_names}")
    print(f"     Metrics:  {training_data.n_metrics} {training_data.metric_names}")
    
    # Show statistics
    print(f"\n   X (Bin IDs) Statistics:")
    print(f"     Mean:  {training_data.X.mean(axis=0)}")
    print(f"     Std:   {training_data.X.std(axis=0)}")
    print(f"     Min:   {training_data.X.min(axis=0)}")
    print(f"     Max:   {training_data.X.max(axis=0)}")
    
    print(f"\n   Y (Quality) Statistics:")
    for i, metric in enumerate(training_data.metric_names):
        mean_val = training_data.Y[:, i].mean()
        std_val = training_data.Y[:, i].std()
        print(f"     {metric:<25} mean={mean_val:.4f}, std={std_val:.4f}")
    
    # Sample training instances
    print(f"\n8. Sample Training Instances")
    print("="*70)
    
    print(f"\n   {'X (Bin Vector)':<30} | Y (Quality Vector - first 3 metrics)")
    print("   " + "-" * 75)
    
    for i in range(min(10, len(training_data.X))):
        x_str = str(training_data.X[i])
        y_str = ', '.join(f'{v:.3f}' for v in training_data.Y[i, :3])
        print(f"   {x_str:<30} | [{y_str}]")
    
    # Save training data
    print(f"\n9. Saving Training Data")
    print("="*70)
    
    output_path = "/tmp/inferq_training_data"
    save_training_data(training_data, output_path)
    
    print(f"   Saved to: {output_path}.npz")
    print(f"   File size: {np.load(output_path + '.npz')['X'].nbytes + np.load(output_path + '.npz')['Y'].nbytes} bytes")
    
    # Data distribution analysis
    print(f"\n10. Data Distribution Analysis")
    print("="*70)
    
    print(f"\n   Bin assignment distribution:")
    for feat_idx, feat_name in enumerate(training_data.feature_names):
        bin_ids = training_data.X[:, feat_idx]
        unique, counts = np.unique(bin_ids, return_counts=True)
        print(f"\n     {feat_name}:")
        print(f"       Unique bins used: {len(unique)}")
        print(f"       Most common: Bin {unique[np.argmax(counts)]} ({counts.max()} samples)")
        print(f"       Least common: Bin {unique[np.argmin(counts)]} ({counts.min()} samples)")
    
    print("\n" + "="*70)
    print("✅ WP 7 Example Complete")
    print("="*70)
    
    print("\nKey Takeaways:")
    print("- BinDictionary enables O(log n) bin lookups via binary search")
    print("- get_bin_vector() maps data rows to bin assignment vectors")
    print("- Training data X = bin vectors, Y = quality vectors")
    print("- Ready for learned index model training (next stage)")
    print(f"- Training data: {training_data.n_samples} samples × {training_data.n_features} features → {training_data.n_metrics} quality metrics")


if __name__ == '__main__':
    main()
