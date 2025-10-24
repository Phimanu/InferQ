"""
WP 3 Example: Multi-Target Quality-aware Discretization (MTQD)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from inferq import (
    get_default_registry,
    InitialPartitioner,
    MTQD,
    summarize_partitions
)


def main():
    print("=" * 70)
    print("WP 3: Multi-Target Quality-aware Discretization (MTQD)")
    print("=" * 70)
    
    # Create sample dataset with quality variations
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'salary': np.random.randint(30000, 120000, n_samples),
        'experience': np.random.randint(0, 40, n_samples),
    })
    
    # Add quality issues in certain ranges
    # Young employees: more missing salary data
    young_mask = df['age'] < 25
    df.loc[young_mask & (np.random.rand(n_samples) < 0.3), 'salary'] = None
    
    # Senior employees: more complete data
    senior_mask = df['age'] > 60
    # (already complete)
    
    print(f"\n1. Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Overall completeness: {(1 - df.isna().sum().sum() / df.size):.3f}")
    
    # Step 1: Create initial fine-grained partitions
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=30)
    
    print(f"\n2. Creating initial partitions (k=30 bins)...")
    initial_partitions = partitioner.partition_all_attributes(df)
    
    print("\n   Initial Partitions:")
    initial_summary = summarize_partitions(initial_partitions)
    print(initial_summary.to_string(index=False))
    
    # Step 2: Apply MTQD with different thresholds
    print("\n3. Applying MTQD merging...")
    
    # Conservative merging
    print("\n   a) Conservative (threshold=0.005):")
    mtqd_conservative = MTQD(registry, merge_threshold=0.005)
    conservative_partitions = mtqd_conservative.discretize_all(initial_partitions, df)
    
    for attr in ['age', 'salary', 'experience']:
        initial = len(initial_partitions[attr].bins)
        optimized = len(conservative_partitions[attr].bins)
        reduction = (initial - optimized) / initial * 100
        print(f"      {attr}: {initial} → {optimized} bins ({reduction:.1f}% reduction)")
    
    # Moderate merging
    print("\n   b) Moderate (threshold=0.01):")
    mtqd_moderate = MTQD(registry, merge_threshold=0.01)
    moderate_partitions = mtqd_moderate.discretize_all(initial_partitions, df)
    
    for attr in ['age', 'salary', 'experience']:
        initial = len(initial_partitions[attr].bins)
        optimized = len(moderate_partitions[attr].bins)
        reduction = (initial - optimized) / initial * 100
        print(f"      {attr}: {initial} → {optimized} bins ({reduction:.1f}% reduction)")
    
    # Aggressive merging
    print("\n   c) Aggressive (threshold=0.05):")
    mtqd_aggressive = MTQD(registry, merge_threshold=0.05)
    aggressive_partitions = mtqd_aggressive.discretize_all(initial_partitions, df)
    
    for attr in ['age', 'salary', 'experience']:
        initial = len(initial_partitions[attr].bins)
        optimized = len(aggressive_partitions[attr].bins)
        reduction = (initial - optimized) / initial * 100
        print(f"      {attr}: {initial} → {optimized} bins ({reduction:.1f}% reduction)")
    
    # Step 3: Analyze quality preservation
    print("\n4. Quality Analysis - 'age' attribute (moderate threshold):")
    print("-" * 70)
    
    age_partition = moderate_partitions['age']
    
    print(f"   Final bins: {len(age_partition.bins)}")
    print(f"   Boundaries: [{age_partition.bin_boundaries[0]:.0f}, ..., "
          f"{age_partition.bin_boundaries[-1]:.0f}]")
    
    print("\n   Sample bins with quality vectors:")
    for i, bin in enumerate(age_partition.bins[:5]):
        completeness = bin.quality_vector.get('completeness', 0)
        outlier_rate = bin.quality_vector.get('outlier_rate', 0)
        print(f"   Bin {i}: [{bin.lower_bound:.0f}, {bin.upper_bound:.0f}), "
              f"n={len(bin.indices)}, "
              f"compl={completeness:.3f}, "
              f"outliers={outlier_rate:.3f}")
    
    # Step 4: Compare metric weights impact
    print("\n5. Impact of Metric Weights:")
    print("-" * 70)
    
    # Equal weights (default)
    mtqd_equal = MTQD(registry, merge_threshold=0.01)
    equal_result = mtqd_equal.discretize(initial_partitions['age'], df)
    
    # Prioritize completeness
    mtqd_completeness = MTQD(
        registry, 
        metric_weights={'completeness': 0.8, 'outlier_rate': 0.2},
        merge_threshold=0.01
    )
    completeness_result = mtqd_completeness.discretize(initial_partitions['age'], df)
    
    print(f"   Equal weights: {len(equal_result.bins)} bins")
    print(f"   Prioritize completeness: {len(completeness_result.bins)} bins")
    
    # Step 5: Final summary
    print("\n6. Final Summary (moderate threshold):")
    print("-" * 70)
    
    optimized_summary = summarize_partitions(moderate_partitions)
    print(optimized_summary.to_string(index=False))
    
    print("\n   Compression ratios:")
    for attr in initial_partitions.keys():
        initial_bins = len(initial_partitions[attr].bins)
        final_bins = len(moderate_partitions[attr].bins)
        ratio = final_bins / initial_bins
        print(f"      {attr}: {ratio:.2f}x ({initial_bins} → {final_bins} bins)")
    
    print("\n" + "=" * 70)
    print("✅ WP 3 Example Complete")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- MTQD successfully merges bins with similar quality characteristics")
    print("- Merge threshold controls compression vs. quality preservation")
    print("- Metric weights allow domain-specific prioritization")
    print("- Final partitions maintain quality annotations for downstream use")


if __name__ == '__main__':
    main()
