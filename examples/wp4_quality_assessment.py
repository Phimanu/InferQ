"""
Example: WP 4 - Discretization Quality Assessment

Demonstrates how to use Homogeneity (H) and Separation (S) scores
to validate and adaptively adjust discretization quality.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')

from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.discretization import MTQD
from inferq.quality_assessment import (
    compute_homogeneity,
    compute_separation,
    AdaptiveMTQD
)


def main():
    print("="*70)
    print("WP 4: Discretization Quality Assessment")
    print("="*70)
    
    # Create dataset with distinct quality patterns
    np.random.seed(42)
    n = 999  # Divisible by 3
    
    # Three age groups with different completeness
    age_young = np.random.normal(25, 5, n//3)
    age_middle = np.random.normal(45, 5, n//3)
    age_senior = np.random.normal(65, 5, n//3)
    
    data = pd.DataFrame({
        'age': np.concatenate([age_young, age_middle, age_senior]),
        'salary': np.random.randint(30000, 150000, n),
        'experience': np.random.randint(0, 40, n)
    })
    
    # Add missing values with different patterns per group
    for i in range(n//3):
        if np.random.random() < 0.3:  # Young: 30% missing
            data.loc[i, 'salary'] = np.nan
    for i in range(n//3, 2*n//3):
        if np.random.random() < 0.1:  # Middle: 10% missing
            data.loc[i, 'salary'] = np.nan
    for i in range(2*n//3, n):
        if np.random.random() < 0.05:  # Senior: 5% missing
            data.loc[i, 'salary'] = np.nan
    
    overall_complete = 1 - data['salary'].isna().mean()
    print(f"\n1. Dataset: {n} rows, {len(data.columns)} columns")
    print(f"   Overall completeness: {overall_complete:.3f}")
    
    # Initialize
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=30)
    
    print(f"\n2. Creating initial partition (k=30 bins)...")
    initial = partitioner.partition_attribute(data, 'age')
    print(f"   Initial bins: {len(initial.bins)}")
    
    # Manual MTQD with different thresholds
    print(f"\n3. Manual MTQD with quality assessment:")
    print("="*70)
    
    thresholds = [0.001, 0.005, 0.01, 0.05]
    results = []
    
    for threshold in thresholds:
        mtqd = MTQD(merge_threshold=threshold, registry=registry)
        partition = mtqd.discretize(initial, data)
        
        H = compute_homogeneity(partition, data)
        S = compute_separation(partition)
        
        results.append({
            'threshold': threshold,
            'n_bins': len(partition.bins),
            'H': H,
            'S': S,
            'score': S - H  # Combined score (higher is better)
        })
        
        print(f"   Threshold={threshold:.4f}: {len(partition.bins):2d} bins, "
              f"H={H:.4f}, S={S:.4f}, Score={S-H:.4f}")
    
    # Find best manual configuration
    best = max(results, key=lambda x: x['score'])
    print(f"\n   Best manual config: threshold={best['threshold']:.4f}, "
          f"{best['n_bins']} bins, Score={best['score']:.4f}")
    
    # Adaptive MTQD
    print(f"\n4. Adaptive MTQD (automatic threshold tuning):")
    print("="*70)
    
    adaptive = AdaptiveMTQD(
        registry=registry,
        max_homogeneity=0.01,
        min_separation=0.005,
        max_iterations=10
    )
    
    final_partition, quality = adaptive.discretize_with_assessment(
        initial, data, initial_threshold=0.01
    )
    
    print(f"   Final: {quality.n_bins} bins")
    print(f"   Homogeneity: {quality.homogeneity:.4f} (target: ≤ 0.01)")
    print(f"   Separation: {quality.separation:.4f} (target: ≥ 0.005)")
    print(f"   Threshold: {quality.threshold:.4f}")
    print(f"   Acceptable: {quality.is_acceptable(0.01, 0.005)}")
    
    # Show sample bins from final partition
    print(f"\n5. Sample bins from final partition:")
    print("="*70)
    
    for i, bin in enumerate(final_partition.bins[:5]):
        compl = bin.quality_vector.get('completeness', 0)
        outliers = bin.quality_vector.get('outlier_rate', 0)
        print(f"   Bin {i}: [{bin.lower_bound:.1f}, {bin.upper_bound:.1f}), "
              f"n={len(bin.indices)}, compl={compl:.3f}, outliers={outliers:.3f}")
    
    if len(final_partition.bins) > 5:
        print(f"   ... ({len(final_partition.bins) - 5} more bins)")
    
    # Multi-attribute analysis
    print(f"\n6. Multi-attribute quality assessment:")
    print("="*70)
    
    attr_results = {}
    for attr in ['age', 'salary', 'experience']:
        initial_attr = partitioner.partition_attribute(data, attr)
        final_attr, quality_attr = adaptive.discretize_with_assessment(
            initial_attr, data
        )
        attr_results[attr] = quality_attr
    
    print(f"   {'Attribute':<12} {'Bins':>5} {'H':>8} {'S':>8} {'Score':>8}")
    print("   " + "-"*43)
    for attr, q in attr_results.items():
        print(f"   {attr:<12} {q.n_bins:>5} {q.homogeneity:>8.4f} "
              f"{q.separation:>8.4f} {q.separation - q.homogeneity:>8.4f}")
    
    print("\n" + "="*70)
    print("✅ WP 4 Example Complete")
    print("="*70)
    
    print("\nKey Takeaways:")
    print("- H (Homogeneity): Lower = bins are internally consistent")
    print("- S (Separation): Higher = bins differ meaningfully")
    print("- Adaptive MTQD automatically finds good threshold")
    print("- Quality criteria ensure effective discretization")


if __name__ == '__main__':
    main()
