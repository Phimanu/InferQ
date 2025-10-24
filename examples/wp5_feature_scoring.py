"""
Example: WP 5 - Feature Importance Scoring

Demonstrates quality-aware feature importance scoring using
IG_multi (Information Gain) and QDP (Quality Detection Power).
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')

from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.discretization import MTQD
from inferq.quality_assessment import AdaptiveMTQD
from inferq.feature_scoring import (
    FeatureScorer,
    get_default_quality_thresholds
)


def main():
    print("="*70)
    print("WP 5: Feature Importance Scoring")
    print("="*70)
    
    # Create dataset with varying quality patterns across attributes
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'customer_age': np.random.randint(18, 80, n),
        'account_balance': np.random.uniform(0, 100000, n),
        'transaction_count': np.random.randint(0, 500, n),
        'account_tenure_days': np.random.randint(1, 3650, n),
        'credit_score': np.random.randint(300, 850, n)
    })
    
    # Introduce quality issues with different patterns per attribute
    
    # customer_age: missing values correlated with age
    for i in range(n):
        if data.loc[i, 'customer_age'] < 25:
            if np.random.random() < 0.35:  # Young customers: 35% missing
                data.loc[i, 'account_balance'] = np.nan
    
    # transaction_count: outliers in high values
    outlier_mask = data['transaction_count'] > 450
    data.loc[outlier_mask, 'transaction_count'] *= 10
    
    # account_tenure_days: completeness issues
    missing_mask = np.random.random(n) < 0.15
    data.loc[missing_mask, 'account_tenure_days'] = np.nan
    
    # credit_score: generally clean
    missing_mask = np.random.random(n) < 0.02
    data.loc[missing_mask, 'credit_score'] = np.nan
    
    print(f"\n1. Dataset: {n} rows, {len(data.columns)} attributes")
    print(f"\n   Quality Summary:")
    for col in data.columns:
        completeness = 1 - data[col].isna().mean()
        print(f"     {col}: {completeness:.1%} complete")
    
    # Stage 1: Discretization for all attributes
    print(f"\n2. Stage 1: Discretizing all attributes...")
    print("="*70)
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=30)
    
    # Create initial partitions
    initial_partitions = {}
    for attr in data.columns:
        initial_partitions[attr] = partitioner.partition_attribute(data, attr)
    
    # Apply adaptive MTQD
    adaptive = AdaptiveMTQD(registry=registry, max_iterations=5)
    final_partitions = {}
    
    for attr, initial_partition in initial_partitions.items():
        final_partition, quality = adaptive.discretize_with_assessment(
            initial_partition, data
        )
        final_partitions[attr] = final_partition
    
    print(f"   Discretization complete:")
    for attr, partition in final_partitions.items():
        print(f"     {attr}: {len(partition.bins)} bins")
    
    # Stage 2: Feature Importance Scoring
    print(f"\n3. Stage 2: Computing feature importance...")
    print("="*70)
    
    # Configure scorer
    metric_weights = {
        'completeness': 0.3,
        'outlier_rate': 0.3,
        'duplicate_rate': 0.2,
        'consistency_score': 0.2
    }
    
    quality_thresholds = get_default_quality_thresholds()
    
    scorer = FeatureScorer(
        metric_weights=metric_weights,
        quality_thresholds=quality_thresholds,
        alpha=0.7,  # Weight for IG_multi
        beta=0.3    # Weight for QDP
    )
    
    # Score all features
    scores = scorer.score_features(final_partitions, data)
    
    print(f"\n   Feature Importance Ranking:")
    print(f"   {'Rank':<6} {'Attribute':<25} {'Importance':>11} {'IG_multi':>11} {'QDP':>8} {'Bins':>6} {'Issues':>7}")
    print("   " + "-"*75)
    
    for rank, score in enumerate(scores, 1):
        print(f"   {rank:<6} {score.attribute:<25} {score.importance:>11.4f} "
              f"{score.ig_multi:>11.4f} {score.qdp:>8.4f} "
              f"{score.n_bins:>6} {score.n_issue_bins:>7}")
    
    # Analyze top features
    print(f"\n4. Top Feature Analysis:")
    print("="*70)
    
    top_feature = scores[0]
    print(f"\n   Most Important: {top_feature.attribute}")
    print(f"   - Importance Score: {top_feature.importance:.4f}")
    print(f"   - IG_multi: {top_feature.ig_multi:.4f} (predictive power)")
    print(f"   - QDP: {top_feature.qdp:.4f} (quality detection power)")
    print(f"   - Bins: {top_feature.n_bins} total, {top_feature.n_issue_bins} with issues")
    
    # Show sample bins from top feature
    top_partition = final_partitions[top_feature.attribute]
    print(f"\n   Sample bins from {top_feature.attribute}:")
    for i, bin in enumerate(top_partition.bins[:5]):
        compl = bin.quality_vector.get('completeness', 0)
        outliers = bin.quality_vector.get('outlier_rate', 0)
        has_issue = '⚠' if compl < 0.95 or outliers > 0.05 else '✓'
        print(f"     {has_issue} Bin {i}: [{bin.lower_bound:.1f}, {bin.upper_bound:.1f}), "
              f"n={len(bin.indices)}, compl={compl:.3f}, outliers={outliers:.3f}")
    
    # Feature selection
    print(f"\n5. Feature Selection:")
    print("="*70)
    
    for k in [3, 5]:
        selected = scorer.select_top_features(final_partitions, data, k=k)
        print(f"   Top-{k} features: {', '.join(selected)}")
    
    # Impact of alpha/beta weights
    print(f"\n6. Impact of α (IG) and β (QDP) weights:")
    print("="*70)
    
    weight_configs = [
        (1.0, 0.0, "IG-only"),
        (0.7, 0.3, "Balanced"),
        (0.5, 0.5, "Equal"),
        (0.3, 0.7, "QDP-focus"),
        (0.0, 1.0, "QDP-only")
    ]
    
    print(f"   {'Config':<12} {'α':>5} {'β':>5} {'Top-3 Features':<60}")
    print("   " + "-"*82)
    
    for alpha, beta, name in weight_configs:
        scorer_test = FeatureScorer(
            metric_weights=metric_weights,
            quality_thresholds=quality_thresholds,
            alpha=alpha,
            beta=beta
        )
        top_3 = scorer_test.select_top_features(final_partitions, data, k=3)
        print(f"   {name:<12} {alpha:>5.1f} {beta:>5.1f} {', '.join(top_3):<60}")
    
    # Metric weight impact
    print(f"\n7. Impact of quality metric weights:")
    print("="*70)
    
    metric_configs = [
        ({'completeness': 1.0}, "Completeness-only"),
        ({'outlier_rate': 1.0}, "Outlier-only"),
        ({'completeness': 0.5, 'outlier_rate': 0.5}, "Compl+Outlier"),
        (metric_weights, "All metrics")
    ]
    
    print(f"   {'Config':<20} {'Top Feature':<25} {'Importance':>11}")
    print("   " + "-"*56)
    
    for weights, name in metric_configs:
        scorer_test = FeatureScorer(
            metric_weights=weights,
            quality_thresholds=quality_thresholds,
            alpha=0.7,
            beta=0.3
        )
        scores_test = scorer_test.score_features(final_partitions, data)
        top = scores_test[0]
        print(f"   {name:<20} {top.attribute:<25} {top.importance:>11.4f}")
    
    print("\n" + "="*70)
    print("✅ WP 5 Example Complete")
    print("="*70)
    
    print("\nKey Takeaways:")
    print("- IG_multi: Measures how much discretization reduces uncertainty")
    print("- QDP: Measures how well attribute isolates quality issues")
    print("- Combined score balances predictive power and practical utility")
    print("- α and β weights allow domain-specific prioritization")
    print("- Feature selection identifies most important attributes for indexing")


if __name__ == '__main__':
    main()
