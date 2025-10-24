"""
Example: WP 6 - Greedy Feature Selection

Demonstrates adaptive greedy feature selection with budget constraints
and granularity adjustment.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')

from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.quality_assessment import AdaptiveMTQD
from inferq.feature_scoring import FeatureScorer, get_default_quality_thresholds
from inferq.feature_selection import (
    GreedyFeatureSelector,
    extract_discretization_boundaries,
    summarize_selection
)


def main():
    print("="*70)
    print("WP 6: Greedy Feature Selection")
    print("="*70)
    
    # Create realistic dataset with 8 attributes
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'customer_id': np.arange(n),
        'age': np.random.randint(18, 80, n),
        'account_balance': np.random.uniform(0, 100000, n),
        'transaction_count': np.random.randint(0, 500, n),
        'account_tenure': np.random.randint(1, 3650, n),
        'credit_score': np.random.randint(300, 850, n),
        'monthly_spending': np.random.uniform(0, 10000, n),
        'loan_amount': np.random.uniform(0, 50000, n),
        'num_products': np.random.randint(1, 6, n)
    })
    
    # Add quality issues
    for i in range(n):
        if data.loc[i, 'age'] < 25:
            if np.random.random() < 0.3:
                data.loc[i, 'account_balance'] = np.nan
        if data.loc[i, 'credit_score'] < 500:
            if np.random.random() < 0.2:
                data.loc[i, 'loan_amount'] = np.nan
    
    # Drop ID column for analysis
    analysis_columns = [c for c in data.columns if c != 'customer_id']
    
    print(f"\n1. Dataset: {n} rows, {len(analysis_columns)} attributes")
    
    # Stage 1: Discretization
    print(f"\n2. Stage 1: Discretizing attributes...")
    print("="*70)
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=30)
    
    initial_partitions = {}
    for attr in analysis_columns:
        initial_partitions[attr] = partitioner.partition_attribute(data, attr)
    
    # Apply adaptive MTQD
    adaptive = AdaptiveMTQD(registry=registry, max_iterations=3)
    final_partitions = {}
    
    print(f"   {'Attribute':<20} {'Initial':>8} {'Final':>8} {'Reduction':>10}")
    print("   " + "-"*46)
    
    for attr, initial in initial_partitions.items():
        final, quality = adaptive.discretize_with_assessment(initial, data)
        final_partitions[attr] = final
        reduction = (1 - len(final.bins) / len(initial.bins)) * 100
        print(f"   {attr:<20} {len(initial.bins):>8} {len(final.bins):>8} {reduction:>9.1f}%")
    
    # Stage 2: Feature Importance Scoring
    print(f"\n3. Stage 2: Computing feature importance...")
    print("="*70)
    
    metric_weights = {
        'completeness': 0.3,
        'outlier_rate': 0.3,
        'duplicate_rate': 0.2,
        'consistency_score': 0.2
    }
    
    scorer = FeatureScorer(
        metric_weights=metric_weights,
        quality_thresholds=get_default_quality_thresholds(),
        alpha=0.7,
        beta=0.3
    )
    
    ranked_features = scorer.score_features(final_partitions, data)
    
    print(f"\n   {'Rank':<6} {'Attribute':<20} {'Importance':>11} {'Bins':>6}")
    print("   " + "-"*43)
    
    for rank, feature in enumerate(ranked_features, 1):
        partition = final_partitions[feature.attribute]
        print(f"   {rank:<6} {feature.attribute:<20} {feature.importance:>11.4f} {len(partition.bins):>6}")
    
    # Scenario 1: Generous Budget
    print(f"\n4. Scenario 1: Generous Budget (B_max = 150)")
    print("="*70)
    
    selector1 = GreedyFeatureSelector(budget=150, registry=registry)
    result1 = selector1.select_features(ranked_features, final_partitions, data)
    
    print(summarize_selection(result1))
    
    # Scenario 2: Moderate Budget
    print(f"\n5. Scenario 2: Moderate Budget (B_max = 80)")
    print("="*70)
    
    selector2 = GreedyFeatureSelector(budget=80, registry=registry)
    result2 = selector2.select_features(ranked_features, final_partitions, data)
    
    print(summarize_selection(result2))
    
    # Scenario 3: Tight Budget with Adaptive Granularity
    print(f"\n6. Scenario 3: Tight Budget with Adaptation (B_max = 50)")
    print("="*70)
    
    selector3 = GreedyFeatureSelector(
        budget=50,
        registry=registry,
        min_bins_per_feature=5,
        max_adjustment_attempts=3
    )
    result3 = selector3.select_features(ranked_features, final_partitions, data)
    
    print(summarize_selection(result3))
    
    # Show adjustment details
    print(f"\n   Granularity Adjustments:")
    for feature in result3.selected_features:
        if feature.adjusted:
            original_bins = len(final_partitions[feature.attribute].bins)
            print(f"     {feature.attribute}: {original_bins} → {feature.n_bins} bins "
                  f"({(1 - feature.n_bins/original_bins)*100:.1f}% reduction)")
    
    # Scenario 4: Very Tight Budget
    print(f"\n7. Scenario 4: Very Tight Budget (B_max = 25)")
    print("="*70)
    
    selector4 = GreedyFeatureSelector(budget=25, registry=registry)
    result4 = selector4.select_features(ranked_features, final_partitions, data)
    
    print(summarize_selection(result4))
    
    # Extract final boundaries
    print(f"\n8. Final Discretization Boundaries:")
    print("="*70)
    
    boundaries = extract_discretization_boundaries(result3.selected_features)
    
    for attr, bounds in boundaries.items():
        print(f"\n   {attr}:")
        print(f"     Bins: {len(bounds) - 1}")
        print(f"     Range: [{bounds[0]:.2f}, {bounds[-1]:.2f}]")
        print(f"     Boundaries: ", end="")
        
        # Show first few and last few boundaries
        if len(bounds) <= 8:
            print(", ".join(f"{b:.2f}" for b in bounds))
        else:
            first_3 = ", ".join(f"{b:.2f}" for b in bounds[:3])
            last_3 = ", ".join(f"{b:.2f}" for b in bounds[-3:])
            print(f"{first_3}, ..., {last_3}")
    
    # Comparison across budgets
    print(f"\n9. Budget Impact Analysis:")
    print("="*70)
    
    results = [
        (150, result1),
        (80, result2),
        (50, result3),
        (25, result4)
    ]
    
    print(f"   {'Budget':>8} {'Selected':>10} {'Total Bins':>12} {'Utilization':>12} {'Adjusted':>10}")
    print("   " + "-"*52)
    
    for budget, result in results:
        n_adjusted = sum(1 for f in result.selected_features if f.adjusted)
        print(f"   {budget:>8} {len(result.selected_features):>10} "
              f"{result.total_bins:>12} {result.budget_utilization:>11.1%} {n_adjusted:>10}")
    
    # Quality-constrained selection
    print(f"\n10. Quality-Constrained Selection (B_max = 60):")
    print("="*70)
    
    selector_quality = GreedyFeatureSelector(budget=60, registry=registry)
    result_quality = selector_quality.select_with_quality_constraints(
        ranked_features,
        final_partitions,
        data,
        max_homogeneity=0.02,
        min_separation=0.005
    )
    
    print(summarize_selection(result_quality))
    print(f"\n   Note: Features validated for H ≤ 0.02 and S ≥ 0.005")
    
    print("\n" + "="*70)
    print("✅ WP 6 Example Complete")
    print("="*70)
    
    print("\nKey Takeaways:")
    print("- Greedy selection maximizes importance within budget")
    print("- Adaptive granularity allows more features by reducing bins")
    print("- Budget constraints force trade-off between coverage and detail")
    print("- Quality constraints ensure discretization effectiveness")
    print("- Final boundaries ready for learned index construction")


if __name__ == '__main__':
    main()
