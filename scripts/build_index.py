"""
WP 9: Offline Index Construction

Master script that orchestrates the complete pipeline for building
a learned quality index from raw data.

Pipeline:
  WP 1: Quality Metrics
  WP 2-4: Stage 1 - Discretization and Quality Assessment
  WP 5-6: Stage 2 - Feature Importance and Selection
  WP 7: Stage 3 - Bin Dictionary and Training Data
  WP 8: Stage 3 - Model Training and Index Creation
"""

import argparse
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.quality_assessment import AdaptiveMTQD
from inferq.feature_scoring import FeatureScorer, get_default_quality_thresholds
from inferq.feature_selection import GreedyFeatureSelector, summarize_selection
from inferq.bin_dictionary import BinDictionary, generate_training_data, save_training_data
from inferq.learned_index import QualityIndex


def build_quality_index(
    data_path: str,
    output_path: str,
    budget: int = 100,
    n_estimators: int = 100,
    max_depth: Optional[int] = 20,
    initial_bins: int = 30,
    test_size: float = 0.2,
    metric_weights: Optional[Dict[str, float]] = None,
    quality_thresholds: Optional[Dict] = None,
    alpha: float = 0.7,
    beta: float = 0.3,
    numeric_only: bool = True,
    verbose: bool = True
) -> QualityIndex:
    """
    Build complete quality index from raw data.
    
    Args:
        data_path: Path to input CSV file
        output_path: Path to save index (without extension)
        budget: Maximum total bins across features (B_max)
        n_estimators: Number of trees in Random Forest
        max_depth: Maximum tree depth
        initial_bins: Initial number of bins for discretization
        test_size: Fraction of data for model testing
        metric_weights: Weights for quality metrics in feature scoring
        quality_thresholds: Thresholds for quality issue detection
        alpha: Weight for IG_multi in feature importance
        beta: Weight for QDP in feature importance
        numeric_only: Only process numeric attributes
        verbose: Print progress
        
    Returns:
        Trained QualityIndex
    """
    start_time = time.time()
    
    if verbose:
        print("="*70)
        print("InferQ: Building Quality-Aware Learned Index")
        print("="*70)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    if verbose:
        print(f"\n[1/8] Loading data from {data_path}...")
    
    data = pd.read_csv(data_path)
    
    if numeric_only:
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out label columns
        numeric_cols = [c for c in numeric_cols if not any(
            x in c.lower() for x in ['label', 'target', 'class', '>50k']
        )]
        data = data[numeric_cols]
    
    if verbose:
        print(f"   Loaded: {len(data)} rows, {len(data.columns)} columns")
        print(f"   Columns: {', '.join(data.columns[:10])}")
        if len(data.columns) > 10:
            print(f"            ... and {len(data.columns) - 10} more")
    
    # =========================================================================
    # WP 1: Initialize Quality Metrics
    # =========================================================================
    if verbose:
        print(f"\n[2/8] WP 1: Initializing quality metrics...")
    
    registry = get_default_registry()
    
    if verbose:
        print(f"   Available metrics: {len(registry._metrics)}")
        print(f"   Categories: {', '.join(set(m.category for m in registry._metrics.values()))}")
    
    # =========================================================================
    # WP 2-4: Stage 1 - Discretization and Quality Assessment
    # =========================================================================
    if verbose:
        print(f"\n[3/8] WP 2-4: Stage 1 - Discretization...")
    
    partitioner = InitialPartitioner(registry, n_bins=initial_bins)
    
    # Create initial partitions
    initial_partitions = {}
    for attr in data.columns:
        if verbose:
            print(f"   Partitioning {attr}...", end=" ", flush=True)
        initial_partitions[attr] = partitioner.partition_attribute(data, attr)
        if verbose:
            print(f"{len(initial_partitions[attr].bins)} bins")
    
    # Apply adaptive MTQD
    if verbose:
        print(f"\n   Applying adaptive MTQD...")
    
    adaptive = AdaptiveMTQD(registry=registry, max_iterations=5)
    final_partitions = {}
    
    for attr, initial in initial_partitions.items():
        if verbose:
            print(f"   {attr}...", end=" ", flush=True)
        final, quality = adaptive.discretize_with_assessment(initial, data)
        final_partitions[attr] = final
        if verbose:
            reduction = (1 - len(final.bins) / len(initial.bins)) * 100
            print(f"{len(initial.bins)} → {len(final.bins)} bins ({reduction:.1f}% reduction)")
    
    # =========================================================================
    # WP 5-6: Stage 2 - Feature Importance and Selection
    # =========================================================================
    if verbose:
        print(f"\n[4/8] WP 5: Feature importance scoring...")
    
    if metric_weights is None:
        metric_weights = {
            'completeness': 0.3,
            'outlier_rate': 0.3,
            'duplicate_rate': 0.2,
            'consistency_score': 0.2
        }
    
    if quality_thresholds is None:
        quality_thresholds = get_default_quality_thresholds()
    
    scorer = FeatureScorer(
        metric_weights=metric_weights,
        quality_thresholds=quality_thresholds,
        alpha=alpha,
        beta=beta
    )
    
    ranked_features = scorer.score_features(final_partitions, data)
    
    if verbose:
        print(f"   Ranked {len(ranked_features)} features")
        print(f"   Top 5:")
        for i, feature in enumerate(ranked_features[:5], 1):
            print(f"     {i}. {feature.attribute}: importance={feature.importance:.4f}")
    
    # Feature selection
    if verbose:
        print(f"\n[5/8] WP 6: Feature selection (budget={budget})...")
    
    selector = GreedyFeatureSelector(
        budget=budget,
        registry=registry,
        min_bins_per_feature=3
    )
    
    result = selector.select_features(ranked_features, final_partitions, data)
    
    if verbose:
        print(f"\n{summarize_selection(result)}")
    
    # =========================================================================
    # WP 7: Stage 3 - Bin Dictionary and Training Data
    # =========================================================================
    if verbose:
        print(f"\n[6/8] WP 7: Building bin dictionary and training data...")
    
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    
    if verbose:
        print(f"   Features: {bin_dict.n_features}")
        print(f"   Total bins: {bin_dict.total_bins}")
    
    training_data = generate_training_data(data, bin_dict)
    
    if verbose:
        print(f"   Training data shape: X={training_data.X.shape}, Y={training_data.Y.shape}")
    
    # Save training data
    training_path = str(Path(output_path).parent / (Path(output_path).stem + "_training"))
    save_training_data(training_data, training_path)
    
    if verbose:
        print(f"   Saved training data to: {training_path}.npz")
    
    # =========================================================================
    # WP 8: Stage 3 - Model Training
    # =========================================================================
    if verbose:
        print(f"\n[7/8] WP 8: Training Random Forest model...")
    
    index = QualityIndex.from_training_data(
        training_data,
        bin_dict,
        n_estimators=n_estimators,
        max_depth=max_depth,
        test_size=test_size,
        verbose=verbose
    )
    
    if verbose and index.metrics:
        print(f"\n   Model Performance:")
        print(f"     Test R²:  {index.metrics.test_r2:.4f}")
        print(f"     Test MSE: {index.metrics.test_mse:.6f}")
        print(f"     Test MAE: {index.metrics.test_mae:.6f}")
    
    # =========================================================================
    # Save Index
    # =========================================================================
    if verbose:
        print(f"\n[8/8] Saving index...")
    
    index.save(output_path)
    
    if verbose:
        print(f"   Saved to: {output_path}.pkl")
    
    # Save metadata
    metadata = {
        'data_path': data_path,
        'n_samples': len(data),
        'n_features_total': len(data.columns),
        'n_features_selected': bin_dict.n_features,
        'selected_features': bin_dict.feature_order,
        'total_bins': bin_dict.total_bins,
        'budget': budget,
        'budget_utilization': result.budget_utilization,
        'model_type': 'RandomForest',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'test_r2': float(index.metrics.test_r2) if index.metrics else None,
        'test_mse': float(index.metrics.test_mse) if index.metrics else None,
        'metric_names': index.metric_names,
        'build_time': time.time() - start_time
    }
    
    metadata_path = output_path + "_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"   Saved metadata to: {metadata_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    
    if verbose:
        print("\n" + "="*70)
        print("✅ Index Construction Complete")
        print("="*70)
        print(f"   Total time: {elapsed:.2f}s")
        print(f"   Data: {len(data)} samples")
        print(f"   Selected: {bin_dict.n_features}/{len(data.columns)} features")
        print(f"   Bins: {bin_dict.total_bins}/{budget} ({result.budget_utilization:.1%})")
        print(f"   Model accuracy (R²): {index.metrics.test_r2:.4f}")
        print(f"   Output: {output_path}.pkl")
        print("="*70)
    
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Build quality-aware learned index from data"
    )
    
    parser.add_argument(
        'data',
        type=str,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='quality_index',
        help='Output path for index (default: quality_index)'
    )
    
    parser.add_argument(
        '-b', '--budget',
        type=int,
        default=100,
        help='Maximum total bins (default: 100)'
    )
    
    parser.add_argument(
        '-n', '--n-estimators',
        type=int,
        default=100,
        help='Number of trees in Random Forest (default: 100)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=20,
        help='Maximum tree depth (default: 20)'
    )
    
    parser.add_argument(
        '--initial-bins',
        type=int,
        default=30,
        help='Initial bins for discretization (default: 30)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction for testing (default: 0.2)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Build index
    index = build_quality_index(
        data_path=args.data,
        output_path=args.output,
        budget=args.budget,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        initial_bins=args.initial_bins,
        test_size=args.test_size,
        verbose=not args.quiet
    )
    
    return index


if __name__ == '__main__':
    main()
