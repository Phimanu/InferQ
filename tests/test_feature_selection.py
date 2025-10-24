"""
Tests for WP 6: Greedy Feature Selection
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')

from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.feature_scoring import FeatureScorer, get_default_quality_thresholds
from inferq.feature_selection import (
    GreedyFeatureSelector,
    extract_discretization_boundaries,
    summarize_selection
)


def test_basic_selection():
    """Test basic greedy selection without budget constraint"""
    print("\n" + "="*70)
    print("Testing WP 6: Greedy Feature Selection")
    print("="*70)
    
    np.random.seed(42)
    n = 400
    data = pd.DataFrame({
        'f1': np.random.uniform(0, 100, n),
        'f2': np.random.uniform(0, 100, n),
        'f3': np.random.uniform(0, 100, n)
    })
    
    # Setup
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    # Select with generous budget (should fit all)
    selector = GreedyFeatureSelector(budget=50, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    print(f"✓ Basic selection works")
    print(f"  Budget: {result.budget}, Used: {result.total_bins}")
    print(f"  Selected: {len(result.selected_features)} features")
    print(f"  Rejected: {len(result.rejected_features)} features")
    
    assert len(result.selected_features) == 3
    assert result.total_bins <= result.budget


def test_budget_constraint():
    """Test selection with tight budget"""
    np.random.seed(42)
    n = 300
    data = pd.DataFrame({
        'a': np.random.uniform(0, 100, n),
        'b': np.random.uniform(0, 100, n),
        'c': np.random.uniform(0, 100, n),
        'd': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=12)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    # Tight budget - can only fit 2 features
    selector = GreedyFeatureSelector(budget=25, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    print(f"✓ Budget constraint works")
    print(f"  Budget: {result.budget}, Used: {result.total_bins}")
    print(f"  Selected: {len(result.selected_features)}")
    print(f"  Rejected: {len(result.rejected_features)}")
    
    assert result.total_bins <= result.budget
    assert len(result.rejected_features) > 0


def test_adaptive_granularity():
    """Test adaptive granularity adjustment"""
    np.random.seed(42)
    n = 500
    data = pd.DataFrame({
        'x': np.random.uniform(0, 100, n),
        'y': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=20)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    # Budget that requires adjustment for second feature
    selector = GreedyFeatureSelector(budget=30, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    print(f"✓ Adaptive granularity works")
    print(f"  Budget: {result.budget}, Used: {result.total_bins}")
    
    for feature in result.selected_features:
        adj_str = "adjusted" if feature.adjusted else "original"
        print(f"    {feature.attribute}: {feature.n_bins} bins ({adj_str})")
    
    # Check if any feature was adjusted
    adjusted_count = sum(1 for f in result.selected_features if f.adjusted)
    print(f"  Features adjusted: {adjusted_count}")
    
    assert result.total_bins <= result.budget


def test_boundary_extraction():
    """Test discretization boundary extraction"""
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'attr1': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    partitions = {'attr1': partitioner.partition_attribute(data, 'attr1')}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    selector = GreedyFeatureSelector(budget=15, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    boundaries = extract_discretization_boundaries(result.selected_features)
    
    print(f"✓ Boundary extraction works")
    print(f"  Extracted boundaries for {len(boundaries)} features")
    
    for attr, bounds in boundaries.items():
        print(f"    {attr}: {len(bounds)} boundaries, range [{bounds[0]:.1f}, {bounds[-1]:.1f}]")
    
    assert 'attr1' in boundaries
    assert len(boundaries['attr1']) > 0


def test_quality_constrained_selection():
    """Test selection with quality constraints"""
    np.random.seed(42)
    n = 350
    data = pd.DataFrame({
        'a': np.random.uniform(0, 100, n),
        'b': np.random.uniform(0, 100, n),
        'c': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=15)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    selector = GreedyFeatureSelector(budget=35, registry=registry)
    result = selector.select_with_quality_constraints(
        ranked, partitions, data,
        max_homogeneity=0.02,
        min_separation=0.005
    )
    
    print(f"✓ Quality-constrained selection works")
    print(f"  Selected: {len(result.selected_features)} features")
    print(f"  Budget utilization: {result.budget_utilization:.1%}")
    
    assert result.total_bins <= result.budget


def test_summary_generation():
    """Test summary generation"""
    np.random.seed(42)
    n = 250
    data = pd.DataFrame({
        'x': np.random.uniform(0, 100, n),
        'y': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=8)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    selector = GreedyFeatureSelector(budget=12, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    summary = summarize_selection(result)
    
    print(f"✓ Summary generation works")
    print("\n" + summary)
    
    assert "Feature Selection Summary" in summary
    assert "Budget:" in summary


def test_min_bins_constraint():
    """Test minimum bins per feature constraint"""
    np.random.seed(42)
    n = 300
    data = pd.DataFrame({
        'a': np.random.uniform(0, 100, n),
        'b': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=15)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    # Very tight budget with min_bins constraint
    selector = GreedyFeatureSelector(
        budget=18,
        registry=registry,
        min_bins_per_feature=5
    )
    result = selector.select_features(ranked, partitions, data)
    
    print(f"✓ Minimum bins constraint works")
    print(f"  Selected features: {len(result.selected_features)}")
    
    # All selected features should have at least min_bins
    for feature in result.selected_features:
        assert feature.n_bins >= 5
        print(f"    {feature.attribute}: {feature.n_bins} bins (>= 5)")


if __name__ == '__main__':
    test_basic_selection()
    test_budget_constraint()
    test_adaptive_granularity()
    test_boundary_extraction()
    test_quality_constrained_selection()
    test_summary_generation()
    test_min_bins_constraint()
    
    print("\n" + "="*70)
    print("✅ All WP 6 tests passed!")
    print("="*70)
