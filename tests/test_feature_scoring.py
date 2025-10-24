"""
Tests for WP 5: Feature Importance Scoring
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')

from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.discretization import MTQD
from inferq.feature_scoring import (
    compute_entropy,
    compute_information_gain,
    compute_ig_multi,
    has_quality_issue,
    compute_qdp,
    FeatureScorer,
    get_default_quality_thresholds
)


def test_entropy():
    """Test entropy computation"""
    print("\n" + "="*70)
    print("Testing WP 5: Feature Importance Scoring")
    print("="*70)
    
    # Uniform distribution - maximum entropy
    uniform = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    h_uniform = compute_entropy(uniform)
    
    # Single value - zero entropy
    constant = np.array([1, 1, 1, 1, 1, 1])
    h_constant = compute_entropy(constant)
    
    print(f"✓ Entropy computation works")
    print(f"  Uniform: {h_uniform:.3f} bits")
    print(f"  Constant: {h_constant:.3f} bits")
    
    assert abs(h_constant) < 1e-6, "Constant should have zero entropy"
    assert h_uniform > 2.5, "Uniform 8-values should have ~3 bits entropy"


def test_information_gain():
    """Test base information gain computation"""
    np.random.seed(42)
    n = 500
    
    # Create data where one attribute correlates with quality
    data = pd.DataFrame({
        'good_feature': np.concatenate([
            np.random.normal(30, 5, n//2),  # Low values = low quality
            np.random.normal(60, 5, n//2)   # High values = high quality
        ]),
        'random_feature': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    # Partition both features
    partition_good = partitioner.partition_attribute(data, 'good_feature')
    partition_random = partitioner.partition_attribute(data, 'random_feature')
    
    # IG for completeness metric
    ig_good = compute_information_gain(partition_good, 'completeness', data)
    ig_random = compute_information_gain(partition_random, 'completeness', data)
    
    print(f"✓ Information gain computation works")
    print(f"  Good feature: IG={ig_good:.4f}")
    print(f"  Random feature: IG={ig_random:.4f}")


def test_ig_multi():
    """Test multi-target information gain"""
    np.random.seed(42)
    n = 400
    data = pd.DataFrame({
        'age': np.random.randint(20, 70, n)
    })
    
    # Add missing values
    missing_mask = np.random.random(n) < 0.2
    data.loc[missing_mask, 'age'] = np.nan
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    partition = partitioner.partition_attribute(data, 'age')
    
    # Compute IG_multi with different weights
    weights = {'completeness': 0.5, 'outlier_rate': 0.3, 'duplicate_rate': 0.2}
    ig_multi = compute_ig_multi(partition, data, weights)
    
    print(f"✓ IG_multi computation works")
    print(f"  IG_multi = {ig_multi:.4f}")
    
    assert ig_multi >= 0, "IG_multi should be non-negative"


def test_quality_issue_detection():
    """Test has_quality_issue helper"""
    from inferq.partitioning import Bin
    
    # Bin with good quality
    good_bin = Bin(
        lower_bound=0,
        upper_bound=10,
        indices=np.array([0, 1, 2]),
        quality_vector={'completeness': 0.98, 'outlier_rate': 0.02}
    )
    
    # Bin with issues
    bad_bin = Bin(
        lower_bound=10,
        upper_bound=20,
        indices=np.array([3, 4, 5]),
        quality_vector={'completeness': 0.80, 'outlier_rate': 0.15}
    )
    
    thresholds = {
        'completeness': (0.95, 1.0),
        'outlier_rate': (0.0, 0.05)
    }
    
    assert not has_quality_issue(good_bin, thresholds)
    assert has_quality_issue(bad_bin, thresholds)
    
    print(f"✓ Quality issue detection works")


def test_qdp():
    """Test Quality Detection Power computation"""
    np.random.seed(42)
    n = 600
    
    # Create data with quality issues in specific ranges
    data = pd.DataFrame({
        'value': np.random.uniform(0, 100, n)
    })
    
    # Add more missing values in certain ranges
    for i in range(n):
        if data.loc[i, 'value'] < 30:  # Low values have issues
            if np.random.random() < 0.3:
                data.loc[i, 'value'] = np.nan
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    partition = partitioner.partition_attribute(data, 'value')
    
    thresholds = get_default_quality_thresholds()
    qdp = compute_qdp(partition, thresholds, len(data))
    
    print(f"✓ QDP computation works")
    print(f"  QDP = {qdp:.4f}")
    
    assert 0 <= qdp <= 1, "QDP should be between 0 and 1"


def test_feature_scorer():
    """Test FeatureScorer class"""
    np.random.seed(42)
    n = 500
    
    # Create multi-attribute dataset
    data = pd.DataFrame({
        'age': np.random.randint(20, 70, n),
        'salary': np.random.randint(30000, 120000, n),
        'experience': np.random.randint(0, 40, n)
    })
    
    # Add quality issues
    for i in range(n):
        if data.loc[i, 'age'] < 30:
            if np.random.random() < 0.3:
                data.loc[i, 'salary'] = np.nan
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    # Create partitions
    partitions = {}
    for attr in ['age', 'salary', 'experience']:
        partitions[attr] = partitioner.partition_attribute(data, attr)
    
    # Score features
    scorer = FeatureScorer(
        metric_weights={'completeness': 0.5, 'outlier_rate': 0.5},
        quality_thresholds=get_default_quality_thresholds(),
        alpha=0.7,
        beta=0.3
    )
    
    scores = scorer.score_features(partitions, data)
    
    print(f"✓ FeatureScorer works")
    print(f"  Scored {len(scores)} features:")
    for score in scores:
        print(f"    {score.attribute}: importance={score.importance:.4f}, "
              f"IG={score.ig_multi:.4f}, QDP={score.qdp:.4f}")
    
    assert len(scores) == 3
    assert all(s.importance >= 0 for s in scores)


def test_feature_selection():
    """Test top-k feature selection"""
    np.random.seed(42)
    n = 400
    
    data = pd.DataFrame({
        'f1': np.random.uniform(0, 100, n),
        'f2': np.random.uniform(0, 100, n),
        'f3': np.random.uniform(0, 100, n),
        'f4': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=8)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    top_2 = scorer.select_top_features(partitions, data, k=2)
    
    print(f"✓ Feature selection works")
    print(f"  Top 2 features: {top_2}")
    
    assert len(top_2) == 2


def test_combined_score():
    """Test that combined score respects alpha and beta weights"""
    np.random.seed(42)
    n = 300
    data = pd.DataFrame({'x': np.random.uniform(0, 100, n)})
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    partition = partitioner.partition_attribute(data, 'x')
    
    # Test different alpha/beta combinations
    scorer1 = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds(),
        alpha=1.0, beta=0.0
    )
    score1 = scorer1.score_feature(partition, data)
    
    scorer2 = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds(),
        alpha=0.0, beta=1.0
    )
    score2 = scorer2.score_feature(partition, data)
    
    print(f"✓ Combined score weighting works")
    print(f"  IG-only (α=1, β=0): {score1.importance:.4f}")
    print(f"  QDP-only (α=0, β=1): {score2.importance:.4f}")
    
    assert abs(score1.importance - score1.ig_multi) < 1e-6
    assert abs(score2.importance - score2.qdp) < 1e-6


if __name__ == '__main__':
    test_entropy()
    test_information_gain()
    test_ig_multi()
    test_quality_issue_detection()
    test_qdp()
    test_feature_scorer()
    test_feature_selection()
    test_combined_score()
    
    print("\n" + "="*70)
    print("✅ All WP 5 tests passed!")
    print("="*70)
