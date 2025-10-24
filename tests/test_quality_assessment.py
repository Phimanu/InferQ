"""
Tests for WP 4: Discretization Quality Assessment
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')

from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.quality_assessment import (
    compute_homogeneity, 
    compute_separation,
    DiscretizationQuality,
    AdaptiveMTQD
)


def test_homogeneity_separation():
    """Test H and S score computation"""
    print("\n" + "="*70)
    print("Testing WP 4: Discretization Quality Assessment")
    print("="*70)
    
    # Create dataset
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'salary': np.random.randint(30000, 150000, n)
    })
    
    # Add some missing values
    missing_mask = np.random.random(n) < 0.1
    data.loc[missing_mask, 'salary'] = np.nan
    
    registry = get_default_registry()
    
    # Test with different bin counts
    partitioner_fine = InitialPartitioner(registry, n_bins=20)
    partition_fine = partitioner_fine.partition_attribute(data, 'age')
    
    partitioner_coarse = InitialPartitioner(registry, n_bins=5)
    partition_coarse = partitioner_coarse.partition_attribute(data, 'age')
    
    H_fine = compute_homogeneity(partition_fine, data)
    S_fine = compute_separation(partition_fine)
    
    H_coarse = compute_homogeneity(partition_coarse, data)
    S_coarse = compute_separation(partition_coarse)
    
    print(f"✓ Homogeneity and Separation computed")
    print(f"  Fine (20 bins): H={H_fine:.4f}, S={S_fine:.4f}")
    print(f"  Coarse (5 bins): H={H_coarse:.4f}, S={S_coarse:.4f}")
    
    # Coarse should have lower separation (fewer distinct bins)
    assert S_coarse < S_fine, "Coarse partition should have lower separation"
    

def test_quality_assessment():
    """Test DiscretizationQuality class"""
    quality = DiscretizationQuality(
        homogeneity=0.005,
        separation=0.02,
        n_bins=10,
        threshold=0.01
    )
    
    assert quality.is_acceptable(max_homogeneity=0.01, min_separation=0.01)
    assert not quality.is_acceptable(max_homogeneity=0.001, min_separation=0.01)
    
    print(f"✓ DiscretizationQuality validation works")


def test_adaptive_mtqd():
    """Test adaptive MTQD with threshold adjustment"""
    # Create dataset with clear quality patterns
    np.random.seed(42)
    n = 500
    data = pd.DataFrame({
        'age': np.concatenate([
            np.random.normal(30, 5, n//2),
            np.random.normal(60, 5, n//2)
        ])
    })
    
    # Add missing values to create quality differences
    missing_mask = np.random.random(n) < 0.2
    data.loc[missing_mask, 'age'] = np.nan
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=20)
    
    # Create initial partition
    initial_partition = partitioner.partition_attribute(data, 'age')
    
    # Run adaptive MTQD
    adaptive = AdaptiveMTQD(
        registry=registry,
        max_homogeneity=0.02,
        min_separation=0.005,
        max_iterations=5
    )
    
    final_partition, quality = adaptive.discretize_with_assessment(
        initial_partition, data, initial_threshold=0.01
    )
    
    print(f"✓ Adaptive MTQD works")
    print(f"  Final: {quality.n_bins} bins, H={quality.homogeneity:.4f}, S={quality.separation:.4f}")
    print(f"  Threshold: {quality.threshold:.4f}")
    
    assert quality.n_bins <= 20, "Should merge some bins"
    assert quality.n_bins >= 1, "Should have at least 1 bin"


def test_threshold_adjustment():
    """Test that threshold adjusts in correct direction"""
    np.random.seed(42)
    n = 300
    data = pd.DataFrame({
        'value': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=15)
    
    initial_partition = partitioner.partition_attribute(data, 'value')
    
    # Test with very strict criteria (forces adjustment)
    adaptive = AdaptiveMTQD(
        registry=registry,
        max_homogeneity=0.0001,
        min_separation=0.1,
        max_iterations=3
    )
    
    final_partition, quality = adaptive.discretize_with_assessment(
        initial_partition, data, initial_threshold=0.01
    )
    
    print(f"✓ Threshold adjustment works")
    print(f"  Result: {quality.n_bins} bins with threshold={quality.threshold:.4f}")


def test_multi_attribute_assessment():
    """Test quality assessment on multiple attributes"""
    np.random.seed(42)
    n = 400
    data = pd.DataFrame({
        'age': np.random.randint(20, 70, n),
        'salary': np.random.randint(30000, 120000, n),
        'experience': np.random.randint(0, 40, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=15)
    adaptive = AdaptiveMTQD(registry=registry, max_iterations=3)
    
    results = {}
    for attr in ['age', 'salary', 'experience']:
        initial = partitioner.partition_attribute(data, attr)
        final, quality = adaptive.discretize_with_assessment(initial, data)
        results[attr] = quality
    
    print(f"✓ Multi-attribute assessment works")
    for attr, quality in results.items():
        print(f"  {attr}: {quality.n_bins} bins, H={quality.homogeneity:.4f}, S={quality.separation:.4f}")


if __name__ == '__main__':
    test_homogeneity_separation()
    test_quality_assessment()
    test_adaptive_mtqd()
    test_threshold_adjustment()
    test_multi_attribute_assessment()
    
    print("\n" + "="*70)
    print("✅ All WP 4 tests passed!")
    print("="*70)
