"""
Tests for WP 3: Multi-Target Quality-aware Discretization
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
    compute_merge_cost
)


def test_compute_merge_cost():
    """Test merge cost computation."""
    from inferq.partitioning import Bin
    
    # Create two similar bins
    bin1 = Bin(0, 10, np.array([0, 1, 2]))
    bin1.quality_vector = {'completeness': 0.9, 'outlier_rate': 0.1}
    
    bin2 = Bin(10, 20, np.array([3, 4, 5]))
    bin2.quality_vector = {'completeness': 0.92, 'outlier_rate': 0.08}
    
    weights = {'completeness': 0.5, 'outlier_rate': 0.5}
    cost = compute_merge_cost(bin1, bin2, weights)
    
    # Cost should be small (bins are similar)
    assert cost >= 0
    assert cost < 0.01
    
    # Create dissimilar bin
    bin3 = Bin(20, 30, np.array([6, 7, 8]))
    bin3.quality_vector = {'completeness': 0.5, 'outlier_rate': 0.5}
    
    cost_dissimilar = compute_merge_cost(bin1, bin3, weights)
    assert cost_dissimilar > cost  # Should be more expensive
    
    print("✓ Merge cost computation works")


def test_mtqd_basic():
    """Test basic MTQD merging."""
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 500),
        'salary': np.random.randint(30000, 120000, 500),
    })
    
    registry = get_default_registry()
    
    # Create initial partition
    partitioner = InitialPartitioner(registry, n_bins=20)
    initial_partition = partitioner.partition_attribute(df, 'age')
    
    initial_bins = len(initial_partition.bins)
    
    # Apply MTQD
    mtqd = MTQD(registry, merge_threshold=0.01)
    optimized_partition = mtqd.discretize(initial_partition, df)
    
    # Should have fewer bins after merging
    assert len(optimized_partition.bins) <= initial_bins
    assert len(optimized_partition.bins) > 0
    
    # Bins should still have quality vectors
    for bin in optimized_partition.bins:
        assert len(bin.quality_vector) > 0
    
    print(f"✓ MTQD merging works ({initial_bins} → {len(optimized_partition.bins)} bins)")


def test_mtqd_threshold():
    """Test that merge threshold controls merging."""
    np.random.seed(42)
    df = pd.DataFrame({
        'value': np.random.randn(300),
        'extra': np.random.randn(300),
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=15)
    initial_partition = partitioner.partition_attribute(df, 'value')
    
    # Strict threshold - less merging
    mtqd_strict = MTQD(registry, merge_threshold=0.001)
    result_strict = mtqd_strict.discretize(initial_partition, df)
    
    # Loose threshold - more merging
    mtqd_loose = MTQD(registry, merge_threshold=0.1)
    result_loose = mtqd_loose.discretize(initial_partition, df)
    
    # Loose should merge more
    assert len(result_loose.bins) <= len(result_strict.bins)
    
    print(f"✓ Threshold control works (strict: {len(result_strict.bins)}, loose: {len(result_loose.bins)} bins)")


def test_mtqd_all_attributes():
    """Test MTQD on multiple attributes."""
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 400),
        'salary': np.random.randint(30000, 120000, 400),
        'experience': np.random.randint(0, 40, 400),
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=15)
    
    # Create initial partitions
    initial_partitions = partitioner.partition_all_attributes(df)
    
    # Apply MTQD to all
    mtqd = MTQD(registry, merge_threshold=0.01)
    optimized_partitions = mtqd.discretize_all(initial_partitions, df)
    
    # Check all attributes were processed
    assert len(optimized_partitions) == len(initial_partitions)
    
    for attr_name in initial_partitions.keys():
        initial_bins = len(initial_partitions[attr_name].bins)
        optimized_bins = len(optimized_partitions[attr_name].bins)
        
        # Should have same or fewer bins
        assert optimized_bins <= initial_bins
        assert optimized_bins > 0
    
    print(f"✓ Multi-attribute MTQD works ({len(optimized_partitions)} attributes)")


def test_merge_preserves_data():
    """Test that merging preserves all data indices."""
    np.random.seed(42)
    df = pd.DataFrame({
        'value': np.random.rand(200),
        'extra': range(200),
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    initial_partition = partitioner.partition_attribute(df, 'value')
    
    # Count initial indices
    initial_indices = set()
    for bin in initial_partition.bins:
        initial_indices.update(bin.indices)
    
    # Apply MTQD
    mtqd = MTQD(registry, merge_threshold=0.05)
    optimized_partition = mtqd.discretize(initial_partition, df)
    
    # Count merged indices
    merged_indices = set()
    for bin in optimized_partition.bins:
        merged_indices.update(bin.indices)
    
    # Should preserve all indices
    assert initial_indices == merged_indices
    
    print("✓ Merging preserves all data")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing WP 3: Multi-Target Quality-aware Discretization")
    print("=" * 70)
    
    test_compute_merge_cost()
    test_mtqd_basic()
    test_mtqd_threshold()
    test_mtqd_all_attributes()
    test_merge_preserves_data()
    
    print("\n" + "=" * 70)
    print("✅ All WP 3 tests passed!")
    print("=" * 70)
