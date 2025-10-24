"""
Tests for WP 2: Initial Partitioning and Annotation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from inferq import get_default_registry, InitialPartitioner, equal_frequency_binning


def test_equal_frequency_binning():
    """Test equal-frequency binning."""
    data = pd.Series(range(100))
    boundaries = equal_frequency_binning(data, n_bins=10)
    
    assert len(boundaries) == 11  # n_bins + 1
    assert boundaries[0] == 0
    assert boundaries[-1] == 99
    print("✓ Equal-frequency binning works")


def test_partition_single_attribute():
    """Test partitioning a single attribute."""
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'salary': np.random.randint(30000, 120000, 1000),
        'value': np.random.randn(1000)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    partition = partitioner.partition_attribute(df, 'age')
    
    assert partition.attribute_name == 'age'
    assert len(partition.bins) > 0
    assert len(partition.bins) <= 10
    
    # Check bins have quality vectors
    for bin in partition.bins:
        assert len(bin.quality_vector) > 0
        assert 'completeness' in bin.quality_vector
        assert len(bin.indices) > 0
    
    print(f"✓ Single attribute partitioning works ({len(partition.bins)} bins created)")


def test_partition_all_attributes():
    """Test partitioning multiple attributes."""
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 500),
        'salary': np.random.randint(30000, 120000, 500),
        'score': np.random.rand(500),
        'name': ['Alice', 'Bob', 'Charlie'] * 166 + ['David', 'Eve']
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=20)
    
    partitions = partitioner.partition_all_attributes(df)
    
    # Should partition numeric columns only
    assert 'age' in partitions
    assert 'salary' in partitions
    assert 'score' in partitions
    assert 'name' not in partitions
    
    # Check each partition
    for attr_name, partition in partitions.items():
        assert partition.attribute_name == attr_name
        assert len(partition.bins) > 0
        
        # Check quality vectors
        for bin in partition.bins:
            assert len(bin.quality_vector) > 0
    
    print(f"✓ Multi-attribute partitioning works ({len(partitions)} attributes)")


def test_bin_quality_vectors():
    """Test that quality vectors are computed correctly."""
    df = pd.DataFrame({
        'age': [25, 26, 27, 28, 29] * 20,  # 100 rows
        'salary': [50000] * 50 + [None] * 50,  # 50% missing
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=5)
    
    partition = partitioner.partition_attribute(df, 'age')
    
    # All bins should have quality vectors
    for bin in partition.bins:
        qv = bin.quality_vector
        
        # Should have computed completeness
        assert 'completeness' in qv
        # Completeness should be between 0 and 1
        assert 0.0 <= qv['completeness'] <= 1.0
    
    print("✓ Quality vector computation works")


def test_edge_cases():
    """Test edge cases."""
    df = pd.DataFrame({
        'varied': np.random.rand(100),  # Normal case
        'few': [1, 2, 3] * 33 + [1],  # Few unique values
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    # Should handle normal case
    partition_varied = partitioner.partition_attribute(df, 'varied')
    assert len(partition_varied.bins) >= 1
    
    # Should handle few unique values (will have fewer bins)
    partition_few = partitioner.partition_attribute(df, 'few')
    assert len(partition_few.bins) >= 1
    assert len(partition_few.bins) <= 3  # Only 3 unique values
    
    print("✓ Edge cases handled")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing WP 2: Initial Partitioning and Annotation")
    print("=" * 70)
    
    test_equal_frequency_binning()
    test_partition_single_attribute()
    test_partition_all_attributes()
    test_bin_quality_vectors()
    test_edge_cases()
    
    print("\n" + "=" * 70)
    print("✅ All WP 2 tests passed!")
    print("=" * 70)
