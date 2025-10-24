"""
Tests for WP 7: Bin Dictionary and Training Data Generation
"""

import numpy as np
import pandas as pd
import sys
import tempfile
import os
sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')

from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.feature_scoring import FeatureScorer, get_default_quality_thresholds
from inferq.feature_selection import GreedyFeatureSelector
from inferq.bin_dictionary import (
    BinDictionary,
    generate_training_data,
    save_training_data,
    load_training_data
)


def test_bin_dictionary_creation():
    """Test BinDictionary creation from selected features"""
    print("\n" + "="*70)
    print("Testing WP 7: Bin Dictionary and Training Data Generation")
    print("="*70)
    
    np.random.seed(42)
    n = 300
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'salary': np.random.uniform(30000, 120000, n)
    })
    
    # Create partitions and select features
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    selector = GreedyFeatureSelector(budget=25, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    # Build dictionary
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    
    print(f"✓ BinDictionary creation works")
    print(f"  Features: {bin_dict.n_features}")
    print(f"  Total bins: {bin_dict.total_bins}")
    print(f"  Numeric features: {len(bin_dict.numeric_boundaries)}")
    
    assert bin_dict.n_features > 0
    assert bin_dict.total_bins > 0
    assert len(bin_dict.feature_order) == bin_dict.n_features


def test_bin_id_lookup():
    """Test single value bin ID lookup"""
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'x': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    partition = partitioner.partition_attribute(data, 'x')
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features({'x': partition}, data)
    
    selector = GreedyFeatureSelector(budget=15, registry=registry)
    result = selector.select_features(ranked, {'x': partition}, data)
    
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    
    # Test lookups
    test_values = [10.0, 50.0, 90.0, np.nan]
    
    print(f"✓ Bin ID lookup works")
    for value in test_values:
        bin_id = bin_dict.get_bin_id('x', value)
        print(f"  Value {value} → Bin {bin_id}")
        assert 0 <= bin_id < bin_dict.total_bins


def test_bin_vector():
    """Test get_bin_vector for multi-feature row"""
    np.random.seed(42)
    n = 250
    data = pd.DataFrame({
        'a': np.random.uniform(0, 100, n),
        'b': np.random.uniform(0, 50, n),
        'c': np.random.uniform(0, 200, n)
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
    
    selector = GreedyFeatureSelector(budget=20, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    
    # Test on sample row
    sample_row = data.iloc[0]
    bin_vector = bin_dict.get_bin_vector(sample_row)
    
    print(f"✓ Bin vector generation works")
    print(f"  Row: {dict(sample_row)}")
    print(f"  Bin vector: {bin_vector}")
    print(f"  Shape: {bin_vector.shape}")
    
    assert len(bin_vector) == bin_dict.n_features
    assert bin_vector.dtype == np.int32


def test_batch_bin_vectors():
    """Test batch bin vector generation"""
    np.random.seed(42)
    n = 400
    data = pd.DataFrame({
        'x': np.random.uniform(0, 100, n),
        'y': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    selector = GreedyFeatureSelector(budget=25, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    
    # Batch processing
    bin_vectors = bin_dict.batch_get_bin_vectors(data)
    
    print(f"✓ Batch bin vector generation works")
    print(f"  Data shape: {data.shape}")
    print(f"  Bin vectors shape: {bin_vectors.shape}")
    print(f"  Sample vectors: {bin_vectors[:3]}")
    
    assert bin_vectors.shape == (len(data), bin_dict.n_features)


def test_quality_vector_retrieval():
    """Test quality vector retrieval from bin assignment"""
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'attr': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    partition = partitioner.partition_attribute(data, 'attr')
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features({'attr': partition}, data)
    
    selector = GreedyFeatureSelector(budget=15, registry=registry)
    result = selector.select_features(ranked, {'attr': partition}, data)
    
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    
    # Get quality for sample row
    sample_row = data.iloc[0]
    bin_vector = bin_dict.get_bin_vector(sample_row)
    quality_vector = bin_dict.get_quality_vector(bin_vector)
    
    print(f"✓ Quality vector retrieval works")
    print(f"  Bin vector: {bin_vector}")
    print(f"  Quality metrics: {list(quality_vector.keys())}")
    print(f"  Sample values: {dict(list(quality_vector.items())[:3])}")
    
    assert len(quality_vector) > 0
    assert 'completeness' in quality_vector


def test_training_data_generation():
    """Test full training data generation"""
    np.random.seed(42)
    n = 350
    data = pd.DataFrame({
        'a': np.random.uniform(0, 100, n),
        'b': np.random.uniform(0, 50, n)
    })
    
    # Add missing values
    missing_mask = np.random.random(n) < 0.1
    data.loc[missing_mask, 'b'] = np.nan
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0, 'outlier_rate': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    selector = GreedyFeatureSelector(budget=20, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    
    # Generate training data
    training_data = generate_training_data(data, bin_dict)
    
    print(f"✓ Training data generation works")
    print(f"  X shape: {training_data.X.shape}")
    print(f"  Y shape: {training_data.Y.shape}")
    print(f"  Features: {training_data.feature_names}")
    print(f"  Metrics: {training_data.metric_names}")
    print(f"  Sample X: {training_data.X[0]}")
    print(f"  Sample Y: {training_data.Y[0]}")
    
    assert training_data.n_samples == len(data)
    assert training_data.n_features == bin_dict.n_features
    assert training_data.n_metrics > 0


def test_training_data_save_load():
    """Test saving and loading training data"""
    np.random.seed(42)
    n = 200
    data = pd.DataFrame({
        'x': np.random.uniform(0, 100, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=8)
    
    partition = partitioner.partition_attribute(data, 'x')
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features({'x': partition}, data)
    
    selector = GreedyFeatureSelector(budget=10, registry=registry)
    result = selector.select_features(ranked, {'x': partition}, data)
    
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    training_data = generate_training_data(data, bin_dict)
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "training")
        save_training_data(training_data, path)
        
        loaded_data = load_training_data(path + ".npz")
        
        print(f"✓ Training data save/load works")
        print(f"  Original X shape: {training_data.X.shape}")
        print(f"  Loaded X shape: {loaded_data.X.shape}")
        
        assert np.array_equal(training_data.X, loaded_data.X)
        assert np.array_equal(training_data.Y, loaded_data.Y)
        assert training_data.feature_names == loaded_data.feature_names


def test_dictionary_summary():
    """Test summary generation"""
    np.random.seed(42)
    n = 150
    data = pd.DataFrame({
        'a': np.random.uniform(0, 100, n),
        'b': np.random.uniform(0, 50, n)
    })
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 1.0},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    selector = GreedyFeatureSelector(budget=20, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    
    summary = bin_dict.summary()
    
    print(f"✓ Dictionary summary generation works")
    print(f"\n{summary}")
    
    assert "Bin Dictionary Summary" in summary
    assert "Features:" in summary


if __name__ == '__main__':
    test_bin_dictionary_creation()
    test_bin_id_lookup()
    test_bin_vector()
    test_batch_bin_vectors()
    test_quality_vector_retrieval()
    test_training_data_generation()
    test_training_data_save_load()
    test_dictionary_summary()
    
    print("\n" + "="*70)
    print("✅ All WP 7 tests passed!")
    print("="*70)
