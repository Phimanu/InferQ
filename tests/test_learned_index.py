"""
Tests for WP 8: Model Training and Index Class
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
from inferq.bin_dictionary import BinDictionary, generate_training_data
from inferq.learned_index import (
    train_model,
    save_model,
    load_model,
    QualityIndex
)


def setup_test_data(n=500):
    """Helper to create test data and training setup"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'a': np.random.uniform(0, 100, n),
        'b': np.random.uniform(0, 50, n),
        'c': np.random.uniform(0, 200, n)
    })
    
    # Add quality issues
    missing_mask = np.random.random(n) < 0.15
    data.loc[missing_mask, 'b'] = np.nan
    
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=10)
    
    partitions = {attr: partitioner.partition_attribute(data, attr) 
                  for attr in data.columns}
    
    scorer = FeatureScorer(
        metric_weights={'completeness': 0.5, 'outlier_rate': 0.5},
        quality_thresholds=get_default_quality_thresholds()
    )
    
    ranked = scorer.score_features(partitions, data)
    
    selector = GreedyFeatureSelector(budget=25, registry=registry)
    result = selector.select_features(ranked, partitions, data)
    
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    training_data = generate_training_data(data, bin_dict)
    
    return data, bin_dict, training_data


def test_model_training():
    """Test Random Forest model training"""
    print("\n" + "="*70)
    print("Testing WP 8: Model Training and Index Class")
    print("="*70)
    
    _, bin_dict, training_data = setup_test_data()
    
    # Train model
    model, metrics = train_model(
        training_data,
        n_estimators=50,
        test_size=0.3,
        verbose=False
    )
    
    print(f"✓ Model training works")
    print(f"  Test R²: {metrics.test_r2:.4f}")
    print(f"  Test MSE: {metrics.test_mse:.6f}")
    print(f"  Train samples: {int(training_data.n_samples * 0.7)}")
    print(f"  Test samples: {int(training_data.n_samples * 0.3)}")
    
    assert model is not None
    assert metrics.test_r2 >= 0  # R² can be negative for bad models
    assert metrics.test_mse >= 0


def test_model_save_load():
    """Test model serialization"""
    _, bin_dict, training_data = setup_test_data(n=300)
    
    model, metrics = train_model(
        training_data,
        n_estimators=30,
        verbose=False
    )
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model")
        save_model(model, path, bin_dict, metrics)
        
        loaded_model, loaded_dict, loaded_metrics = load_model(path + ".pkl")
        
        print(f"✓ Model save/load works")
        print(f"  Original estimators: {model.n_estimators}")
        print(f"  Loaded estimators: {loaded_model.n_estimators}")
        
        assert loaded_model.n_estimators == model.n_estimators
        assert loaded_dict is not None
        assert loaded_metrics.test_r2 == metrics.test_r2


def test_quality_index_creation():
    """Test QualityIndex creation"""
    _, bin_dict, training_data = setup_test_data()
    
    # Create index from training data
    index = QualityIndex.from_training_data(
        training_data,
        bin_dict,
        n_estimators=50,
        verbose=False
    )
    
    print(f"✓ QualityIndex creation works")
    print(f"  Features: {index.bin_dictionary.n_features}")
    print(f"  Metrics: {len(index.metric_names)}")
    print(f"  Model estimators: {index.model.n_estimators}")
    
    assert index.bin_dictionary is not None
    assert index.model is not None
    assert len(index.metric_names) > 0


def test_single_prediction():
    """Test single row prediction"""
    data, bin_dict, training_data = setup_test_data()
    
    index = QualityIndex.from_training_data(
        training_data,
        bin_dict,
        n_estimators=40,
        verbose=False
    )
    
    # Predict on sample row
    sample_row = data.iloc[0]
    prediction = index.predict(sample_row)
    
    print(f"✓ Single prediction works")
    print(f"  Input: {dict(sample_row)}")
    print(f"  Predicted metrics: {list(prediction.keys())}")
    print(f"  Sample predictions:")
    for metric, value in list(prediction.items())[:3]:
        print(f"    {metric}: {value:.4f}")
    
    assert isinstance(prediction, dict)
    assert len(prediction) > 0
    assert all(isinstance(v, float) for v in prediction.values())


def test_batch_prediction():
    """Test batch prediction"""
    data, bin_dict, training_data = setup_test_data()
    
    index = QualityIndex.from_training_data(
        training_data,
        bin_dict,
        n_estimators=40,
        verbose=False
    )
    
    # Predict on batch
    predictions = index.predict_batch(data.head(50))
    
    print(f"✓ Batch prediction works")
    print(f"  Input shape: {data.head(50).shape}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Columns: {list(predictions.columns)}")
    
    assert predictions.shape[0] == 50
    assert len(predictions.columns) == len(index.metric_names)


def test_prediction_with_bin_info():
    """Test prediction with bin information"""
    data, bin_dict, training_data = setup_test_data(n=200)
    
    index = QualityIndex.from_training_data(
        training_data,
        bin_dict,
        n_estimators=30,
        verbose=False
    )
    
    sample_row = data.iloc[0]
    result = index.predict_with_bin_info(sample_row)
    
    print(f"✓ Prediction with bin info works")
    print(f"  Bin vector: {result['bin_vector']}")
    print(f"  Features: {result['features']}")
    print(f"  Prediction keys: {list(result['prediction'].keys())}")
    
    assert 'prediction' in result
    assert 'bin_vector' in result
    assert 'features' in result
    assert len(result['bin_vector']) == index.bin_dictionary.n_features


def test_feature_importance():
    """Test feature importance extraction"""
    _, bin_dict, training_data = setup_test_data()
    
    index = QualityIndex.from_training_data(
        training_data,
        bin_dict,
        n_estimators=40,
        verbose=False
    )
    
    importances = index.get_feature_importance()
    
    print(f"✓ Feature importance extraction works")
    for feature, importance in importances.items():
        print(f"  {feature}: {importance:.4f}")
    
    assert len(importances) == index.bin_dictionary.n_features
    assert all(0 <= v <= 1 for v in importances.values())
    assert abs(sum(importances.values()) - 1.0) < 1e-6  # Should sum to 1


def test_index_save_load():
    """Test QualityIndex serialization"""
    _, bin_dict, training_data = setup_test_data(n=250)
    
    index = QualityIndex.from_training_data(
        training_data,
        bin_dict,
        n_estimators=30,
        verbose=False
    )
    
    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "index")
        index.save(path)
        
        loaded_index = QualityIndex.load(path + ".pkl")
        
        print(f"✓ Index save/load works")
        print(f"  Original features: {index.bin_dictionary.n_features}")
        print(f"  Loaded features: {loaded_index.bin_dictionary.n_features}")
        
        # Test that loaded index works
        data = pd.DataFrame({
            'a': [50.0],
            'b': [25.0],
            'c': [100.0]
        })
        
        pred1 = index.predict(data.iloc[0])
        pred2 = loaded_index.predict(data.iloc[0])
        
        # Predictions should be identical
        for metric in pred1.keys():
            assert abs(pred1[metric] - pred2[metric]) < 1e-6


def test_summary_generation():
    """Test summary generation"""
    _, bin_dict, training_data = setup_test_data()
    
    index = QualityIndex.from_training_data(
        training_data,
        bin_dict,
        n_estimators=40,
        verbose=False
    )
    
    summary = index.summary()
    
    print(f"✓ Summary generation works")
    print(f"\n{summary}")
    
    assert "Quality Index Summary" in summary
    assert "Features:" in summary
    assert "Model Performance:" in summary


def test_per_metric_performance():
    """Test per-metric evaluation"""
    _, bin_dict, training_data = setup_test_data()
    
    model, metrics = train_model(
        training_data,
        n_estimators=50,
        verbose=False
    )
    
    print(f"✓ Per-metric performance evaluation works")
    print(f"  Per-metric R² scores:")
    for metric, r2 in metrics.per_metric_r2.items():
        print(f"    {metric:<25} {r2:.4f}")
    
    assert len(metrics.per_metric_r2) > 0
    assert len(metrics.per_metric_mse) > 0


if __name__ == '__main__':
    test_model_training()
    test_model_save_load()
    test_quality_index_creation()
    test_single_prediction()
    test_batch_prediction()
    test_prediction_with_bin_info()
    test_feature_importance()
    test_index_save_load()
    test_summary_generation()
    test_per_metric_performance()
    
    print("\n" + "="*70)
    print("✅ All WP 8 tests passed!")
    print("="*70)
