"""Diagnose why index accuracy is low and test improvements."""

import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/sc/home/philipp.hildebrandt/InferQ/src')

from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.discretization import MTQD
from inferq.quality_assessment import AdaptiveMTQD, compute_homogeneity, compute_separation
from inferq.feature_scoring import FeatureScorer, get_default_quality_thresholds
from inferq.feature_selection import GreedyFeatureSelector
from inferq.bin_dictionary import BinDictionary, generate_training_data
from inferq.learned_index import QualityIndex
import time

def compute_ground_truth_simple(data, registry):
    """Compute ground truth quality metrics for all rows"""
    metrics = []
    
    for idx in range(len(data)):
        row_data = data.iloc[idx:idx+1]
        row_metrics = {}
        for metric_name in registry.list_metrics():
            metric = registry.get(metric_name)
            if not metric.requires_config:
                try:
                    value = metric.compute(row_data)
                    row_metrics[metric_name] = value
                except:
                    row_metrics[metric_name] = 0.0
        metrics.append(row_metrics)
    
    return pd.DataFrame(metrics)

def test_configuration(data, registry, config_name, **kwargs):
    """Test a specific configuration and return accuracy."""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    
    # Extract parameters
    n_bins = kwargs.get('n_bins', 50)
    initial_bins = kwargs.get('initial_bins', 100)
    n_estimators = kwargs.get('n_estimators', 50)
    max_depth = kwargs.get('max_depth', 15)
    merge_threshold = kwargs.get('merge_threshold', 0.01)
    use_adaptive = kwargs.get('use_adaptive', True)
    
    print(f"  n_bins={n_bins}, initial_bins={initial_bins}")
    print(f"  n_estimators={n_estimators}, max_depth={max_depth}")
    print(f"  merge_threshold={merge_threshold}, use_adaptive={use_adaptive}")
    
    try:
        # Step 1: Initial partitioning
        start = time.time()
        partitioner = InitialPartitioner(registry, n_bins=initial_bins)
        initial_partitions = {}
        for attr in data.select_dtypes(include=[np.number]).columns:
            try:
                initial_partitions[attr] = partitioner.partition_attribute(data, attr)
            except:
                pass
        
        # Step 2: MTQD (optional adaptive)
        if use_adaptive:
            adaptive = AdaptiveMTQD(registry=registry, max_iterations=5)
            final_partitions = {}
            for attr, initial in initial_partitions.items():
                final, _ = adaptive.discretize_with_assessment(initial, data)
                final_partitions[attr] = final
        else:
            mtqd = MTQD(registry=registry, merge_threshold=merge_threshold)
            final_partitions = {}
            for attr, initial in initial_partitions.items():
                final_partitions[attr] = mtqd.discretize(initial, data)
        
        # Step 3: Feature scoring and selection
        scorer = FeatureScorer(
            metric_weights={'completeness': 0.3, 'outlier_rate': 0.3, 
                           'duplicate_rate': 0.2, 'consistency_score': 0.2},
            quality_thresholds=get_default_quality_thresholds(),
            alpha=0.7, beta=0.3
        )
        ranked_features = scorer.score_features(final_partitions, data)
        
        selector = GreedyFeatureSelector(budget=n_bins, registry=registry, min_bins_per_feature=3)
        result = selector.select_features(ranked_features, final_partitions, data)
        
        print(f"  Selected features: {len(result.selected_features)}")
        print(f"  Total bins used: {sum(f.n_bins for f in result.selected_features)}")
        
        # Step 4: Build index
        bin_dict = BinDictionary.from_selected_features(result.selected_features)
        training_data = generate_training_data(data, bin_dict)
        
        print(f"  Feature dimension: {len(bin_dict.feature_order)}")
        
        index = QualityIndex.from_training_data(
            training_data, bin_dict,
            n_estimators=n_estimators,
            max_depth=max_depth,
            test_size=0.2,
            verbose=False
        )
        
        build_time = time.time() - start
        
        # Step 5: Test accuracy
        test_size = min(500, len(data) // 4)
        test_data = data.sample(n=test_size, random_state=99)
        
        predictions = index.predict_batch(test_data)
        ground_truth = compute_ground_truth_simple(test_data, registry)
        
        # Match columns
        common_cols = [col for col in predictions.columns if col in ground_truth.columns]
        pred_filtered = predictions[common_cols]
        gt_filtered = ground_truth[common_cols]
        
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(gt_filtered, pred_filtered)
        mae = mean_absolute_error(gt_filtered, pred_filtered)
        
        # Also check model's internal test metrics
        model_r2 = index.metrics.test_r2
        model_mse = index.metrics.test_mse
        
        print(f"\n  ✅ Results:")
        print(f"     Prediction R²: {r2:.4f}")
        print(f"     Prediction MAE: {mae:.4f}")
        print(f"     Model Test R²: {model_r2:.4f}")
        print(f"     Model Test MSE: {model_mse:.4f}")
        print(f"     Build time: {build_time:.2f}s")
        
        return {
            'config': config_name,
            'r2': r2,
            'mae': mae,
            'model_r2': model_r2,
            'model_mse': model_mse,
            'build_time': build_time,
            'n_features': len(result.selected_features),
            'n_bins': sum(f.n_bins for f in result.selected_features),
        }
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*80)
    print("DIAGNOSING INDEX ACCURACY AND TESTING IMPROVEMENTS")
    print("="*80)
    
    # Load dataset
    df = pd.read_csv('/sc/home/philipp.hildebrandt/InferQ/example_data/adult.csv')
    print(f"\nDataset: {len(df)} rows, {len(df.select_dtypes(include=[np.number]).columns)} numeric columns")
    
    # Sample for faster testing
    sample_size = min(5000, len(df))
    data = df.sample(n=sample_size, random_state=42)
    print(f"Using sample: {len(data)} rows")
    
    registry = get_default_registry()
    
    results = []
    
    # Baseline: Current configuration
    result = test_configuration(
        data, registry, "Baseline (Current)",
        n_bins=50, initial_bins=100, n_estimators=50, max_depth=15,
        merge_threshold=0.01, use_adaptive=True
    )
    if result:
        results.append(result)
    
    # Improvement 1: More trees
    result = test_configuration(
        data, registry, "More Trees (100)",
        n_bins=50, initial_bins=100, n_estimators=100, max_depth=15,
        merge_threshold=0.01, use_adaptive=True
    )
    if result:
        results.append(result)
    
    # Improvement 2: Deeper trees
    result = test_configuration(
        data, registry, "Deeper Trees (depth=25)",
        n_bins=50, initial_bins=100, n_estimators=50, max_depth=25,
        merge_threshold=0.01, use_adaptive=True
    )
    if result:
        results.append(result)
    
    # Improvement 3: More bins
    result = test_configuration(
        data, registry, "More Bins (100)",
        n_bins=100, initial_bins=150, n_estimators=50, max_depth=15,
        merge_threshold=0.01, use_adaptive=True
    )
    if result:
        results.append(result)
    
    # Improvement 4: Finer initial discretization
    result = test_configuration(
        data, registry, "Finer Initial (200 bins)",
        n_bins=50, initial_bins=200, n_estimators=50, max_depth=15,
        merge_threshold=0.01, use_adaptive=True
    )
    if result:
        results.append(result)
    
    # Improvement 5: Stricter merging
    result = test_configuration(
        data, registry, "Stricter Merging (0.005)",
        n_bins=50, initial_bins=100, n_estimators=50, max_depth=15,
        merge_threshold=0.005, use_adaptive=True
    )
    if result:
        results.append(result)
    
    # Improvement 6: Combined (more trees + deeper + more bins)
    result = test_configuration(
        data, registry, "Combined (100 trees, depth=20, 100 bins)",
        n_bins=100, initial_bins=150, n_estimators=100, max_depth=20,
        merge_threshold=0.005, use_adaptive=True
    )
    if result:
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF IMPROVEMENTS")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    print("\n", df_results.to_string(index=False))
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best = df_results.loc[df_results['r2'].idxmax()]
    print(f"\n✅ Best configuration: {best['config']}")
    print(f"   R² Score: {best['r2']:.4f}")
    print(f"   MAE: {best['mae']:.4f}")
    print(f"   Build time: {best['build_time']:.2f}s")
    print(f"   Features: {best['n_features']}, Bins: {best['n_bins']}")
    
    baseline = df_results[df_results['config'] == 'Baseline (Current)'].iloc[0]
    improvement = (best['r2'] - baseline['r2']) / baseline['r2'] * 100
    time_increase = (best['build_time'] - baseline['build_time']) / baseline['build_time'] * 100
    
    print(f"\n   Improvement over baseline:")
    print(f"   - R² increase: {improvement:.1f}%")
    print(f"   - Build time increase: {time_increase:.1f}%")

if __name__ == '__main__':
    main()
