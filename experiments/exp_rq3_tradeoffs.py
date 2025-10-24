"""
RQ3: Trade-offs

Experiment 4: Index Size vs Accuracy
- Budget (B_max) vs R² score
- Pareto frontier analysis
- Sweet spot identification

Experiment 9: Ablation Study
- Full InferQ vs variants
- Component importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def experiment_4_tradeoffs(runner, datasets, registry):
    """
    Experiment 4: Budget vs accuracy trade-off
    """
    print_header("Experiment 4: Size-Accuracy Trade-offs")
    
    results = []
    
    # Test different budgets
    budgets = [10, 25, 50, 100, 200]
    
    for dataset_name, data in datasets.items():
        print(f"\n  Processing {dataset_name}...")
        
        # Sample for speed
        sample_size = min(3000, len(data))
        data_sample = data.sample(n=sample_size, random_state=42)
        
        for budget in budgets:
            print(f"    Budget: {budget} bins...")
            
            try:
                # Build index
                start = time.time()
                index, info = build_index_for_dataset(
                    data_sample, 
                    budget=budget, 
                    initial_bins=max(20, budget // 3),
                    use_row_metrics=True
                )
                build_time = time.time() - start
                
                # Quick accuracy check on test set
                test_size = min(200, len(data_sample) // 4)
                test_data = data_sample.sample(n=test_size, random_state=43)
                
                predictions = index.predict_batch(test_data)
                ground_truth = compute_ground_truth(test_data, info['registry'], info['metrics_dict'])
                
                # Filter to common columns
                common_cols = [col for col in predictions.columns if col in ground_truth.columns]
                predictions_filtered = predictions[common_cols]
                ground_truth_filtered = ground_truth[common_cols]
                
                from sklearn.metrics import r2_score, mean_absolute_error
                r2 = r2_score(ground_truth_filtered, predictions_filtered)
                mae = mean_absolute_error(ground_truth_filtered, predictions_filtered)
                
                results.append({
                    'dataset': dataset_name,
                    'budget': budget,
                    'n_features': info['n_features_selected'],
                    'n_bins_used': info['n_bins'],
                    'r2': r2,
                    'mae': mae,
                    'build_time': build_time,
                    'budget_utilization': info['n_bins'] / budget,
                })
                
                print(f"      Features: {info['n_features_selected']}, "
                      f"Bins: {info['n_bins']}/{budget}, R²: {r2:.4f}")
                
            except Exception as e:
                print(f"      Failed: {e}")
                continue
    
    runner.results['exp4_tradeoffs'] = results
    
    if not results:
        print("  ⚠️  No successful runs to visualize")
        return
    
    # Visualization 1: R² vs Budget (Pareto frontier)
    df_results = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    for dataset in df_results['dataset'].unique():
        df_ds = df_results[df_results['dataset'] == dataset]
        ax1.plot(df_ds['budget'], df_ds['r2'],
                marker='o', label=dataset, linewidth=2, markersize=8)
    
    ax1.axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='90% threshold')
    ax1.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95% threshold')
    ax1.set_xlabel('Budget (total bins)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Accuracy vs Index Size')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0.5, 1.05])
    
    # MAE vs Budget
    for dataset in df_results['dataset'].unique():
        df_ds = df_results[df_results['dataset'] == dataset]
        ax2.plot(df_ds['budget'], df_ds['mae'],
                marker='o', label=dataset, linewidth=2, markersize=8)
    
    ax2.set_xlabel('Budget (total bins)')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Prediction Error vs Index Size')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    runner.save_figure('exp4_budget_accuracy')
    plt.close()
    
    # Visualization 2: Feature count vs Budget
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for dataset in df_results['dataset'].unique():
        df_ds = df_results[df_results['dataset'] == dataset]
        ax.plot(df_ds['budget'], df_ds['n_features'],
                marker='o', label=dataset, linewidth=2, markersize=8)
    
    ax.set_xlabel('Budget (total bins)')
    ax.set_ylabel('Number of Features Selected')
    ax.set_title('Feature Selection vs Budget')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    runner.save_figure('exp4_budget_features')
    plt.close()
    
    print("\n  ✅ Experiment 4 complete")
    return results


def experiment_9_ablation(runner, datasets, registry):
    """
    Experiment 9: Ablation study - component importance
    """
    print_header("Experiment 9: Ablation Study")
    
    from inferq.partitioning import InitialPartitioner
    from inferq.feature_scoring import FeatureScorer, get_default_quality_thresholds
    from inferq.feature_selection import GreedyFeatureSelector
    from inferq.bin_dictionary import BinDictionary, generate_training_data
    from inferq.learned_index import QualityIndex
    
    results = []
    
    # Pick one representative dataset
    dataset_name = list(datasets.keys())[0]
    data = datasets[dataset_name]
    
    print(f"\n  Using {dataset_name} dataset")
    
    sample_size = min(3000, len(data))
    data_sample = data.sample(n=sample_size, random_state=42)
    
    # Shared setup
    budget = 50
    initial_bins = 20
    
    # Variant 1: Full InferQ (baseline)
    print(f"\n    Variant 1: Full InferQ...")
    start = time.time()
    index_full, info_full = build_index_for_dataset(data_sample, budget=budget, initial_bins=initial_bins, use_row_metrics=True)
    time_full = time.time() - start
    
    results.append({
        'variant': 'Full InferQ',
        'r2': info_full['test_r2'],
        'mse': info_full['test_mse'],
        'n_features': info_full['n_features_selected'],
        'n_bins': info_full['n_bins'],
        'build_time': time_full,
    })
    
    # Variant 2: No Adaptive Granularity (fixed bin allocation)
    print(f"    Variant 2: No Adaptive Granularity...")
    try:
        start = time.time()
        
        partitioner = InitialPartitioner(registry, n_bins=initial_bins)
        initial_partitions = {}
        for attr in data_sample.columns:
            initial_partitions[attr] = partitioner.partition_attribute(data_sample, attr)
        
        # No MTQD - use initial partitions
        final_partitions = initial_partitions
        
        scorer = FeatureScorer(
            metric_weights={'completeness': 0.3, 'outlier_rate': 0.3, 
                           'duplicate_rate': 0.2, 'consistency_score': 0.2},
            quality_thresholds=get_default_quality_thresholds(),
            alpha=0.7, beta=0.3
        )
        ranked_features = scorer.score_features(final_partitions, data_sample)
        
        # Fixed allocation: divide budget equally
        bins_per_feature = budget // len(ranked_features[:5])  # Use top 5 features
        from inferq.feature_selection import SelectedFeature
        selected = [
            SelectedFeature(f.attribute, f.partition, min(len(f.partition.bins), bins_per_feature))
            for f in ranked_features[:5]
        ][:budget // bins_per_feature]
        
        bin_dict = BinDictionary.from_selected_features(selected)
        training_data = generate_training_data(data_sample, bin_dict)
        index_nograd = QualityIndex.from_training_data(
            training_data, bin_dict,
            n_estimators=50, max_depth=15, test_size=0.2, verbose=False
        )
        
        time_nograd = time.time() - start
        
        results.append({
            'variant': 'No Adaptive Granularity',
            'r2': float(index_nograd.metrics.test_r2),
            'mse': float(index_nograd.metrics.test_mse),
            'n_features': len(selected),
            'n_bins': bin_dict.total_bins,
            'build_time': time_nograd,
        })
    except Exception as e:
        print(f"      Failed: {e}")
    
    # Variant 3: Fewer trees (25 instead of 50)
    print(f"    Variant 3: Fewer Trees (25)...")
    try:
        start = time.time()
        
        partitioner = InitialPartitioner(registry, n_bins=initial_bins)
        initial_partitions = {}
        for attr in data_sample.columns:
            initial_partitions[attr] = partitioner.partition_attribute(data_sample, attr)
        
        from inferq.quality_assessment import AdaptiveMTQD
        adaptive = AdaptiveMTQD(registry=registry, max_iterations=3)
        final_partitions = {}
        for attr, initial in initial_partitions.items():
            final, _ = adaptive.discretize_with_assessment(initial, data_sample)
            final_partitions[attr] = final
        
        scorer = FeatureScorer(
            metric_weights={'completeness': 0.3, 'outlier_rate': 0.3, 
                           'duplicate_rate': 0.2, 'consistency_score': 0.2},
            quality_thresholds=get_default_quality_thresholds(),
            alpha=0.7, beta=0.3
        )
        ranked_features = scorer.score_features(final_partitions, data_sample)
        selector = GreedyFeatureSelector(budget=budget, registry=registry, min_bins_per_feature=3)
        result = selector.select_features(ranked_features, final_partitions, data_sample)
        
        bin_dict = BinDictionary.from_selected_features(result.selected_features)
        training_data = generate_training_data(data_sample, bin_dict)
        
        index_fewtrees = QualityIndex.from_training_data(
            training_data, bin_dict,
            n_estimators=25, max_depth=15, test_size=0.2, verbose=False  # 25 trees
        )
        
        time_fewtrees = time.time() - start
        
        results.append({
            'variant': 'Fewer Trees (25)',
            'r2': float(index_fewtrees.metrics.test_r2),
            'mse': float(index_fewtrees.metrics.test_mse),
            'n_features': len(result.selected_features),
            'n_bins': bin_dict.total_bins,
            'build_time': time_fewtrees,
        })
    except Exception as e:
        print(f"      Failed: {e}")
    
    # Variant 4: IG-only (no QDP, α=1.0, β=0.0)
    print(f"    Variant 4: IG-only Feature Selection...")
    try:
        start = time.time()
        
        partitioner = InitialPartitioner(registry, n_bins=initial_bins)
        initial_partitions = {}
        for attr in data_sample.columns:
            initial_partitions[attr] = partitioner.partition_attribute(data_sample, attr)
        
        adaptive = AdaptiveMTQD(registry=registry, max_iterations=3)
        final_partitions = {}
        for attr, initial in initial_partitions.items():
            final, _ = adaptive.discretize_with_assessment(initial, data_sample)
            final_partitions[attr] = final
        
        scorer = FeatureScorer(
            metric_weights={'completeness': 0.3, 'outlier_rate': 0.3, 
                           'duplicate_rate': 0.2, 'consistency_score': 0.2},
            quality_thresholds=get_default_quality_thresholds(),
            alpha=1.0, beta=0.0  # IG-only
        )
        ranked_features = scorer.score_features(final_partitions, data_sample)
        selector = GreedyFeatureSelector(budget=budget, registry=registry, min_bins_per_feature=3)
        result = selector.select_features(ranked_features, final_partitions, data_sample)
        
        bin_dict = BinDictionary.from_selected_features(result.selected_features)
        training_data = generate_training_data(data_sample, bin_dict)
        index_igonly = QualityIndex.from_training_data(
            training_data, bin_dict,
            n_estimators=50, max_depth=15, test_size=0.2, verbose=False
        )
        
        time_igonly = time.time() - start
        
        results.append({
            'variant': 'IG-only (no QDP)',
            'r2': float(index_igonly.metrics.test_r2),
            'mse': float(index_igonly.metrics.test_mse),
            'n_features': len(result.selected_features),
            'n_bins': bin_dict.total_bins,
            'build_time': time_igonly,
        })
    except Exception as e:
        print(f"      Failed: {e}")
    
    runner.results['exp9_ablation'] = results
    
    # Visualization: Component comparison
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # R² scores
    ax = axes[0]
    bars = ax.bar(range(len(df_results)), df_results['r2'], color='steelblue', alpha=0.8)
    bars[0].set_color('green')  # Highlight baseline
    ax.set_ylabel('R² Score')
    ax.set_xlabel('Variant')
    ax.set_title('Prediction Accuracy by Variant')
    ax.set_xticks(range(len(df_results)))
    ax.set_xticklabels(df_results['variant'], rotation=45, ha='right')
    ax.axhline(y=df_results.iloc[0]['r2'], color='g', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.8, 1.05])
    
    # Build time
    ax = axes[1]
    bars = ax.bar(range(len(df_results)), df_results['build_time'], color='coral', alpha=0.8)
    bars[0].set_color('green')
    ax.set_ylabel('Build Time (seconds)')
    ax.set_xlabel('Variant')
    ax.set_title('Construction Time by Variant')
    ax.set_xticks(range(len(df_results)))
    ax.set_xticklabels(df_results['variant'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Index size (bins)
    ax = axes[2]
    bars = ax.bar(range(len(df_results)), df_results['n_bins'], color='teal', alpha=0.8)
    bars[0].set_color('green')
    ax.set_ylabel('Number of Bins')
    ax.set_xlabel('Variant')
    ax.set_title('Index Size by Variant')
    ax.set_xticks(range(len(df_results)))
    ax.set_xticklabels(df_results['variant'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    runner.save_figure('exp9_ablation')
    plt.close()
    
    print("\n  ✅ Experiment 9 complete")
    return results


# Helper functions
from run_experiments import (
    print_header, build_index_for_dataset,
    compute_ground_truth
)
