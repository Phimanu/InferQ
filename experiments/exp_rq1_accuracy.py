"""
RQ1: Prediction Accuracy

Experiment 1: Index vs Ground Truth Accuracy
- Per-metric R² scores
- Overall prediction quality
- Error distribution analysis

Experiment 5: MTQD vs Baseline Discretization
- Compare multi-target vs single-metric discretization
- Homogeneity and Separation scores
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time


def experiment_1_accuracy(runner, datasets, registry):
    """
    Experiment 1: Measure prediction accuracy vs ground truth
    """
    print_header("Experiment 1: Prediction Accuracy")
    
    results = []
    per_metric_results = {}
    
    for dataset_name, data in datasets.items():
        print(f"\n  Processing {dataset_name}...")
        
        # Sample for faster computation
        sample_size = min(2000, len(data))
        data_sample = data.sample(n=sample_size, random_state=42)
        
        # Build index
        print(f"    Building index...")
        start = time.time()
        index, info = build_index_for_dataset(data_sample, budget=50, initial_bins=20, use_row_metrics=True)
        build_time = time.time() - start
        
        # Use the registry from the build process
        index_registry = info['registry']
        metrics_dict = info['metrics_dict']
        
        # Get predictions
        print(f"    Computing predictions...")
        pred_start = time.time()
        predictions = index.predict_batch(data_sample)
        pred_time = time.time() - pred_start
        
        # Compute ground truth (sample further for speed)
        gt_sample_size = min(500, len(data_sample))
        gt_indices = np.random.choice(len(data_sample), gt_sample_size, replace=False)
        gt_data = data_sample.iloc[gt_indices]
        
        print(f"    Computing ground truth ({gt_sample_size} samples)...")
        gt_start = time.time()
        ground_truth = compute_ground_truth(gt_data, index_registry, metrics_dict)
        gt_time = time.time() - gt_start
        
        # Align predictions with ground truth
        pred_aligned = predictions.iloc[gt_indices]
        
        # Filter to common columns only
        common_cols = [col for col in pred_aligned.columns if col in ground_truth.columns]
        pred_aligned = pred_aligned[common_cols]
        ground_truth_filtered = ground_truth[common_cols]
        
        # Compute metrics
        overall_r2 = r2_score(ground_truth_filtered, pred_aligned)
        overall_mae = mean_absolute_error(ground_truth_filtered, pred_aligned)
        overall_mse = mean_squared_error(ground_truth_filtered, pred_aligned)
        
        # Per-metric R²
        per_metric_r2 = {}
        for col in common_cols:
            per_metric_r2[col] = r2_score(ground_truth_filtered[col], pred_aligned[col])
        
        per_metric_results[dataset_name] = per_metric_r2
        
        results.append({
            'dataset': dataset_name,
            'n_samples': len(data_sample),
            'n_features': len(index.bin_dictionary.feature_order),
            'n_bins': index.bin_dictionary.total_bins,
            'r2_overall': overall_r2,
            'mae_overall': overall_mae,
            'mse_overall': overall_mse,
            'build_time': build_time,
            'pred_time': pred_time,
            'gt_time': gt_time,
            'speedup': gt_time / pred_time,
        })
        
        print(f"    R²: {overall_r2:.4f}, MAE: {overall_mae:.4f}")
        print(f"    Build: {build_time:.2f}s, Pred: {pred_time:.2f}s, GT: {gt_time:.2f}s")
    
    runner.results['exp1_accuracy'] = results
    runner.results['exp1_per_metric'] = per_metric_results
    
    # Visualization 1: Overall R² by dataset
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    df_results = pd.DataFrame(results)
    
    # R² scores
    ax1.bar(df_results['dataset'], df_results['r2_overall'], color='steelblue', alpha=0.8)
    ax1.axhline(y=0.95, color='r', linestyle='--', label='Target (0.95)')
    ax1.set_ylabel('R² Score')
    ax1.set_xlabel('Dataset')
    ax1.set_title('Overall Prediction Accuracy')
    ax1.legend()
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3)
    
    # MAE
    ax2.bar(df_results['dataset'], df_results['mae_overall'], color='coral', alpha=0.8)
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_xlabel('Dataset')
    ax2.set_title('Prediction Error')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    runner.save_figure('exp1_overall_accuracy')
    plt.close()
    
    # Visualization 2: Per-metric R² heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for heatmap
    all_metrics = set()
    for metrics in per_metric_results.values():
        all_metrics.update(metrics.keys())
    
    heatmap_data = []
    for dataset in per_metric_results.keys():
        row = [per_metric_results[dataset].get(m, np.nan) for m in sorted(all_metrics)]
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=list(per_metric_results.keys()),
        columns=sorted(all_metrics)
    )
    
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'R² Score'},
                ax=ax)
    ax.set_title('Per-Metric Prediction Accuracy')
    ax.set_xlabel('Quality Metric')
    ax.set_ylabel('Dataset')
    
    plt.tight_layout()
    runner.save_figure('exp1_per_metric_accuracy')
    plt.close()
    
    print("\n  ✅ Experiment 1 complete")
    return results


def experiment_5_mtqd_comparison(runner, datasets, registry):
    """
    Experiment 5: Compare MTQD vs baseline discretization methods
    """
    print_header("Experiment 5: Multi-Target Discretization")
    
    from inferq.quality_assessment import compute_homogeneity, compute_separation
    
    results = []
    
    for dataset_name, data in datasets.items():
        print(f"\n  Processing {dataset_name}...")
        
        # Sample for speed
        sample_size = min(3000, len(data))
        data_sample = data.sample(n=sample_size, random_state=42)
        
        # Test first 3 attributes only for speed
        test_attrs = data_sample.columns[:min(3, len(data_sample.columns))]
        
        partitioner = InitialPartitioner(registry, n_bins=30)
        
        for attr in test_attrs:
            print(f"    Testing {attr}...")
            
            # Initial partition (equal-frequency)
            initial_partition = partitioner.partition_attribute(data_sample, attr)
            
            # Baseline 1: No merging (equal-frequency)
            h_baseline = compute_homogeneity(initial_partition, data_sample)
            s_baseline = compute_separation(initial_partition)
            
            # MTQD with merging
            adaptive = AdaptiveMTQD(registry=registry, max_iterations=3)
            mtqd_partition, _ = adaptive.discretize_with_assessment(initial_partition, data_sample)
            h_mtqd = compute_homogeneity(mtqd_partition, data_sample)
            s_mtqd = compute_separation(mtqd_partition)
            
            # Aggressive merging (higher threshold allows more merging)
            mtqd_aggressive = MTQD(registry=registry, merge_threshold=0.1)
            mtqd_agg_partition = mtqd_aggressive.discretize(initial_partition, data_sample)
            h_agg = compute_homogeneity(mtqd_agg_partition, data_sample)
            s_agg = compute_separation(mtqd_agg_partition)
            
            results.append({
                'dataset': dataset_name,
                'attribute': attr,
                'method': 'Equal-Freq (No Merge)',
                'n_bins': len(initial_partition.bins),
                'homogeneity': h_baseline,
                'separation': s_baseline,
            })
            
            results.append({
                'dataset': dataset_name,
                'attribute': attr,
                'method': 'MTQD (Adaptive)',
                'n_bins': len(mtqd_partition.bins),
                'homogeneity': h_mtqd,
                'separation': s_mtqd,
            })
            
            results.append({
                'dataset': dataset_name,
                'attribute': attr,
                'method': 'MTQD (Aggressive)',
                'n_bins': len(mtqd_agg_partition.bins),
                'homogeneity': h_agg,
                'separation': s_agg,
            })
    
    runner.results['exp5_mtqd'] = results
    
    # Visualization: Homogeneity vs Separation scatter
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, (dataset_name, ax) in enumerate(zip(datasets.keys(), axes)):
        df_ds = df_results[df_results['dataset'] == dataset_name]
        
        for method in df_ds['method'].unique():
            df_method = df_ds[df_ds['method'] == method]
            ax.scatter(df_method['homogeneity'], df_method['separation'],
                      label=method, s=100, alpha=0.7)
        
        ax.set_xlabel('Homogeneity (lower is better)')
        ax.set_ylabel('Separation (higher is better)')
        ax.set_title(f'{dataset_name}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    runner.save_figure('exp5_homogeneity_separation')
    plt.close()
    
    # Visualization 2: Number of bins
    fig, ax = plt.subplots(figsize=(10, 5))
    
    df_pivot = df_results.pivot_table(
        values='n_bins',
        index=['dataset', 'attribute'],
        columns='method',
        aggfunc='mean'
    )
    
    df_pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Number of Bins')
    ax.set_xlabel('Dataset / Attribute')
    ax.set_title('Bin Count Comparison')
    ax.legend(title='Method')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    runner.save_figure('exp5_bin_counts')
    plt.close()
    
    print("\n  ✅ Experiment 5 complete")
    return results


# Helper functions
from run_experiments import (
    print_header, build_index_for_dataset,
    compute_ground_truth
)

# Additional imports
from inferq.partitioning import InitialPartitioner
from inferq.quality_assessment import AdaptiveMTQD
from inferq.discretization import MTQD
