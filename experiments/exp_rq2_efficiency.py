"""
RQ2: Efficiency

Experiment 2: Query Speed Comparison
- Single-tuple latency
- Batch query throughput
- Index vs ground truth computation

Experiment 3: Scalability Analysis
- Query time vs dataset size
- Index size vs dataset size
- Memory efficiency
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def experiment_2_speed(runner, datasets, registry):
    """
    Experiment 2: Query speed comparison
    """
    print_header("Experiment 2: Query Speed")
    
    results = []
    
    for dataset_name, data in datasets.items():
        print(f"\n  Processing {dataset_name}...")
        
        # Sample for index building
        sample_size = min(5000, len(data))
        data_sample = data.sample(n=sample_size, random_state=42)
        
        # Build index
        print(f"    Building index...")
        index, info = build_index_for_dataset(data_sample, budget=50, initial_bins=20, use_row_metrics=True)
        
        # Query samples
        query_sizes = [1, 10, 100, 1000]
        
        for query_size in query_sizes:
            if query_size > len(data_sample):
                continue
            
            query_data = data_sample.iloc[:query_size]
            
            # Measure index prediction time
            times_index = []
            for _ in range(5):  # 5 repetitions
                start = time.time()
                if query_size == 1:
                    _ = index.predict(query_data.iloc[0])
                else:
                    _ = index.predict_batch(query_data)
                times_index.append(time.time() - start)
            
            time_index = np.mean(times_index)
            
            # Measure ground truth computation time
            times_gt = []
            for _ in range(min(3, 5) if query_size > 100 else 5):  # Fewer reps for large queries
                start = time.time()
                _ = compute_ground_truth(query_data, registry)
                times_gt.append(time.time() - start)
            
            time_gt = np.mean(times_gt)
            
            # Calculate per-tuple latency
            latency_index = (time_index / query_size) * 1e6  # microseconds
            latency_gt = (time_gt / query_size) * 1e6
            
            speedup = time_gt / time_index if time_index > 0 else 0
            
            results.append({
                'dataset': dataset_name,
                'query_size': query_size,
                'time_index': time_index,
                'time_gt': time_gt,
                'latency_index_us': latency_index,
                'latency_gt_us': latency_gt,
                'speedup': speedup,
                'throughput_index': query_size / time_index if time_index > 0 else 0,
                'throughput_gt': query_size / time_gt if time_gt > 0 else 0,
            })
            
            print(f"    Query size {query_size}: {speedup:.1f}× speedup "
                  f"({latency_index:.1f}µs vs {latency_gt:.1f}µs per tuple)")
    
    runner.results['exp2_speed'] = results
    
    # Visualization 1: Speedup vs query size
    df_results = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    for dataset in df_results['dataset'].unique():
        df_ds = df_results[df_results['dataset'] == dataset]
        ax1.plot(df_ds['query_size'], df_ds['speedup'], 
                marker='o', label=dataset, linewidth=2)
    
    ax1.set_xlabel('Query Size (tuples)')
    ax1.set_ylabel('Speedup (×)')
    ax1.set_title('InferQ Speedup vs Query Size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Visualization 2: Latency per tuple
    for dataset in df_results['dataset'].unique():
        df_ds = df_results[df_results['dataset'] == dataset]
        ax2.plot(df_ds['query_size'], df_ds['latency_index_us'],
                marker='o', label=f'{dataset} (Index)', linewidth=2)
        ax2.plot(df_ds['query_size'], df_ds['latency_gt_us'],
                marker='s', linestyle='--', label=f'{dataset} (Ground Truth)', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Query Size (tuples)')
    ax2.set_ylabel('Latency per Tuple (µs)')
    ax2.set_title('Per-Tuple Latency')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    runner.save_figure('exp2_query_speed')
    plt.close()
    
    # Visualization 3: Throughput comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(df_results['dataset'].unique()))
    width = 0.35
    
    query_size = 1000  # Focus on batch queries
    df_batch = df_results[df_results['query_size'] == query_size]
    
    if len(df_batch) > 0:
        datasets_list = df_batch['dataset'].tolist()
        throughput_index = df_batch['throughput_index'].tolist()
        throughput_gt = df_batch['throughput_gt'].tolist()
        
        x = np.arange(len(datasets_list))
        ax.bar(x - width/2, throughput_index, width, label='Index', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, throughput_gt, width, label='Ground Truth', color='coral', alpha=0.8)
        
        ax.set_ylabel('Throughput (tuples/sec)')
        ax.set_xlabel('Dataset')
        ax.set_title(f'Throughput Comparison (batch size={query_size})')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_list)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    runner.save_figure('exp2_throughput')
    plt.close()
    
    print("\n  ✅ Experiment 2 complete")
    return results


def experiment_3_scalability(runner, datasets, registry):
    """
    Experiment 3: Scalability analysis
    """
    print_header("Experiment 3: Scalability")
    
    results = []
    
    # Use largest dataset for scalability test
    largest_dataset = max(datasets.items(), key=lambda x: len(x[1]))
    dataset_name, full_data = largest_dataset
    
    print(f"\n  Using {dataset_name} dataset ({len(full_data)} rows)")
    
    # Test different data sizes
    sizes = [1000, 5000, 10000, 20000, 50000]
    sizes = [s for s in sizes if s <= len(full_data)]
    
    for size in sizes:
        print(f"\n    Testing size: {size} rows...")
        
        data_subset = full_data.sample(n=size, random_state=42)
        
        # Build index
        start_build = time.time()
        index, info = build_index_for_dataset(data_subset, budget=50, initial_bins=20, use_row_metrics=True)
        build_time = time.time() - start_build
        
        # Measure query time (should be constant)
        query_sample = data_subset.sample(n=min(100, size), random_state=42)
        
        start_query = time.time()
        _ = index.predict_batch(query_sample)
        query_time = time.time() - start_query
        
        # Estimate index size
        import sys
        index_size_bytes = sys.getsizeof(index.model) + \
                          sum(sys.getsizeof(v) for v in index.bin_dictionary.numeric_boundaries.values()) + \
                          sum(sys.getsizeof(v) for v in index.bin_dictionary.categorical_maps.values())
        index_size_mb = index_size_bytes / (1024 * 1024)
        
        # Data size
        data_size_mb = data_subset.memory_usage(deep=True).sum() / (1024 * 1024)
        
        results.append({
            'dataset': dataset_name,
            'n_rows': size,
            'n_features': len(data_subset.columns),
            'build_time': build_time,
            'query_time': query_time,
            'query_latency_us': (query_time / len(query_sample)) * 1e6,
            'index_size_mb': index_size_mb,
            'data_size_mb': data_size_mb,
            'compression_ratio': data_size_mb / index_size_mb if index_size_mb > 0 else 0,
            'r2': info['test_r2'],
        })
        
        print(f"      Build: {build_time:.2f}s, Query: {query_time*1000:.1f}ms")
        print(f"      Index: {index_size_mb:.2f}MB, Data: {data_size_mb:.2f}MB "
              f"(compression: {data_size_mb/index_size_mb:.1f}×)")
    
    runner.results['exp3_scalability'] = results
    
    # Visualization 1: Build time vs data size
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Build time
    ax = axes[0, 0]
    ax.plot(df_results['n_rows'], df_results['build_time'], 
            marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Dataset Size (rows)')
    ax.set_ylabel('Build Time (seconds)')
    ax.set_title('Index Construction Time')
    ax.grid(alpha=0.3)
    
    # Query latency (should be flat)
    ax = axes[0, 1]
    ax.plot(df_results['n_rows'], df_results['query_latency_us'],
            marker='o', linewidth=2, markersize=8, color='coral')
    ax.axhline(y=df_results['query_latency_us'].mean(), 
              color='r', linestyle='--', label='Mean', alpha=0.7)
    ax.set_xlabel('Dataset Size (rows)')
    ax.set_ylabel('Query Latency (µs/tuple)')
    ax.set_title('Query Time (constant w.r.t. data size)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Index size
    ax = axes[1, 0]
    ax.plot(df_results['n_rows'], df_results['index_size_mb'],
            marker='o', linewidth=2, markersize=8, label='Index Size', color='green')
    ax.plot(df_results['n_rows'], df_results['data_size_mb'],
            marker='s', linewidth=2, markersize=8, label='Data Size', color='purple', alpha=0.7)
    ax.set_xlabel('Dataset Size (rows)')
    ax.set_ylabel('Size (MB)')
    ax.set_title('Memory Footprint')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Compression ratio
    ax = axes[1, 1]
    ax.plot(df_results['n_rows'], df_results['compression_ratio'],
            marker='o', linewidth=2, markersize=8, color='teal')
    ax.set_xlabel('Dataset Size (rows)')
    ax.set_ylabel('Compression Ratio (×)')
    ax.set_title('Data Compression')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    runner.save_figure('exp3_scalability')
    plt.close()
    
    print("\n  ✅ Experiment 3 complete")
    return results


# Helper functions
from run_experiments import (
    print_header, build_index_for_dataset,
    compute_ground_truth
)
