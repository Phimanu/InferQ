"""
WP 2 Example: Initial Partitioning and Annotation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from inferq import get_default_registry, InitialPartitioner, summarize_partitions


def main():
    print("=" * 70)
    print("WP 2: Initial Partitioning and Annotation Example")
    print("=" * 70)
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'salary': np.random.randint(30000, 120000, n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'rating': np.random.rand(n_samples) * 5,
        'department': np.random.choice(['Sales', 'IT', 'HR'], n_samples)
    })
    
    # Add some quality issues
    df.loc[np.random.choice(n_samples, 50, replace=False), 'salary'] = None
    df.loc[np.random.choice(n_samples, 30, replace=False), 'rating'] = None
    
    print(f"\n1. Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Numeric attributes: {df.select_dtypes(include=[np.number]).columns.tolist()}")
    
    # Initialize partitioner
    registry = get_default_registry()
    partitioner = InitialPartitioner(registry, n_bins=20)
    
    print(f"\n2. Creating initial partitions (k={partitioner.n_bins} bins per attribute)...")
    
    # Partition all numeric attributes
    partitions = partitioner.partition_all_attributes(df)
    
    print(f"   Partitioned {len(partitions)} attributes")
    
    # Display summary
    print("\n3. Partition Summary:")
    print("-" * 70)
    summary = summarize_partitions(partitions)
    print(summary.to_string(index=False))
    
    # Detailed view of one attribute
    print("\n4. Detailed View: 'age' attribute")
    print("-" * 70)
    age_partition = partitions['age']
    print(f"   Attribute: {age_partition.attribute_name}")
    print(f"   Number of bins: {len(age_partition.bins)}")
    print(f"   Bin boundaries: [{age_partition.bin_boundaries[0]:.1f}, ..., {age_partition.bin_boundaries[-1]:.1f}]")
    
    # Show first 5 bins
    print("\n   First 5 bins:")
    for i, bin in enumerate(age_partition.bins[:5]):
        print(f"   Bin {i}: [{bin.lower_bound:.1f}, {bin.upper_bound:.1f}), n={len(bin.indices)}")
    
    # Show quality vectors for first 3 bins
    print("\n5. Quality Vectors (first 3 bins of 'age'):")
    print("-" * 70)
    for i, bin in enumerate(age_partition.bins[:3]):
        print(f"\n   Bin {i}: [{bin.lower_bound:.1f}, {bin.upper_bound:.1f})")
        print(f"   Size: {len(bin.indices)} rows")
        print("   Quality Metrics:")
        for metric, value in sorted(bin.quality_vector.items())[:5]:
            print(f"      {metric}: {value:.3f}")
    
    # Compare quality across bins
    print("\n6. Quality Comparison Across All Bins:")
    print("-" * 70)
    
    completeness_values = [b.quality_vector.get('completeness', 0) 
                          for b in age_partition.bins]
    outlier_values = [b.quality_vector.get('outlier_rate', 0) 
                     for b in age_partition.bins]
    
    print(f"   Completeness: min={min(completeness_values):.3f}, "
          f"max={max(completeness_values):.3f}, "
          f"avg={np.mean(completeness_values):.3f}")
    print(f"   Outlier Rate: min={min(outlier_values):.3f}, "
          f"max={max(outlier_values):.3f}, "
          f"avg={np.mean(outlier_values):.3f}")
    
    # Show all attributes
    print("\n7. All Partitioned Attributes:")
    print("-" * 70)
    for attr_name in partitions.keys():
        partition = partitions[attr_name]
        avg_completeness = np.mean([b.quality_vector.get('completeness', 0) 
                                   for b in partition.bins])
        print(f"   {attr_name}: {len(partition.bins)} bins, "
              f"avg completeness={avg_completeness:.3f}")
    
    print("\n" + "=" * 70)
    print("✅ WP 2 Example Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
