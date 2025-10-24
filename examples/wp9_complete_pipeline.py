"""
WP 9 Example: Complete Pipeline - Offline Construction & Online Monitoring

This example demonstrates the full InferQ workflow:
1. Build index offline from historical data
2. Use index online for real-time monitoring
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from inferq.learned_index import QualityIndex

# Import the build_index function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from build_index import build_quality_index


def create_sample_data(n_samples=1000, anomaly_rate=0.05):
    """Create sample data with some quality issues"""
    np.random.seed(42)
    
    # Normal data
    n_normal = int(n_samples * (1 - anomaly_rate))
    normal_data = {
        'age': np.random.randint(20, 70, n_normal),
        'income': np.random.randint(20000, 100000, n_normal),
        'credit_score': np.random.randint(300, 850, n_normal),
    }
    
    # Anomalous data
    n_anomaly = n_samples - n_normal
    anomaly_data = {
        'age': np.concatenate([
            np.random.randint(20, 70, n_anomaly // 2),
            np.random.choice([999, -1], n_anomaly - n_anomaly // 2)  # Invalid values
        ]),
        'income': np.concatenate([
            np.random.randint(20000, 100000, n_anomaly // 2),
            np.random.choice([0, 999999], n_anomaly - n_anomaly // 2)  # Outliers
        ]),
        'credit_score': np.concatenate([
            np.random.randint(300, 850, n_anomaly // 2),
            np.random.choice([0, 999], n_anomaly - n_anomaly // 2)  # Outliers
        ]),
    }
    
    # Combine
    data = {
        'age': np.concatenate([normal_data['age'], anomaly_data['age']]),
        'income': np.concatenate([normal_data['income'], anomaly_data['income']]),
        'credit_score': np.concatenate([normal_data['credit_score'], anomaly_data['credit_score']]),
    }
    
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    print("="*70)
    print("WP 9: Complete Pipeline Demonstration")
    print("="*70)
    
    # =========================================================================
    # Step 1: Generate Training Data
    # =========================================================================
    print("\n[Step 1] Generating training data...")
    
    train_data = create_sample_data(n_samples=5000, anomaly_rate=0.03)
    
    # Save to CSV
    os.makedirs('temp_data', exist_ok=True)
    train_path = 'temp_data/train.csv'
    train_data.to_csv(train_path, index=False)
    
    print(f"   Generated {len(train_data)} training samples")
    print(f"   Features: {', '.join(train_data.columns)}")
    print(f"   Saved to: {train_path}")
    
    # =========================================================================
    # Step 2: Build Quality Index (Offline)
    # =========================================================================
    print("\n[Step 2] Building quality index (offline)...")
    
    index = build_quality_index(
        data_path=train_path,
        output_path='temp_data/quality_index',
        budget=30,
        n_estimators=50,
        max_depth=15,
        initial_bins=20,
        verbose=False  # Suppress detailed output for example
    )
    
    print(f"   ✅ Index built successfully")
    print(f"   Model R²: {index.metrics.test_r2:.4f}")
    print(f"   Features: {len(index.bin_dictionary.feature_order)}")
    print(f"   Metrics tracked: {len(index.metric_names)}")
    
    # =========================================================================
    # Step 3: Load Index for Online Monitoring
    # =========================================================================
    print("\n[Step 3] Loading index for online monitoring...")
    
    loaded_index = QualityIndex.load('temp_data/quality_index.pkl')
    
    print(f"   ✅ Index loaded")
    print(f"   Ready for predictions")
    
    # =========================================================================
    # Step 4: Simulate Online Monitoring
    # =========================================================================
    print("\n[Step 4] Simulating online monitoring stream...")
    
    # Generate new streaming data
    stream_data = create_sample_data(n_samples=100, anomaly_rate=0.10)
    
    print(f"   Processing {len(stream_data)} incoming tuples...")
    
    # Define alert thresholds
    thresholds = {
        'completeness': 0.95,
        'outlier_rate': 0.05,
    }
    
    # Monitor stream
    alerts = []
    predictions_list = []
    
    for idx, row in stream_data.iterrows():
        # Predict quality
        pred = loaded_index.predict(row)
        predictions_list.append(pred)
        
        # Check thresholds
        for metric, value in pred.items():
            if metric not in thresholds:
                continue
            
            threshold = thresholds[metric]
            
            # Alert logic
            if metric == 'outlier_rate' and value > threshold:
                alerts.append({
                    'tuple_id': idx,
                    'metric': metric,
                    'value': value,
                    'threshold': threshold
                })
            elif metric != 'outlier_rate' and value < threshold:
                alerts.append({
                    'tuple_id': idx,
                    'metric': metric,
                    'value': value,
                    'threshold': threshold
                })
    
    # =========================================================================
    # Step 5: Report Results
    # =========================================================================
    print(f"\n[Step 5] Monitoring results:")
    print(f"   Total tuples: {len(stream_data)}")
    print(f"   Alerts triggered: {len(alerts)} ({len(alerts)/len(stream_data):.1%})")
    
    if alerts:
        print(f"\n   Sample alerts:")
        for alert in alerts[:5]:
            print(f"      Tuple {alert['tuple_id']}: "
                  f"{alert['metric']}={alert['value']:.4f} "
                  f"(threshold: {alert['threshold']})")
        
        if len(alerts) > 5:
            print(f"      ... and {len(alerts) - 5} more")
    
    # Show quality distribution
    if predictions_list:
        pred_df = pd.DataFrame(predictions_list)
        print(f"\n   Quality metric statistics:")
        print(f"   {'Metric':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("   " + "-"*60)
        for col in pred_df.columns:
            print(f"   {col:<25} {pred_df[col].mean():<10.4f} "
                  f"{pred_df[col].std():<10.4f} {pred_df[col].min():<10.4f} "
                  f"{pred_df[col].max():<10.4f}")
    
    # =========================================================================
    # Cleanup
    # =========================================================================
    print(f"\n[Cleanup] Removing temporary files...")
    import shutil
    if os.path.exists('temp_data'):
        shutil.rmtree('temp_data')
    print(f"   ✅ Done")
    
    print("\n" + "="*70)
    print("✅ Pipeline demonstration complete!")
    print("="*70)
    print("\nKey takeaways:")
    print("  1. Index built offline from historical data")
    print("  2. Index loaded for online monitoring")
    print("  3. Streaming tuples processed in real-time")
    print("  4. Alerts generated based on quality predictions")
    print("  5. Fast prediction enables real-time monitoring")
    print("="*70)


if __name__ == '__main__':
    main()
