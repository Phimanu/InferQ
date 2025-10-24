"""
WP 9: Online Quality Monitoring

Script for real-time data quality monitoring using a pre-trained learned index.
Simulates streaming data and generates quality alerts based on predictions.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

from inferq.learned_index import QualityIndex


@dataclass
class Alert:
    """Quality alert for a data tuple"""
    timestamp: float
    tuple_id: int
    metric: str
    predicted_value: float
    threshold: float
    severity: str  # 'warning' or 'critical'
    
    def __str__(self):
        return (f"[{self.severity.upper()}] Tuple {self.tuple_id}: "
                f"{self.metric}={self.predicted_value:.4f} "
                f"(threshold: {self.threshold:.4f})")


class QualityMonitor:
    """Online quality monitoring system"""
    
    def __init__(
        self,
        index: QualityIndex,
        alert_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        log_file: Optional[str] = None
    ):
        """
        Initialize monitor.
        
        Args:
            index: Trained QualityIndex
            alert_thresholds: Dict mapping metric names to {'warning': val, 'critical': val}
            log_file: Path to log alerts (optional)
        """
        self.index = index
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self.log_file = log_file
        
        # Statistics
        self.n_processed = 0
        self.n_alerts = 0
        self.alert_history: List[Alert] = []
        
        if log_file:
            # Create log file with header
            with open(log_file, 'w') as f:
                f.write("timestamp,tuple_id,metric,predicted_value,threshold,severity\n")
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default alert thresholds for common metrics"""
        return {
            'completeness': {'warning': 0.95, 'critical': 0.90},
            'outlier_rate': {'warning': 0.05, 'critical': 0.10},
            'duplicate_rate': {'warning': 0.05, 'critical': 0.10},
            'consistency_score': {'warning': 0.95, 'critical': 0.90},
            'constraint_satisfaction': {'warning': 0.95, 'critical': 0.90},
            'overall_quality': {'warning': 0.90, 'critical': 0.80},
        }
    
    def process_tuple(
        self,
        tuple_data: pd.Series,
        tuple_id: Optional[int] = None
    ) -> Tuple[Dict[str, float], List[Alert]]:
        """
        Process single data tuple and generate alerts.
        
        Args:
            tuple_data: Data tuple as pandas Series
            tuple_id: Optional tuple identifier
            
        Returns:
            (predictions, alerts) tuple
        """
        if tuple_id is None:
            tuple_id = self.n_processed
        
        # Get predictions
        predictions = self.index.predict(tuple_data)
        
        # Check for quality issues
        alerts = []
        timestamp = time.time()
        
        for metric_name, predicted_value in predictions.items():
            if metric_name not in self.alert_thresholds:
                continue
            
            thresholds = self.alert_thresholds[metric_name]
            
            # Check critical threshold
            if 'critical' in thresholds:
                critical = thresholds['critical']
                # For rates (higher is worse), check if above threshold
                if 'rate' in metric_name.lower():
                    if predicted_value > critical:
                        alert = Alert(
                            timestamp=timestamp,
                            tuple_id=tuple_id,
                            metric=metric_name,
                            predicted_value=predicted_value,
                            threshold=critical,
                            severity='critical'
                        )
                        alerts.append(alert)
                        continue
                # For scores (lower is worse), check if below threshold
                else:
                    if predicted_value < critical:
                        alert = Alert(
                            timestamp=timestamp,
                            tuple_id=tuple_id,
                            metric=metric_name,
                            predicted_value=predicted_value,
                            threshold=critical,
                            severity='critical'
                        )
                        alerts.append(alert)
                        continue
            
            # Check warning threshold
            if 'warning' in thresholds:
                warning = thresholds['warning']
                # For rates
                if 'rate' in metric_name.lower():
                    if predicted_value > warning:
                        alert = Alert(
                            timestamp=timestamp,
                            tuple_id=tuple_id,
                            metric=metric_name,
                            predicted_value=predicted_value,
                            threshold=warning,
                            severity='warning'
                        )
                        alerts.append(alert)
                # For scores
                else:
                    if predicted_value < warning:
                        alert = Alert(
                            timestamp=timestamp,
                            tuple_id=tuple_id,
                            metric=metric_name,
                            predicted_value=predicted_value,
                            threshold=warning,
                            severity='warning'
                        )
                        alerts.append(alert)
        
        # Update statistics
        self.n_processed += 1
        self.n_alerts += len(alerts)
        self.alert_history.extend(alerts)
        
        # Log alerts
        if self.log_file and alerts:
            with open(self.log_file, 'a') as f:
                for alert in alerts:
                    f.write(f"{alert.timestamp},{alert.tuple_id},{alert.metric},"
                           f"{alert.predicted_value},{alert.threshold},{alert.severity}\n")
        
        return predictions, alerts
    
    def process_batch(
        self,
        batch_data: pd.DataFrame,
        verbose: bool = True
    ) -> Dict:
        """
        Process batch of tuples.
        
        Args:
            batch_data: DataFrame of tuples
            verbose: Print progress
            
        Returns:
            Summary statistics
        """
        start_time = time.time()
        all_predictions = []
        all_alerts = []
        
        for idx, row in batch_data.iterrows():
            predictions, alerts = self.process_tuple(row, tuple_id=idx)
            all_predictions.append(predictions)
            all_alerts.extend(alerts)
            
            if verbose and alerts:
                for alert in alerts:
                    print(f"  {alert}")
        
        elapsed = time.time() - start_time
        
        summary = {
            'n_tuples': len(batch_data),
            'n_alerts': len(all_alerts),
            'alert_rate': len(all_alerts) / len(batch_data) if len(batch_data) > 0 else 0,
            'processing_time': elapsed,
            'throughput': len(batch_data) / elapsed if elapsed > 0 else 0,
        }
        
        if verbose:
            print(f"\n  Processed {len(batch_data)} tuples in {elapsed:.2f}s")
            print(f"  Throughput: {summary['throughput']:.0f} tuples/sec")
            print(f"  Alerts: {len(all_alerts)} ({summary['alert_rate']:.1%})")
        
        return summary
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        stats = {
            'n_processed': self.n_processed,
            'n_alerts': self.n_alerts,
            'alert_rate': self.n_alerts / self.n_processed if self.n_processed > 0 else 0,
        }
        
        # Alert breakdown by severity
        n_critical = sum(1 for a in self.alert_history if a.severity == 'critical')
        n_warning = sum(1 for a in self.alert_history if a.severity == 'warning')
        
        stats['n_critical'] = n_critical
        stats['n_warning'] = n_warning
        
        # Alert breakdown by metric
        metric_counts = {}
        for alert in self.alert_history:
            metric_counts[alert.metric] = metric_counts.get(alert.metric, 0) + 1
        
        stats['alerts_by_metric'] = metric_counts
        
        return stats


def simulate_stream(
    data: pd.DataFrame,
    batch_size: int = 100,
    delay: float = 0.0
) -> pd.DataFrame:
    """
    Simulate streaming data by yielding batches.
    
    Args:
        data: Full dataset
        batch_size: Tuples per batch
        delay: Delay between batches (seconds)
        
    Yields:
        Batches of data
    """
    n_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch = data.iloc[start_idx:end_idx]
        
        if delay > 0:
            time.sleep(delay)
        
        yield batch


def main():
    parser = argparse.ArgumentParser(
        description="Monitor data quality using learned index"
    )
    
    parser.add_argument(
        'index',
        type=str,
        help='Path to trained index (.pkl file)'
    )
    
    parser.add_argument(
        'data',
        type=str,
        help='Path to data file to monitor (CSV)'
    )
    
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=100,
        help='Tuples per batch (default: 100)'
    )
    
    parser.add_argument(
        '-d', '--delay',
        type=float,
        default=0.0,
        help='Delay between batches in seconds (default: 0.0)'
    )
    
    parser.add_argument(
        '-t', '--thresholds',
        type=str,
        help='Path to JSON file with custom alert thresholds'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        help='Path to log file for alerts'
    )
    
    parser.add_argument(
        '-n', '--max-tuples',
        type=int,
        help='Maximum tuples to process (default: all)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Load index
    if verbose:
        print("="*70)
        print("InferQ: Online Quality Monitoring")
        print("="*70)
        print(f"\n[1/3] Loading index from {args.index}...")
    
    index = QualityIndex.load(args.index)
    
    if verbose:
        print(f"   Features: {len(index.bin_dictionary.feature_order)}")
        print(f"   Metrics: {', '.join(index.metric_names)}")
    
    # Load data
    if verbose:
        print(f"\n[2/3] Loading data from {args.data}...")
    
    data = pd.read_csv(args.data)
    
    # Filter to only features used by index
    available_cols = [col for col in index.bin_dictionary.feature_order if col in data.columns]
    data = data[available_cols]
    
    if args.max_tuples:
        data = data.head(args.max_tuples)
    
    if verbose:
        print(f"   Loaded: {len(data)} tuples")
        print(f"   Features: {', '.join(data.columns)}")
    
    # Load custom thresholds
    alert_thresholds = None
    if args.thresholds:
        with open(args.thresholds) as f:
            alert_thresholds = json.load(f)
        if verbose:
            print(f"   Loaded custom thresholds from {args.thresholds}")
    
    # Initialize monitor
    monitor = QualityMonitor(
        index=index,
        alert_thresholds=alert_thresholds,
        log_file=args.log
    )
    
    # Start monitoring
    if verbose:
        print(f"\n[3/3] Monitoring stream (batch_size={args.batch_size})...")
        print()
    
    start_time = time.time()
    batch_count = 0
    
    for batch in simulate_stream(data, batch_size=args.batch_size, delay=args.delay):
        batch_count += 1
        
        if verbose:
            print(f"Batch {batch_count} ({len(batch)} tuples):")
        
        monitor.process_batch(batch, verbose=verbose)
        
        if verbose:
            print()
    
    total_time = time.time() - start_time
    
    # Final statistics
    stats = monitor.get_statistics()
    
    if verbose:
        print("="*70)
        print("✅ Monitoring Complete")
        print("="*70)
        print(f"   Total tuples: {stats['n_processed']}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Throughput: {stats['n_processed'] / total_time:.0f} tuples/sec")
        print(f"   Total alerts: {stats['n_alerts']} ({stats['alert_rate']:.1%})")
        print(f"     - Critical: {stats['n_critical']}")
        print(f"     - Warning: {stats['n_warning']}")
        
        if stats['alerts_by_metric']:
            print(f"   Alerts by metric:")
            for metric, count in sorted(stats['alerts_by_metric'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"     - {metric}: {count}")
        
        if args.log:
            print(f"   Log saved to: {args.log}")
        
        print("="*70)


if __name__ == '__main__':
    main()
