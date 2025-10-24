"""
Experimental Evaluation Framework for InferQ

RQ1: Accuracy (Exp 1, 5)
RQ2: Efficiency (Exp 2, 3)
RQ3: Trade-offs (Exp 4, 9)

Designed to run in ~30 minutes total with publication-ready figures.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# InferQ imports
from inferq.quality_metrics import get_default_registry
from inferq.partitioning import InitialPartitioner
from inferq.discretization import MTQD
from inferq.quality_assessment import AdaptiveMTQD
from inferq.feature_scoring import FeatureScorer, get_default_quality_thresholds
from inferq.feature_selection import GreedyFeatureSelector
from inferq.bin_dictionary import BinDictionary, generate_training_data
from inferq.learned_index import train_model, QualityIndex
from inferq.row_metrics import get_row_level_registry, fit_row_metrics


class ExperimentRunner:
    """Manages experiment execution and result collection"""
    
    def __init__(self, output_dir='experiments/results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def save_figure(self, name: str):
        """Save current figure"""
        path = self.figures_dir / f'{name}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {path}")
        
    def save_results(self):
        """Save all results to JSON"""
        path = self.output_dir / 'results.json'
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✅ Results saved: {path}")


def load_dataset(name: str, max_rows: int = None) -> pd.DataFrame:
    """Load and prepare dataset"""
    path = f'example_data/{name}.csv'
    df = pd.read_csv(path)
    
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out label columns
    numeric_cols = [c for c in numeric_cols if not any(
        x in c.lower() for x in ['label', 'target', 'class', 'price', 'arrived']
    )]
    
    df = df[numeric_cols]
    
    # Remove any remaining NaN
    df = df.dropna()
    
    return df


def compute_ground_truth(data: pd.DataFrame, registry, metrics_dict=None) -> pd.DataFrame:
    """Compute ground truth quality metrics for all rows"""
    metrics = []
    
    for idx in range(len(data)):
        row_data = data.iloc[idx:idx+1]
        # Compute metrics directly for this row
        row_metrics = {}
        for metric_name, metric_func in registry._metrics.items():
            try:
                value = metric_func.compute(row_data)
                row_metrics[metric_name] = value
            except:
                row_metrics[metric_name] = 0.0
        metrics.append(row_metrics)
    
    return pd.DataFrame(metrics)


def build_index_for_dataset(
    data: pd.DataFrame,
    budget: int = 50,
    initial_bins: int = 20,
    verbose: bool = False,
    use_row_metrics: bool = True
) -> Tuple[QualityIndex, Dict]:
    """Build InferQ index and return timing info"""
    
    start_time = time.time()
    
    # WP 1: Metrics
    if use_row_metrics:
        registry, metrics_dict = get_row_level_registry()
        # Fit metrics on training data
        fit_row_metrics(data, metrics_dict)
    else:
        registry = get_default_registry()
        metrics_dict = None
    
    # WP 2-4: Discretization
    partitioner = InitialPartitioner(registry, n_bins=initial_bins)
    initial_partitions = {}
    for attr in data.columns:
        initial_partitions[attr] = partitioner.partition_attribute(data, attr)
    
    adaptive = AdaptiveMTQD(registry=registry, max_iterations=3)
    final_partitions = {}
    for attr, initial in initial_partitions.items():
        final, _ = adaptive.discretize_with_assessment(initial, data)
        final_partitions[attr] = final
    
    # WP 5-6: Feature selection
    scorer = FeatureScorer(
        metric_weights={'completeness': 0.3, 'outlier_rate': 0.3, 
                       'duplicate_rate': 0.2, 'consistency_score': 0.2},
        quality_thresholds=get_default_quality_thresholds(),
        alpha=0.7, beta=0.3
    )
    ranked_features = scorer.score_features(final_partitions, data)
    
    selector = GreedyFeatureSelector(budget=budget, registry=registry, min_bins_per_feature=3)
    result = selector.select_features(ranked_features, final_partitions, data)
    
    # WP 7-8: Training
    bin_dict = BinDictionary.from_selected_features(result.selected_features)
    
    if use_row_metrics:
        # Generate training data with actual row-level ground truth
        from inferq.bin_dictionary import TrainingData
        X = bin_dict.batch_get_bin_vectors(data)
        Y_df = compute_ground_truth(data, registry, metrics_dict)
        Y = Y_df.values.astype(np.float32)
        
        training_data = TrainingData(
            X=X,
            Y=Y,
            feature_names=bin_dict.feature_order,
            metric_names=list(Y_df.columns)
        )
    else:
        training_data = generate_training_data(data, bin_dict)
    
    index = QualityIndex.from_training_data(
        training_data, bin_dict,
        n_estimators=50, max_depth=15,
        test_size=0.2, verbose=False
    )
    
    build_time = time.time() - start_time
    
    info = {
        'build_time': build_time,
        'n_features_selected': len(result.selected_features),
        'n_bins': bin_dict.total_bins,
        'test_r2': float(index.metrics.test_r2),
        'test_mse': float(index.metrics.test_mse),
        'registry': registry,
        'metrics_dict': metrics_dict,
    }
    
    return index, info


# Dataset metadata
DATASETS = {
    'adult': {'name': 'Adult', 'rows': 32561, 'cols': 6},
    'cardio': {'name': 'Cardio', 'rows': 70000, 'cols': 12},
    'flight-price': {'name': 'Flight', 'rows': 300153, 'cols': 11},
}


def print_header(title: str):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


if __name__ == '__main__':
    print("="*70)
    print("  InferQ: Experimental Evaluation")
    print("="*70)
    print(f"  Datasets: {len(DATASETS)}")
    print(f"  Target time: ~30 minutes")
    print("="*70)
    
    runner = ExperimentRunner()
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = {}
    for name, meta in DATASETS.items():
        # Limit sizes for speed
        max_rows = min(meta['rows'], 50000)  # Cap at 50K for experiments
        datasets[name] = load_dataset(name, max_rows=max_rows)
        print(f"   {meta['name']}: {len(datasets[name])} rows × {len(datasets[name].columns)} cols")
    
    print("\n✅ Ready to run experiments")
    print("\nStarting experiments...")
    print("\n⚠️  Using ROW-LEVEL metrics for accurate per-row quality assessment:")
    print("   1. row_completeness - fraction of non-null values")
    print("   2. row_range_conformance - fraction of values within expected ranges")
    print("   3. row_consistency - consistency of value patterns")
    
    overall_start = time.time()
    
    # Initialize row-level registry
    registry, metrics_dict = get_row_level_registry()
    
    # Import experiment modules
    from exp_rq1_accuracy import experiment_1_accuracy, experiment_5_mtqd_comparison
    from exp_rq2_efficiency import experiment_2_speed, experiment_3_scalability
    from exp_rq3_tradeoffs import experiment_4_tradeoffs, experiment_9_ablation
    
    # RQ1: Accuracy
    print_header("RQ1: PREDICTION ACCURACY")
    experiment_1_accuracy(runner, datasets, registry)
    experiment_5_mtqd_comparison(runner, datasets, registry)
    
    # RQ2: Efficiency
    print_header("RQ2: EFFICIENCY")
    experiment_2_speed(runner, datasets, registry)
    experiment_3_scalability(runner, datasets, registry)
    
    # RQ3: Trade-offs
    print_header("RQ3: TRADE-OFFS")
    experiment_4_tradeoffs(runner, datasets, registry)
    experiment_9_ablation(runner, datasets, registry)
    
    # Save all results
    runner.save_results()
    
    overall_time = time.time() - overall_start
    
    # Final summary
    print("\n" + "="*70)
    print("  EXPERIMENTAL EVALUATION COMPLETE")
    print("="*70)
    print(f"  Total time: {overall_time/60:.1f} minutes")
    print(f"  Results: {runner.output_dir / 'results.json'}")
    print(f"  Figures: {runner.figures_dir}")
    print("="*70)
    print("\nGenerated figures:")
    for fig in sorted(runner.figures_dir.glob('*.png')):
        print(f"  - {fig.name}")
    print("="*70)
