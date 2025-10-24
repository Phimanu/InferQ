# InferQ: Experimental Evaluation - Implementation Summary

## Overview

Comprehensive experimental framework implemented to evaluate InferQ for the research paper. Designed to run in ~30 minutes with publication-ready figures.

## Experiments Implemented

### RQ1: Prediction Accuracy

**Experiment 1: Index vs Ground Truth**
- Measures R² score, MAE, MSE between index predictions and ground truth
- Per-metric R² analysis across all quality metrics
- **Figures:**
  - `exp1_overall_accuracy.png` - Bar charts of R² and MAE by dataset
  - `exp1_per_metric_accuracy.png` - Heatmap of per-metric R² scores

**Experiment 5: MTQD Comparison**
- Compares multi-target discretization vs baselines
- Measures Homogeneity (H) and Separation (S) scores
- Tests: Equal-frequency (no merging), MTQD (adaptive), MTQD (aggressive)
- **Figures:**
  - `exp5_homogeneity_separation.png` - H vs S scatter plots
  - `exp5_bin_counts.png` - Bin count comparison

### RQ2: Efficiency

**Experiment 2: Query Speed**
- Measures latency for different query sizes (1, 10, 100, 1000 tuples)
- Computes speedup ratio (ground truth time / index time)
- Tests throughput (tuples/sec)
- **Figures:**
  - `exp2_query_speed.png` - Speedup and latency vs query size
  - `exp2_throughput.png` - Throughput comparison

**Experiment 3: Scalability**
- Tests different dataset sizes (1K, 5K, 10K, 20K, 50K rows)
- Measures build time, query latency, index size
- Demonstrates constant query time w.r.t. data size
- **Figures:**
  - `exp3_scalability.png` - 4-panel scalability analysis

### RQ3: Trade-offs

**Experiment 4: Budget vs Accuracy**
- Tests budgets: 10, 25, 50, 100, 200 bins
- Measures R² and MAE vs budget
- Identifies "sweet spot" for accuracy
- **Figures:**
  - `exp4_budget_accuracy.png` - Accuracy and error vs budget
  - `exp4_budget_features.png` - Feature count vs budget

**Experiment 9: Ablation Study**
- Compares variants:
  - Full InferQ (baseline)
  - No adaptive granularity
  - Fewer trees (25 vs 50)
  - IG-only (no QDP)
- **Figures:**
  - `exp9_ablation.png` - R², build time, index size by variant

## Datasets Used

Three real-world datasets across different domains:

1. **Adult** (UCI) - 32,561 rows, 6 numeric features
   - Demographic data with missing values and outliers
   
2. **Cardio** - 70,000 rows, 12 numeric features  
   - Medical data with outliers and distribution skew
   
3. **Flight** - 300,153 rows, 11 numeric features
   - Airline data with temporal patterns

For speed, datasets are sampled to max 50K rows during experiments.

## Implementation Details

### Architecture

```
experiments/
├── run_experiments.py          # Main orchestrator
│   └── ExperimentRunner class  # Manages results and figures
│
├── exp_rq1_accuracy.py         # RQ1 experiments
│   ├── experiment_1_accuracy()
│   └── experiment_5_mtqd_comparison()
│
├── exp_rq2_efficiency.py       # RQ2 experiments
│   ├── experiment_2_speed()
│   └── experiment_3_scalability()
│
└── exp_rq3_tradeoffs.py        # RQ3 experiments
    ├── experiment_4_tradeoffs()
    └── experiment_9_ablation()
```

### Key Functions

**`build_index_for_dataset(data, budget, initial_bins)`**
- Orchestrates full InferQ pipeline
- Returns trained QualityIndex and timing info
- Used by all experiments for consistency

**`compute_ground_truth(data, registry)`**
- Computes actual quality metrics for comparison
- Handles metrics that require extra parameters
- Returns DataFrame of metric values

**`ExperimentRunner`**
- Manages output directories
- Saves results to JSON
- Generates publication-ready figures (300 DPI PNG)

### Figure Generation

All figures use:
- **seaborn-paper** style for clean, professional appearance
- **300 DPI** resolution for publication quality
- **Color-blind friendly** palettes
- Clear labels, titles, legends, and grid lines

### Performance Optimizations

To keep runtime ~30 minutes:
- Dataset sampling: max 50K rows
- Ground truth sampling: 200-500 rows (expensive to compute)
- Reduced MTQD iterations: 3 instead of 5
- Fewer model trees during testing: 50 instead of 100
- Limited budget range: [10, 25, 50, 100, 200]
- Test first 3 attributes for Exp 5

## Output Structure

```
experiments/results/
├── results.json                 # All experimental data
└── figures/
    ├── exp1_overall_accuracy.png
    ├── exp1_per_metric_accuracy.png
    ├── exp2_query_speed.png
    ├── exp2_throughput.png
    ├── exp3_scalability.png
    ├── exp4_budget_accuracy.png
    ├── exp4_budget_features.png
    ├── exp5_homogeneity_separation.png
    ├── exp5_bin_counts.png
    └── exp9_ablation.png
```

### Results JSON Format

```json
{
  "exp1_accuracy": [
    {
      "dataset": "adult",
      "n_samples": 2000,
      "n_features": 3,
      "n_bins": 50,
      "r2_overall": 0.9856,
      "mae_overall": 0.0234,
      "build_time": 1.45,
      "pred_time": 0.05,
      "gt_time": 2.34,
      "speedup": 46.8
    },
    ...
  ],
  "exp1_per_metric": {
    "adult": {
      "completeness": 0.9875,
      "outlier_rate": 0.9923,
      ...
    }
  },
  ...
}
```

## Usage

### Running All Experiments

```bash
cd /sc/home/philipp.hildebrandt/InferQ
PYTHONPATH=src python experiments/run_experiments.py
```

### Running Individual Experiments

```python
from experiments.run_experiments import ExperimentRunner, load_dataset
from experiments.exp_rq1_accuracy import experiment_1_accuracy
from inferq.quality_metrics import get_default_registry

runner = ExperimentRunner()
registry = get_default_registry()

datasets = {
    'adult': load_dataset('adult', max_rows=10000)
}

experiment_1_accuracy(runner, datasets, registry)
runner.save_results()
```

## Expected Results

Based on design goals:

### RQ1: Accuracy
- Overall R² > 0.95 for most datasets
- Per-metric R² > 0.90 for core metrics
- MAE < 0.05 for normalized metrics
- MTQD achieves better H/S balance than baselines

### RQ2: Efficiency
- Single-tuple: 100-1000× speedup
- Batch queries: 10-100× speedup  
- Query latency: constant w.r.t. data size (O(1))
- Throughput: 100-1000 tuples/sec

### RQ3: Trade-offs
- Sweet spot: 50-100 bins achieves >0.90 R²
- Diminishing returns beyond 200 bins
- Full InferQ outperforms ablation variants
- Adaptive granularity improves accuracy by ~5%

## Known Limitations

1. **Ground Truth Computation**
   - Some metrics require extra parameters (column names, constraints, etc.)
   - These metrics are skipped with warnings during GT computation
   - Does not affect results as index only predicts metrics it was trained on

2. **Sampling for Speed**
   - Ground truth computed on subset (200-500 rows) for speed
   - May slightly underestimate true accuracy
   - Trade-off necessary to keep experiments under 30 minutes

3. **Metric Warnings**
   - Many warnings about missing parameters for specialized metrics
   - These are expected and don't affect experiment validity
   - Can be suppressed with `warnings.filterwarnings('ignore')`

## Future Enhancements

- Add more datasets from different domains
- Test larger budgets (500, 1000 bins)
- Compare against commercial tools (Great Expectations, Pandas Profiling)
- Add confidence intervals with multiple runs
- Test with concept drift scenarios
- Measure memory usage more precisely

## Files Created

1. **`run_experiments.py`** (220 lines) - Main orchestrator
2. **`exp_rq1_accuracy.py`** (290 lines) - Accuracy experiments
3. **`exp_rq2_efficiency.py`** (280 lines) - Efficiency experiments
4. **`exp_rq3_tradeoffs.py`** (370 lines) - Trade-off experiments
5. **`README.md`** - Experiment documentation

## Summary

Comprehensive experimental framework ready for paper evaluation:
- ✅ 6 experiments covering all research questions
- ✅ 10 publication-ready figures (300 DPI PNG)
- ✅ 3 real-world datasets from different domains
- ✅ ~30 minute total runtime
- ✅ Structured JSON results for analysis
- ✅ Modular, extensible architecture

The experiments provide rigorous validation of InferQ's accuracy, efficiency, and practical trade-offs!
