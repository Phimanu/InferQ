# InferQ Experimental Evaluation

Comprehensive experimental framework for evaluating InferQ's accuracy, efficiency, and trade-offs.

## Overview

This directory contains experiments designed to validate InferQ's performance for the research paper:

- **RQ1: Accuracy** - How accurately does the index predict quality metrics?
- **RQ2: Efficiency** - How fast is the index compared to ground truth computation?
- **RQ3: Trade-offs** - What are the size-accuracy trade-offs?

## Running Experiments

### Quick Start

```bash
# Run all experiments (~30 minutes)
cd /path/to/InferQ
PYTHONPATH=src python experiments/run_experiments.py
```

### Output

Results are saved to `experiments/results/`:
- `results.json` - All experimental data
- `figures/` - Publication-ready PNG figures

## Experiments

### Experiment 1: Prediction Accuracy
**Goal:** Validate index predictions against ground truth

**Metrics:**
- Overall R² score
- Mean Absolute Error (MAE)
- Per-metric R² scores

**Figures:**
- `exp1_overall_accuracy.png` - R² and MAE by dataset
- `exp1_per_metric_accuracy.png` - Heatmap of per-metric R²

### Experiment 2: Query Speed
**Goal:** Measure speedup vs direct computation

**Metrics:**
- Latency per tuple (μs)
- Throughput (tuples/sec)
- Speedup ratio

**Figures:**
- `exp2_query_speed.png` - Speedup and latency vs query size
- `exp2_throughput.png` - Throughput comparison

### Experiment 3: Scalability
**Goal:** Show constant-time queries regardless of data size

**Metrics:**
- Build time vs data size
- Query latency (should be constant)
- Index size vs data size
- Compression ratio

**Figures:**
- `exp3_scalability.png` - 4-panel scalability analysis

### Experiment 4: Size-Accuracy Trade-offs
**Goal:** Find optimal budget (B_max)

**Metrics:**
- R² vs budget
- MAE vs budget
- Features selected vs budget

**Figures:**
- `exp4_budget_accuracy.png` - Accuracy and error vs budget
- `exp4_budget_features.png` - Feature count vs budget

### Experiment 5: MTQD Effectiveness
**Goal:** Validate multi-target discretization

**Metrics:**
- Homogeneity score (lower is better)
- Separation score (higher is better)
- Bin count reduction

**Comparisons:**
- Equal-frequency (no merging)
- MTQD (adaptive)
- MTQD (aggressive)

**Figures:**
- `exp5_homogeneity_separation.png` - H vs S scatter plots
- `exp5_bin_counts.png` - Bin count comparison

### Experiment 9: Ablation Study
**Goal:** Understand component importance

**Variants:**
- Full InferQ (baseline)
- No adaptive granularity
- Fewer trees (25 instead of 50)
- IG-only feature selection (no QDP)

**Figures:**
- `exp9_ablation.png` - R², build time, and index size by variant

## Datasets

Three real-world datasets from different domains:

1. **Adult** (UCI) - 32,561 rows, demographic data
   - Quality issues: missing values, outliers
   
2. **Cardio** - 70,000 rows, medical data
   - Quality issues: outliers, distribution skew
   
3. **Flight** - 300,153 rows, airline data
   - Quality issues: missing values, temporal patterns

## Configuration

Experiments are configured for ~30 minute total runtime:

- Dataset samples: max 50K rows per dataset
- Ground truth samples: 200-500 rows (for speed)
- Budget range: [10, 25, 50, 100, 200]
- Model: 50 estimators, depth 15

To run faster experiments, modify sample sizes in `run_experiments.py`:

```python
# Line 196
max_rows = min(meta['rows'], 10000)  # Reduce to 10K
```

## Results Structure

```
experiments/
├── run_experiments.py          # Main orchestrator
├── exp_rq1_accuracy.py         # RQ1 experiments
├── exp_rq2_efficiency.py       # RQ2 experiments
├── exp_rq3_tradeoffs.py        # RQ3 experiments
├── results/
│   ├── results.json            # All experimental data
│   └── figures/                # Publication figures
│       ├── exp1_overall_accuracy.png
│       ├── exp1_per_metric_accuracy.png
│       ├── exp2_query_speed.png
│       ├── exp2_throughput.png
│       ├── exp3_scalability.png
│       ├── exp4_budget_accuracy.png
│       ├── exp4_budget_features.png
│       ├── exp5_homogeneity_separation.png
│       ├── exp5_bin_counts.png
│       └── exp9_ablation.png
└── run.log                     # Execution log
```

## Expected Results

### RQ1: Accuracy
- Overall R² > 0.95 across datasets
- Per-metric R² > 0.90 for most metrics
- Low MAE (< 0.05)

### RQ2: Efficiency
- 100-1000× speedup for single tuples
- 10-100× speedup for batches
- Throughput: 100-1000 tuples/sec

### RQ3: Trade-offs
- Sweet spot: ~50-100 bins achieves >0.90 R²
- Diminishing returns beyond 200 bins
- Full InferQ outperforms all ablation variants

## Publication Figures

All figures are saved as 300 DPI PNG files suitable for publication. They use a clean, professional style with:
- Clear labels and titles
- Color-blind friendly palettes
- Grid lines for readability
- Legends where appropriate

## Troubleshooting

**Out of memory:**
```python
# Reduce sample sizes in run_experiments.py
max_rows = min(meta['rows'], 10000)
```

**Too slow:**
```python
# Reduce ground truth samples in exp_rq1_accuracy.py
gt_sample_size = min(200, len(data_sample))
```

**Missing dependencies:**
```bash
pip install matplotlib seaborn scikit-learn pandas numpy
```

## Citation

If you use these experiments in your research, please cite:

```bibtex
@article{inferq2025,
  title={InferQ: Quality-Aware Learned Indexes for Efficient Data Profiling},
  author={...},
  journal={...},
  year={2025}
}
```
