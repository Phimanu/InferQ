"""
InferQ: Quality-Aware Learned Index

A system for efficient, real-time data quality monitoring using learned indexes.
"""

__version__ = "0.1.0"

from inferq.quality_metrics import (
    QualityMetricRegistry,
    get_default_registry,
)
from inferq.partitioning import (
    Bin,
    AttributePartition,
    InitialPartitioner,
    equal_frequency_binning,
    summarize_partitions,
)
from inferq.discretization import (
    MTQD,
    compute_merge_cost,
    merge_bins,
)
from inferq.quality_assessment import (
    compute_homogeneity,
    compute_separation,
    DiscretizationQuality,
    AdaptiveMTQD,
)
from inferq.feature_scoring import (
    compute_information_gain,
    compute_ig_multi,
    compute_qdp,
    has_quality_issue,
    FeatureImportance,
    FeatureScorer,
    get_default_quality_thresholds,
)
from inferq.feature_selection import (
    GreedyFeatureSelector,
    SelectedFeature,
    SelectionResult,
    extract_discretization_boundaries,
    summarize_selection,
)
from inferq.bin_dictionary import (
    BinDictionary,
    TrainingData,
    generate_training_data,
    save_training_data,
    load_training_data,
)
from inferq.learned_index import (
    QualityIndex,
    train_model,
    save_model,
    load_model,
    ModelMetrics,
)

__all__ = [
    "QualityMetricRegistry",
    "get_default_registry",
    "Bin",
    "AttributePartition",
    "InitialPartitioner",
    "equal_frequency_binning",
    "summarize_partitions",
    "MTQD",
    "compute_merge_cost",
    "merge_bins",
    "compute_homogeneity",
    "compute_separation",
    "DiscretizationQuality",
    "AdaptiveMTQD",
    "compute_information_gain",
    "compute_ig_multi",
    "compute_qdp",
    "has_quality_issue",
    "FeatureImportance",
    "FeatureScorer",
    "get_default_quality_thresholds",
    "GreedyFeatureSelector",
    "SelectedFeature",
    "SelectionResult",
    "extract_discretization_boundaries",
    "summarize_selection",
    "BinDictionary",
    "TrainingData",
    "generate_training_data",
    "save_training_data",
    "load_training_data",
    "QualityIndex",
    "train_model",
    "save_model",
    "load_model",
    "ModelMetrics",
]
