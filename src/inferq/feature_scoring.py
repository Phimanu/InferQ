"""
WP 5: Stage 2 - Feature Importance Scoring

Implements quality-aware feature importance metrics to rank attributes
for learned index construction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass

from .partitioning import AttributePartition


def compute_entropy(values: np.ndarray) -> float:
    """
    Compute Shannon entropy of a discrete distribution.
    
    Args:
        values: Array of values
        
    Returns:
        Entropy in bits
    """
    if len(values) == 0:
        return 0.0
    
    # Get value counts
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)
    
    # Compute entropy: H = -Σ p(x) log2(p(x))
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy


def compute_information_gain(partition: AttributePartition,
                            quality_metric: str,
                            data: pd.DataFrame) -> float:
    """
    Compute base information gain IG(A_i, q_j).
    
    Measures how much the discretization of attribute A_i reduces
    uncertainty about quality metric q_j.
    
    IG(A_i, q_j) = H(q_j) - H(q_j | bins(A_i))
    
    Args:
        partition: Discretized attribute partition
        quality_metric: Name of quality metric
        data: Full dataset
        
    Returns:
        Information gain (higher = more informative)
    """
    bins = partition.bins
    if len(bins) == 0:
        return 0.0
    
    # Get quality values for all bins
    quality_values = np.array([b.quality_vector.get(quality_metric, 0.0) for b in bins])
    
    # Discretize quality values into categories for entropy computation
    # Use quantile-based binning to create discrete categories
    if len(np.unique(quality_values)) <= 10:
        discretized_quality = quality_values
    else:
        discretized_quality = pd.qcut(quality_values, q=10, labels=False, duplicates='drop')
    
    # Prior entropy H(q_j) - entropy before partitioning
    prior_entropy = compute_entropy(discretized_quality)
    
    # Conditional entropy H(q_j | bins) - entropy after partitioning
    conditional_entropy = 0.0
    total_size = sum(len(b.indices) for b in bins)
    
    for i, bin in enumerate(bins):
        bin_size = len(bin.indices)
        if bin_size == 0:
            continue
        
        # For this bin, all tuples have same quality value (bin's aggregate)
        # So within-bin entropy is 0 (perfect information)
        bin_weight = bin_size / total_size
        bin_entropy = 0.0  # All tuples in bin have same quality
        
        conditional_entropy += bin_weight * bin_entropy
    
    # Information gain = reduction in entropy
    information_gain = prior_entropy - conditional_entropy
    return information_gain


def compute_ig_multi(partition: AttributePartition,
                    data: pd.DataFrame,
                    metric_weights: Dict[str, float]) -> float:
    """
    Compute multi-target information gain (Equation 4).
    
    IG_multi(A_i) = Σ_j w_j · IG(A_i, q_j)
    
    Weighted sum of information gain across all quality metrics.
    
    Args:
        partition: Discretized attribute partition
        data: Full dataset
        metric_weights: Weights for each quality metric
        
    Returns:
        Multi-target information gain
    """
    ig_multi = 0.0
    
    for metric_name, weight in metric_weights.items():
        ig = compute_information_gain(partition, metric_name, data)
        ig_multi += weight * ig
    
    return ig_multi


def has_quality_issue(bin, 
                     quality_thresholds: Dict[str, Tuple[float, float]]) -> bool:
    """
    Check if a bin has quality issues.
    
    A bin has issues if any quality metric falls outside acceptable range.
    
    Args:
        bin: Bin to check
        quality_thresholds: Dict mapping metric names to (min, max) acceptable values
        
    Returns:
        True if bin has quality issues
    """
    for metric_name, (min_val, max_val) in quality_thresholds.items():
        quality_value = bin.quality_vector.get(metric_name, 1.0)
        
        # Check if outside acceptable range
        if quality_value < min_val or quality_value > max_val:
            return True
    
    return False


def compute_qdp(partition: AttributePartition,
               quality_thresholds: Dict[str, Tuple[float, float]],
               total_records: int) -> float:
    """
    Compute Quality Detection Power (Equation 5).
    
    QDP(A_i) = Σ_b |b| · 𝟙[hasQualityIssue(b)] / |D|
    
    Measures how well the attribute isolates quality issues.
    Higher QDP = more records with quality issues are concentrated in identifiable bins.
    
    Args:
        partition: Discretized attribute partition
        quality_thresholds: Acceptable ranges for each metric
        total_records: Total number of records in dataset
        
    Returns:
        Quality Detection Power (0 to 1)
    """
    if total_records == 0:
        return 0.0
    
    issue_records = 0
    
    for bin in partition.bins:
        if has_quality_issue(bin, quality_thresholds):
            issue_records += len(bin.indices)
    
    qdp = issue_records / total_records
    return qdp


@dataclass
class FeatureImportance:
    """Feature importance scores for an attribute."""
    attribute: str
    ig_multi: float        # Multi-target information gain
    qdp: float             # Quality detection power
    importance: float      # Combined score
    n_bins: int
    n_issue_bins: int      # Number of bins with quality issues


class FeatureScorer:
    """
    Scores feature importance using IG_multi and QDP metrics.
    """
    
    def __init__(self,
                 metric_weights: Optional[Dict[str, float]] = None,
                 quality_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
                 alpha: float = 0.7,
                 beta: float = 0.3):
        """
        Initialize feature scorer.
        
        Args:
            metric_weights: Weights for each quality metric in IG_multi
            quality_thresholds: Acceptable ranges for each metric (min, max)
            alpha: Weight for IG_multi in combined score
            beta: Weight for QDP in combined score
        """
        self.metric_weights = metric_weights or {}
        self.quality_thresholds = quality_thresholds or {}
        self.alpha = alpha
        self.beta = beta
    
    def score_feature(self,
                     partition: AttributePartition,
                     data: pd.DataFrame) -> FeatureImportance:
        """
        Compute importance score for a single feature.
        
        importance[A_i] = α · IG_multi(A_i) + β · QDP(A_i)
        
        Args:
            partition: Discretized attribute partition
            data: Full dataset
            
        Returns:
            FeatureImportance with all scores
        """
        # Compute IG_multi
        ig_multi = compute_ig_multi(partition, data, self.metric_weights)
        
        # Compute QDP
        qdp = compute_qdp(partition, self.quality_thresholds, len(data))
        
        # Combined importance score
        importance = self.alpha * ig_multi + self.beta * qdp
        
        # Count bins with issues
        n_issue_bins = sum(
            1 for b in partition.bins 
            if has_quality_issue(b, self.quality_thresholds)
        )
        
        return FeatureImportance(
            attribute=partition.attribute_name,
            ig_multi=ig_multi,
            qdp=qdp,
            importance=importance,
            n_bins=len(partition.bins),
            n_issue_bins=n_issue_bins
        )
    
    def score_features(self,
                      partitions: Dict[str, AttributePartition],
                      data: pd.DataFrame) -> List[FeatureImportance]:
        """
        Score all features and return sorted by importance.
        
        Args:
            partitions: Dict mapping attribute names to their partitions
            data: Full dataset
            
        Returns:
            List of FeatureImportance, sorted by importance (descending)
        """
        scores = []
        
        for attr_name, partition in partitions.items():
            score = self.score_feature(partition, data)
            scores.append(score)
        
        # Sort by importance (highest first)
        scores.sort(key=lambda x: x.importance, reverse=True)
        return scores
    
    def select_top_features(self,
                           partitions: Dict[str, AttributePartition],
                           data: pd.DataFrame,
                           k: int) -> List[str]:
        """
        Select top-k most important features.
        
        Args:
            partitions: Dict mapping attribute names to their partitions
            data: Full dataset
            k: Number of features to select
            
        Returns:
            List of top-k attribute names
        """
        scores = self.score_features(partitions, data)
        return [s.attribute for s in scores[:k]]


def get_default_quality_thresholds() -> Dict[str, Tuple[float, float]]:
    """
    Get default quality thresholds for common metrics.
    
    Returns:
        Dict mapping metric names to (min_acceptable, max_acceptable) values
    """
    return {
        'completeness': (0.95, 1.0),           # At least 95% complete
        'outlier_rate': (0.0, 0.05),           # Max 5% outliers
        'duplicate_rate': (0.0, 0.05),         # Max 5% duplicates
        'constraint_violation_rate': (0.0, 0.05),  # Max 5% violations
        'consistency_score': (0.9, 1.0),       # At least 90% consistent
    }
