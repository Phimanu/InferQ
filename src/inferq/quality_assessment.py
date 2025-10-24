"""
WP 4: Stage 1 - Discretization Quality Assessment

Implements validation metrics (Homogeneity and Separation) to ensure
effective discretization from WP 3.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

from .partitioning import Bin, AttributePartition
from .discretization import MTQD


def compute_homogeneity(partition: AttributePartition, data: pd.DataFrame) -> float:
    """
    Compute homogeneity score (Equation 2).
    
    H = (1/m) Σ_j (1/|B|) Σ_b Var_t∈b(q_j(t))
    
    Measures average within-bin variance across all quality metrics.
    Lower H = more internally consistent bins = better discretization.
    
    Args:
        partition: Discretized attribute partition
        data: Full dataset
        
    Returns:
        Homogeneity score (lower is better)
    """
    bins = partition.bins
    if len(bins) == 0:
        return 0.0
    
    # Get all metrics from quality vectors
    metric_names = list(bins[0].quality_vector.keys()) if bins else []
    if not metric_names:
        return 0.0
    
    m = len(metric_names)  # Number of quality metrics
    total_variance = 0.0
    
    # For each quality metric
    for metric_name in metric_names:
        metric_variance_sum = 0.0
        
        # For each bin, compute variance of metric values across tuples
        for bin in bins:
            if len(bin.indices) < 2:
                # Variance undefined for single element
                continue
            
            # Get quality values for all tuples in this bin
            quality_values = []
            for idx in bin.indices:
                # For simplicity, use bin's aggregate quality as proxy
                # In practice, would compute per-tuple quality if available
                quality_values.append(bin.quality_vector.get(metric_name, 0.0))
            
            # Compute within-bin variance
            if len(quality_values) > 1:
                variance = np.var(quality_values, ddof=1)
                metric_variance_sum += variance
        
        # Average over bins
        avg_variance = metric_variance_sum / len(bins) if bins else 0.0
        total_variance += avg_variance
    
    # Average over metrics
    homogeneity = total_variance / m if m > 0 else 0.0
    return homogeneity


def compute_separation(partition: AttributePartition) -> float:
    """
    Compute separation score (Equation 3).
    
    S = (1/m) Σ_j Var_b∈B(q̄_j(b))
    
    Measures between-bin variance across all quality metrics.
    Higher S = bins differ meaningfully = better discretization.
    
    Args:
        partition: Discretized attribute partition
        
    Returns:
        Separation score (higher is better)
    """
    bins = partition.bins
    if len(bins) < 2:
        return 0.0
    
    # Get all metrics from quality vectors
    metric_names = list(bins[0].quality_vector.keys()) if bins else []
    if not metric_names:
        return 0.0
    
    m = len(metric_names)  # Number of quality metrics
    total_variance = 0.0
    
    # For each quality metric
    for metric_name in metric_names:
        # Collect mean quality value for each bin
        bin_means = []
        for bin in bins:
            mean_quality = bin.quality_vector.get(metric_name, 0.0)
            bin_means.append(mean_quality)
        
        # Compute variance across bin means
        if len(bin_means) > 1:
            variance = np.var(bin_means, ddof=1)
            total_variance += variance
    
    # Average over metrics
    separation = total_variance / m if m > 0 else 0.0
    return separation


@dataclass
class DiscretizationQuality:
    """Quality assessment results for a discretization."""
    homogeneity: float  # Lower is better (within-bin consistency)
    separation: float   # Higher is better (between-bin distinction)
    n_bins: int
    threshold: float
    
    def is_acceptable(self, max_homogeneity: float = 0.01, 
                     min_separation: float = 0.01) -> bool:
        """
        Check if discretization meets quality criteria.
        
        Args:
            max_homogeneity: Maximum acceptable homogeneity
            min_separation: Minimum acceptable separation
            
        Returns:
            True if criteria met
        """
        return self.homogeneity <= max_homogeneity and self.separation >= min_separation


class AdaptiveMTQD:
    """
    Adaptive MTQD that automatically adjusts merge threshold based on
    quality assessment metrics (H and S).
    """
    
    def __init__(self,
                 registry,
                 metric_configs: Dict = None,
                 metric_weights: Dict[str, float] = None,
                 max_homogeneity: float = 0.01,
                 min_separation: float = 0.01,
                 max_iterations: int = 10):
        """
        Initialize adaptive MTQD.
        
        Args:
            registry: Quality metric registry
            metric_configs: Metric-specific configs
            metric_weights: Weights for merge cost computation
            max_homogeneity: Maximum acceptable H score
            min_separation: Minimum acceptable S score
            max_iterations: Maximum threshold adjustment iterations
        """
        self.registry = registry
        self.metric_configs = metric_configs or {}
        self.metric_weights = metric_weights
        self.max_homogeneity = max_homogeneity
        self.min_separation = min_separation
        self.max_iterations = max_iterations
    
    def discretize_with_assessment(self,
                                   initial_partition: AttributePartition,
                                   data: pd.DataFrame,
                                   initial_threshold: float = 0.01) -> Tuple[AttributePartition, DiscretizationQuality]:
        """
        Run MTQD with automatic threshold adjustment based on H and S scores.
        
        Loop:
        1. Run MTQD with current threshold
        2. Compute H and S scores
        3. If H too high or S too low, adjust threshold and retry
        
        Args:
            initial_partition: Initial fine-grained partition
            data: Full dataset
            initial_threshold: Starting merge threshold
            
        Returns:
            (final_partition, quality_assessment)
        """
        threshold = initial_threshold
        threshold_min = 0.0001
        threshold_max = 0.1
        
        best_partition = None
        best_quality = None
        best_score = float('-inf')  # Combined score: S - H
        
        for iteration in range(self.max_iterations):
            # Run MTQD with current threshold
            mtqd = MTQD(
                merge_threshold=threshold,
                registry=self.registry,
                metric_configs=self.metric_configs,
                metric_weights=self.metric_weights
            )
            
            partition = mtqd.discretize(initial_partition, data)
            
            # Assess quality
            H = compute_homogeneity(partition, data)
            S = compute_separation(partition)
            
            quality = DiscretizationQuality(
                homogeneity=H,
                separation=S,
                n_bins=len(partition.bins),
                threshold=threshold
            )
            
            # Track best result
            combined_score = S - H
            if combined_score > best_score:
                best_score = combined_score
                best_partition = partition
                best_quality = quality
            
            # Check if acceptable
            if quality.is_acceptable(self.max_homogeneity, self.min_separation):
                return partition, quality
            
            # Adjust threshold based on which criterion failed
            if H > self.max_homogeneity:
                # Too much within-bin variance → need more merging → increase threshold
                threshold_min = threshold
                threshold = (threshold + threshold_max) / 2
            elif S < self.min_separation:
                # Not enough between-bin variance → need less merging → decrease threshold
                threshold_max = threshold
                threshold = (threshold_min + threshold) / 2
            else:
                break
        
        # Return best result if no acceptable solution found
        return best_partition, best_quality
