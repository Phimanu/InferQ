"""
WP 6: Stage 2 - Greedy Feature Selection

Implements adaptive greedy selection to choose features that fit within
the index budget B_max.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .partitioning import AttributePartition
from .discretization import MTQD
from .quality_assessment import compute_homogeneity, compute_separation
from .feature_scoring import FeatureScorer, FeatureImportance


@dataclass
class SelectedFeature:
    """Information about a selected feature."""
    attribute: str
    partition: AttributePartition
    importance: float
    n_bins: int
    adjusted: bool = False  # Whether granularity was adjusted to fit budget


@dataclass
class SelectionResult:
    """Result of feature selection process."""
    selected_features: List[SelectedFeature]
    total_bins: int
    budget: int
    budget_utilization: float  # Fraction of budget used
    rejected_features: List[str]  # Features that couldn't fit


class GreedyFeatureSelector:
    """
    Adaptive greedy feature selection with budget constraints.
    
    Selects features in order of importance, adjusting granularity
    when needed to fit within index budget B_max.
    """
    
    def __init__(self,
                 budget: int,
                 registry,
                 metric_configs: Optional[Dict] = None,
                 metric_weights: Optional[Dict[str, float]] = None,
                 min_bins_per_feature: int = 3,
                 max_adjustment_attempts: int = 3):
        """
        Initialize greedy feature selector.
        
        Args:
            budget: Maximum total bins across all features (B_max)
            registry: Quality metric registry
            metric_configs: Metric-specific configurations
            metric_weights: Weights for merge cost computation
            min_bins_per_feature: Minimum bins required per feature
            max_adjustment_attempts: Maximum attempts to adjust granularity
        """
        self.budget = budget
        self.registry = registry
        self.metric_configs = metric_configs or {}
        self.metric_weights = metric_weights
        self.min_bins_per_feature = min_bins_per_feature
        self.max_adjustment_attempts = max_adjustment_attempts
    
    def select_features(self,
                       ranked_features: List[FeatureImportance],
                       partitions: Dict[str, AttributePartition],
                       data: pd.DataFrame) -> SelectionResult:
        """
        Select features using greedy algorithm with adaptive granularity.
        
        Algorithm:
        1. Rank features by importance (already done)
        2. Iterate through ranked features
        3. Try to add each feature:
           a. If fits budget: add it
           b. If exceeds budget: try to adjust granularity (reduce bins)
           c. If still doesn't fit or too few bins: reject
        
        Args:
            ranked_features: Features sorted by importance (descending)
            partitions: Dict mapping attribute names to partitions
            data: Full dataset
            
        Returns:
            SelectionResult with selected features and statistics
        """
        selected = []
        rejected = []
        cumulative_bins = 0
        
        for feature in ranked_features:
            attr_name = feature.attribute
            partition = partitions[attr_name]
            n_bins = len(partition.bins)
            
            # Check if feature fits in remaining budget
            remaining_budget = self.budget - cumulative_bins
            
            if n_bins <= remaining_budget:
                # Feature fits - add it
                selected.append(SelectedFeature(
                    attribute=attr_name,
                    partition=partition,
                    importance=feature.importance,
                    n_bins=n_bins,
                    adjusted=False
                ))
                cumulative_bins += n_bins
                
            elif remaining_budget >= self.min_bins_per_feature:
                # Try adaptive granularity adjustment
                adjusted_partition = self._adjust_granularity(
                    partition=partition,
                    data=data,
                    target_bins=remaining_budget,
                    attribute=attr_name
                )
                
                if adjusted_partition is not None:
                    n_bins_adjusted = len(adjusted_partition.bins)
                    selected.append(SelectedFeature(
                        attribute=attr_name,
                        partition=adjusted_partition,
                        importance=feature.importance,
                        n_bins=n_bins_adjusted,
                        adjusted=True
                    ))
                    cumulative_bins += n_bins_adjusted
                else:
                    rejected.append(attr_name)
            else:
                # Not enough budget remaining
                rejected.append(attr_name)
        
        utilization = cumulative_bins / self.budget if self.budget > 0 else 0.0
        
        return SelectionResult(
            selected_features=selected,
            total_bins=cumulative_bins,
            budget=self.budget,
            budget_utilization=utilization,
            rejected_features=rejected
        )
    
    def _adjust_granularity(self,
                           partition: AttributePartition,
                           data: pd.DataFrame,
                           target_bins: int,
                           attribute: str) -> Optional[AttributePartition]:
        """
        Adjust feature granularity to fit budget using adaptive MTQD.
        
        Strategy:
        1. Try increasingly aggressive merge thresholds
        2. Stop when target_bins achieved or quality degrades too much
        
        Args:
            partition: Original partition
            data: Full dataset
            target_bins: Target number of bins to fit budget
            attribute: Attribute name
            
        Returns:
            Adjusted partition or None if adjustment failed
        """
        if target_bins < self.min_bins_per_feature:
            return None
        
        current_bins = len(partition.bins)
        
        # Try multiple threshold levels
        threshold_multipliers = [2.0, 5.0, 10.0, 20.0, 50.0]
        
        best_partition = None
        best_diff = float('inf')
        
        for multiplier in threshold_multipliers[:self.max_adjustment_attempts]:
            # Compute aggressive threshold
            # Start from a base threshold and increase
            base_threshold = 0.01
            threshold = base_threshold * multiplier
            
            # Apply MTQD with aggressive threshold
            mtqd = MTQD(
                merge_threshold=threshold,
                registry=self.registry,
                metric_configs=self.metric_configs,
                metric_weights=self.metric_weights
            )
            
            adjusted = mtqd.discretize(partition, data)
            n_bins = len(adjusted.bins)
            
            # Check if it fits and is better than previous attempts
            if n_bins <= target_bins and n_bins >= self.min_bins_per_feature:
                diff = target_bins - n_bins
                if diff < best_diff:
                    best_diff = diff
                    best_partition = adjusted
            
            # Stop if we hit target exactly or very close
            if n_bins <= target_bins and n_bins >= target_bins - 2:
                break
        
        return best_partition
    
    def select_with_quality_constraints(self,
                                       ranked_features: List[FeatureImportance],
                                       partitions: Dict[str, AttributePartition],
                                       data: pd.DataFrame,
                                       max_homogeneity: float = 0.02,
                                       min_separation: float = 0.005) -> SelectionResult:
        """
        Select features with both budget and quality constraints.
        
        Only accepts features (even with adjusted granularity) if they
        meet quality criteria.
        
        Args:
            ranked_features: Features sorted by importance
            partitions: Dict mapping attribute names to partitions
            data: Full dataset
            max_homogeneity: Maximum acceptable H score
            min_separation: Minimum acceptable S score
            
        Returns:
            SelectionResult with quality-validated features
        """
        selected = []
        rejected = []
        cumulative_bins = 0
        
        for feature in ranked_features:
            attr_name = feature.attribute
            partition = partitions[attr_name]
            n_bins = len(partition.bins)
            
            remaining_budget = self.budget - cumulative_bins
            
            # Try to fit feature
            final_partition = None
            is_adjusted = False
            
            if n_bins <= remaining_budget:
                final_partition = partition
            elif remaining_budget >= self.min_bins_per_feature:
                # Try adjustment
                final_partition = self._adjust_granularity(
                    partition, data, remaining_budget, attr_name
                )
                is_adjusted = True
            
            if final_partition is not None:
                # Validate quality
                H = compute_homogeneity(final_partition, data)
                S = compute_separation(final_partition)
                
                if H <= max_homogeneity and S >= min_separation:
                    # Meets quality criteria - accept
                    n_bins_final = len(final_partition.bins)
                    selected.append(SelectedFeature(
                        attribute=attr_name,
                        partition=final_partition,
                        importance=feature.importance,
                        n_bins=n_bins_final,
                        adjusted=is_adjusted
                    ))
                    cumulative_bins += n_bins_final
                else:
                    rejected.append(attr_name)
            else:
                rejected.append(attr_name)
        
        utilization = cumulative_bins / self.budget if self.budget > 0 else 0.0
        
        return SelectionResult(
            selected_features=selected,
            total_bins=cumulative_bins,
            budget=self.budget,
            budget_utilization=utilization,
            rejected_features=rejected
        )


def extract_discretization_boundaries(selected_features: List[SelectedFeature]) -> Dict[str, np.ndarray]:
    """
    Extract final discretization boundaries for selected features.
    
    Args:
        selected_features: List of selected features with partitions
        
    Returns:
        Dict mapping attribute names to boundary arrays
    """
    boundaries = {}
    
    for feature in selected_features:
        partition = feature.partition
        # Extract boundaries from bins
        bin_boundaries = np.array([b.lower_bound for b in partition.bins] + 
                                 [partition.bins[-1].upper_bound])
        boundaries[feature.attribute] = bin_boundaries
    
    return boundaries


def summarize_selection(result: SelectionResult) -> str:
    """
    Create human-readable summary of selection result.
    
    Args:
        result: SelectionResult to summarize
        
    Returns:
        Formatted string summary
    """
    lines = []
    lines.append(f"Feature Selection Summary")
    lines.append(f"=" * 60)
    lines.append(f"Budget: {result.total_bins} / {result.budget} bins ({result.budget_utilization:.1%} utilized)")
    lines.append(f"Selected: {len(result.selected_features)} features")
    lines.append(f"Rejected: {len(result.rejected_features)} features")
    lines.append("")
    
    if result.selected_features:
        lines.append(f"Selected Features:")
        lines.append(f"{'Rank':<6} {'Attribute':<20} {'Bins':>6} {'Importance':>11} {'Adjusted':>9}")
        lines.append("-" * 60)
        for i, feature in enumerate(result.selected_features, 1):
            adj_mark = "✓" if feature.adjusted else ""
            lines.append(f"{i:<6} {feature.attribute:<20} {feature.n_bins:>6} "
                        f"{feature.importance:>11.4f} {adj_mark:>9}")
    
    if result.rejected_features:
        lines.append("")
        lines.append(f"Rejected Features: {', '.join(result.rejected_features)}")
    
    return "\n".join(lines)
