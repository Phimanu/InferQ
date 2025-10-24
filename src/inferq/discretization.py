"""
WP 3: Multi-Target Quality-aware Discretization (MTQD)

Implements the core bottom-up merging algorithm with multi-objective optimization.
"""

import numpy as np
import pandas as pd
import heapq
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .partitioning import Bin, AttributePartition


def compute_merge_cost(bin1: Bin, bin2: Bin, metric_weights: Dict[str, float]) -> float:
    """
    Compute merge cost between two adjacent bins (Equation 1).
    
    MergeCost(b1, b2) = Σ w_j · (|b1|·|b2|)/(|b1|+|b2|) · (q_j(b1) - q_j(b2))²
    
    Size-weighted quality dissimilarity: larger bins or larger quality differences
    result in higher merge cost. Lower cost = better to merge.
    
    Args:
        bin1: First bin
        bin2: Second bin  
        metric_weights: Dict mapping metric names to weights w_j
        
    Returns:
        Merge cost (lower = more similar)
    """
    # Size-based weighting factor
    size1 = len(bin1.indices)
    size2 = len(bin2.indices)
    size_weight = (size1 * size2) / (size1 + size2)
    
    cost = 0.0
    
    for metric_name, weight in metric_weights.items():
        q1 = bin1.quality_vector.get(metric_name, 0.0)
        q2 = bin2.quality_vector.get(metric_name, 0.0)
        
        # Weighted squared difference with size penalty
        cost += weight * size_weight * (q1 - q2) ** 2
    
    return cost


def merge_bins(bin1: Bin, bin2: Bin, data: pd.DataFrame, 
               registry, metric_configs: Optional[Dict] = None) -> Bin:
    """
    Merge two adjacent bins into a new bin.
    
    Args:
        bin1: First bin
        bin2: Second bin
        data: Full dataset (to recompute quality on merged data)
        registry: Quality metric registry
        metric_configs: Optional metric configurations
        
    Returns:
        New merged bin
    """
    # Combine indices
    merged_indices = np.concatenate([bin1.indices, bin2.indices])
    
    # New boundaries
    new_bin = Bin(
        lower_bound=bin1.lower_bound,
        upper_bound=bin2.upper_bound,
        indices=merged_indices
    )
    
    # Recompute quality vector on merged data
    merged_data = data.iloc[merged_indices]
    
    # Compute basic metrics
    quality_vector = {}
    for metric_name in registry.list_metrics():
        metric = registry.get(metric_name)
        if not metric.requires_config:
            try:
                quality_vector[metric_name] = registry.compute(metric_name, merged_data)
            except Exception:
                quality_vector[metric_name] = 0.0
    
    # Compute configured metrics
    if metric_configs:
        for metric_name, config in metric_configs.items():
            try:
                quality_vector[metric_name] = registry.compute(metric_name, merged_data, **config)
            except Exception:
                quality_vector[metric_name] = 0.0
    
    new_bin.quality_vector = quality_vector
    return new_bin


class MTQD:
    """Multi-Target Quality-aware Discretization."""
    
    def __init__(self, 
                 registry,
                 metric_weights: Optional[Dict[str, float]] = None,
                 merge_threshold: float = 0.01,
                 metric_configs: Optional[Dict] = None):
        """
        Args:
            registry: Quality metric registry
            metric_weights: Weights for each metric (default: equal weights)
            merge_threshold: Stop merging when all costs exceed this
            metric_configs: Optional metric configurations
        """
        self.registry = registry
        self.merge_threshold = merge_threshold
        self.metric_configs = metric_configs or {}
        
        # Set default equal weights if not provided
        if metric_weights is None:
            basic_metrics = [m for m in registry.list_metrics() 
                           if not registry.get(m).requires_config]
            self.metric_weights = {m: 1.0 / len(basic_metrics) for m in basic_metrics}
        else:
            self.metric_weights = metric_weights
    
    def discretize(self, 
                   initial_partition: AttributePartition,
                   data: pd.DataFrame) -> AttributePartition:
        """
        Apply MTQD merging to an initial partition.
        
        Args:
            initial_partition: Initial fine-grained partition from WP 2
            data: Full dataset
            
        Returns:
            Optimized partition with merged bins
        """
        # Start with copy of initial bins
        bins = list(initial_partition.bins)
        
        if len(bins) <= 1:
            return initial_partition
        
        # Priority queue: (cost, counter, index_of_first_bin, bin1, bin2)
        # Counter breaks ties to avoid comparing Bin objects
        heap = []
        counter = 0
        
        # Initialize: compute costs for all adjacent pairs
        for i in range(len(bins) - 1):
            cost = compute_merge_cost(bins[i], bins[i+1], self.metric_weights)
            heapq.heappush(heap, (cost, counter, i, bins[i], bins[i+1]))
            counter += 1
        
        # Keep track of which bins are still active (not merged)
        active = {id(b): True for b in bins}
        
        # Iterative merging
        iteration = 0
        while heap:
            cost, _, idx, bin1, bin2 = heapq.heappop(heap)
            
            # Skip if bins were already merged
            if not active.get(id(bin1), False) or not active.get(id(bin2), False):
                continue
            
            # Stop if cost exceeds threshold
            if cost > self.merge_threshold:
                break
            
            # Merge bins
            new_bin = merge_bins(bin1, bin2, data, self.registry, self.metric_configs)
            
            # Mark old bins as inactive
            active[id(bin1)] = False
            active[id(bin2)] = False
            active[id(new_bin)] = True
            
            # Find position in bins list
            try:
                pos = bins.index(bin1)
            except ValueError:
                continue
            
            # Replace bin1 and bin2 with new_bin
            bins[pos] = new_bin
            if pos + 1 < len(bins):
                bins.pop(pos + 1)
            
            # Update costs with neighbors
            # Left neighbor
            if pos > 0:
                left_neighbor = bins[pos - 1]
                if active.get(id(left_neighbor), False):
                    new_cost = compute_merge_cost(left_neighbor, new_bin, self.metric_weights)
                    heapq.heappush(heap, (new_cost, counter, pos - 1, left_neighbor, new_bin))
                    counter += 1
            
            # Right neighbor  
            if pos + 1 < len(bins):
                right_neighbor = bins[pos + 1]
                if active.get(id(right_neighbor), False):
                    new_cost = compute_merge_cost(new_bin, right_neighbor, self.metric_weights)
                    heapq.heappush(heap, (new_cost, counter, pos, new_bin, right_neighbor))
                    counter += 1
            
            iteration += 1
        
        # Create new partition with merged bins
        # Filter out inactive bins and rebuild boundaries
        final_bins = [b for b in bins if active.get(id(b), False)]
        boundaries = np.array([b.lower_bound for b in final_bins] + [final_bins[-1].upper_bound])
        
        return AttributePartition(
            attribute_name=initial_partition.attribute_name,
            bins=final_bins,
            bin_boundaries=boundaries
        )
    
    def discretize_all(self,
                      initial_partitions: Dict[str, AttributePartition],
                      data: pd.DataFrame) -> Dict[str, AttributePartition]:
        """
        Apply MTQD to all attribute partitions.
        
        Args:
            initial_partitions: Initial partitions from WP 2
            data: Full dataset
            
        Returns:
            Dict of optimized partitions
        """
        optimized = {}
        
        for attr_name, partition in initial_partitions.items():
            optimized[attr_name] = self.discretize(partition, data)
        
        return optimized
