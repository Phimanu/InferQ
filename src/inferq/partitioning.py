"""
WP 2: Initial Partitioning and Annotation

Implements equal-frequency binning and quality vector annotation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .quality_metrics import QualityMetricRegistry


@dataclass
class Bin:
    """A single bin with boundaries, data indices, and quality metrics."""
    lower_bound: float
    upper_bound: float
    indices: np.ndarray
    quality_vector: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Bin([{self.lower_bound:.2f}, {self.upper_bound:.2f}], n={len(self.indices)})"


@dataclass
class AttributePartition:
    """Complete partitioning for one attribute."""
    attribute_name: str
    bins: List[Bin]
    bin_boundaries: np.ndarray
    
    def __repr__(self):
        return f"AttributePartition('{self.attribute_name}', {len(self.bins)} bins)"


def equal_frequency_binning(data: pd.Series, n_bins: int = 100) -> np.ndarray:
    """
    Create equal-frequency bin boundaries.
    
    Args:
        data: Numeric data series
        n_bins: Number of bins to create
        
    Returns:
        Array of bin boundaries (length n_bins + 1)
    """
    data_clean = data.dropna()
    
    if len(data_clean) < n_bins:
        n_bins = max(1, len(data_clean))
    
    # Use quantiles for equal-frequency
    quantiles = np.linspace(0, 1, n_bins + 1)
    boundaries = np.quantile(data_clean, quantiles)
    
    # Ensure unique boundaries
    boundaries = np.unique(boundaries)
    
    return boundaries


class InitialPartitioner:
    """Creates fine-grained bins and annotates with quality metrics."""
    
    def __init__(self, 
                 registry: QualityMetricRegistry,
                 n_bins: int = 100,
                 metric_configs: Optional[Dict[str, Dict]] = None):
        """
        Args:
            registry: Quality metric registry
            n_bins: Number of initial bins per attribute
            metric_configs: Optional configurations for metrics
        """
        self.registry = registry
        self.n_bins = n_bins
        self.metric_configs = metric_configs or {}
        
        # Metrics that don't require config
        self.basic_metrics = [m for m in registry.list_metrics() 
                             if not registry.get(m).requires_config]
    
    def partition_attribute(self, 
                           data: pd.DataFrame,
                           attribute: str) -> AttributePartition:
        """
        Partition a single attribute and annotate bins with quality vectors.
        
        Args:
            data: Full dataset
            attribute: Name of numeric attribute to partition
            
        Returns:
            AttributePartition with annotated bins
        """
        if attribute not in data.columns:
            raise ValueError(f"Attribute '{attribute}' not found")
        
        attr_data = data[attribute]
        
        # Create bin boundaries
        boundaries = equal_frequency_binning(attr_data, self.n_bins)
        
        # Create bins
        bins = []
        for i in range(len(boundaries) - 1):
            lower = boundaries[i]
            upper = boundaries[i + 1]
            
            # Find indices in this bin
            if i < len(boundaries) - 2:
                mask = (attr_data >= lower) & (attr_data < upper)
            else:
                # Last bin includes upper boundary
                mask = (attr_data >= lower) & (attr_data <= upper)
            
            indices = np.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # Create bin
            bin_obj = Bin(
                lower_bound=lower,
                upper_bound=upper,
                indices=indices
            )
            
            # Compute quality vector
            bin_data = data.iloc[indices]
            quality_vector = self._compute_quality_vector(bin_data)
            bin_obj.quality_vector = quality_vector
            
            bins.append(bin_obj)
        
        return AttributePartition(
            attribute_name=attribute,
            bins=bins,
            bin_boundaries=boundaries
        )
    
    def partition_all_attributes(self, 
                                data: pd.DataFrame,
                                attributes: Optional[List[str]] = None) -> Dict[str, AttributePartition]:
        """
        Partition all numeric attributes.
        
        Args:
            data: Full dataset
            attributes: List of attributes to partition (None = all numeric)
            
        Returns:
            Dict mapping attribute names to their partitions
        """
        if attributes is None:
            # Auto-detect numeric columns
            attributes = data.select_dtypes(include=[np.number]).columns.tolist()
        
        partitions = {}
        for attr in attributes:
            partitions[attr] = self.partition_attribute(data, attr)
        
        return partitions
    
    def _compute_quality_vector(self, bin_data: pd.DataFrame) -> Dict[str, float]:
        """Compute quality metrics for a bin's data subset."""
        quality_vector = {}
        
        # Compute basic metrics (no config needed)
        for metric_name in self.basic_metrics:
            try:
                score = self.registry.compute(metric_name, bin_data)
                quality_vector[metric_name] = score
            except Exception:
                quality_vector[metric_name] = 0.0
        
        # Compute configured metrics
        for metric_name, config in self.metric_configs.items():
            try:
                score = self.registry.compute(metric_name, bin_data, **config)
                quality_vector[metric_name] = score
            except Exception:
                quality_vector[metric_name] = 0.0
        
        return quality_vector


def summarize_partitions(partitions: Dict[str, AttributePartition]) -> pd.DataFrame:
    """Create summary statistics for partitions."""
    summary_data = []
    
    for attr_name, partition in partitions.items():
        summary_data.append({
            'attribute': attr_name,
            'n_bins': len(partition.bins),
            'min_bin_size': min(len(b.indices) for b in partition.bins),
            'max_bin_size': max(len(b.indices) for b in partition.bins),
            'avg_bin_size': np.mean([len(b.indices) for b in partition.bins]),
        })
    
    return pd.DataFrame(summary_data)
