"""
WP 7: Stage 3 - Bin Dictionary and Training Data Generation

Implements the lookup component (BinDictionary) and generates training data
for the learned index model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import bisect

from .partitioning import AttributePartition, Bin
from .feature_selection import SelectedFeature


@dataclass
class BinDictionary:
    """
    Fast lookup structure for bin assignments.
    
    Stores bin boundaries for numeric features and hash maps for categorical
    features. Enables efficient get_bin_vector() lookups.
    """
    
    # Numeric features: attribute -> sorted boundary array
    numeric_boundaries: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Categorical features: attribute -> {value -> bin_id}
    categorical_maps: Dict[str, Dict[Any, int]] = field(default_factory=dict)
    
    # Feature ordering for vector construction
    feature_order: List[str] = field(default_factory=list)
    
    # Bin-level quality vectors: (attribute, bin_id) -> quality_vector
    bin_quality_vectors: Dict[Tuple[str, int], Dict[str, float]] = field(default_factory=dict)
    
    # Metadata
    n_features: int = 0
    total_bins: int = 0
    
    @classmethod
    def from_selected_features(cls, 
                              selected_features: List[SelectedFeature],
                              data: Optional[pd.DataFrame] = None) -> 'BinDictionary':
        """
        Build BinDictionary from selected features.
        
        Args:
            selected_features: Features selected by WP 6
            data: Optional dataset for categorical feature inference
            
        Returns:
            BinDictionary ready for lookups
        """
        bin_dict = cls()
        
        for feature in selected_features:
            attr = feature.attribute
            partition = feature.partition
            
            bin_dict.feature_order.append(attr)
            
            # Check if numeric or categorical
            is_numeric = cls._is_numeric_partition(partition)
            
            if is_numeric:
                # Extract sorted boundaries
                boundaries = np.array([b.lower_bound for b in partition.bins] + 
                                     [partition.bins[-1].upper_bound])
                bin_dict.numeric_boundaries[attr] = boundaries
            else:
                # Build categorical hash map
                cat_map = {}
                for bin_id, bin in enumerate(partition.bins):
                    # For categorical, store representative values
                    # This is a simplified approach - in practice would need
                    # more sophisticated categorical handling
                    cat_map[bin_id] = bin_id
                bin_dict.categorical_maps[attr] = cat_map
            
            # Store bin-level quality vectors
            for bin_id, bin in enumerate(partition.bins):
                bin_dict.bin_quality_vectors[(attr, bin_id)] = bin.quality_vector.copy()
        
        bin_dict.n_features = len(selected_features)
        bin_dict.total_bins = sum(len(f.partition.bins) for f in selected_features)
        
        return bin_dict
    
    @staticmethod
    def _is_numeric_partition(partition: AttributePartition) -> bool:
        """Check if partition is for numeric or categorical attribute."""
        # Assume numeric if bins have meaningful numeric boundaries
        if len(partition.bins) == 0:
            return True
        
        first_bin = partition.bins[0]
        # Check if boundaries are numeric (not categorical identifiers)
        return isinstance(first_bin.lower_bound, (int, float, np.number))
    
    def get_bin_id(self, attribute: str, value: Any) -> int:
        """
        Get bin ID for a single attribute value.
        
        Args:
            attribute: Attribute name
            value: Value to lookup
            
        Returns:
            Bin ID (0-indexed)
        """
        if pd.isna(value):
            # Handle missing values - assign to first bin by convention
            return 0
        
        if attribute in self.numeric_boundaries:
            # Binary search for numeric attribute
            boundaries = self.numeric_boundaries[attribute]
            bin_id = bisect.bisect_right(boundaries, value) - 1
            
            # Clamp to valid range
            bin_id = max(0, min(bin_id, len(boundaries) - 2))
            return bin_id
            
        elif attribute in self.categorical_maps:
            # Hash lookup for categorical attribute
            cat_map = self.categorical_maps[attribute]
            return cat_map.get(value, 0)  # Default to bin 0 if unknown
        
        else:
            raise ValueError(f"Attribute '{attribute}' not found in dictionary")
    
    def get_bin_vector(self, row: Union[pd.Series, Dict]) -> np.ndarray:
        """
        Get k-dimensional bin assignment vector for a data row.
        
        Args:
            row: Data row as Series or dict
            
        Returns:
            Array of bin IDs [bin_id_A1, bin_id_A2, ..., bin_id_Ak]
        """
        if isinstance(row, dict):
            row = pd.Series(row)
        
        bin_vector = np.zeros(self.n_features, dtype=np.int32)
        
        for i, attr in enumerate(self.feature_order):
            value = row.get(attr, np.nan)
            bin_vector[i] = self.get_bin_id(attr, value)
        
        return bin_vector
    
    def get_quality_vector(self, bin_vector: np.ndarray) -> Dict[str, float]:
        """
        Get aggregated quality vector for a bin assignment.
        
        Aggregates quality across all attributes in the bin vector.
        
        Args:
            bin_vector: Bin assignment vector from get_bin_vector()
            
        Returns:
            Dict mapping metric names to aggregated quality values
        """
        quality_vectors = []
        
        for i, attr in enumerate(self.feature_order):
            bin_id = int(bin_vector[i])
            key = (attr, bin_id)
            
            if key in self.bin_quality_vectors:
                quality_vectors.append(self.bin_quality_vectors[key])
        
        if not quality_vectors:
            return {}
        
        # Aggregate by averaging
        metric_names = quality_vectors[0].keys()
        aggregated = {}
        
        for metric in metric_names:
            values = [qv.get(metric, 0.0) for qv in quality_vectors]
            aggregated[metric] = np.mean(values)
        
        return aggregated
    
    def batch_get_bin_vectors(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get bin vectors for multiple rows efficiently.
        
        Args:
            data: DataFrame with multiple rows
            
        Returns:
            Array of shape (n_rows, n_features) with bin IDs
        """
        n_rows = len(data)
        bin_vectors = np.zeros((n_rows, self.n_features), dtype=np.int32)
        
        for i, attr in enumerate(self.feature_order):
            if attr not in data.columns:
                continue
            
            values = data[attr].values
            
            if attr in self.numeric_boundaries:
                # Vectorized binary search for numeric
                boundaries = self.numeric_boundaries[attr]
                bin_ids = np.searchsorted(boundaries, values, side='right') - 1
                bin_ids = np.clip(bin_ids, 0, len(boundaries) - 2)
                bin_vectors[:, i] = bin_ids
                
            elif attr in self.categorical_maps:
                # Lookup for categorical
                cat_map = self.categorical_maps[attr]
                for j, value in enumerate(values):
                    if pd.isna(value):
                        bin_vectors[j, i] = 0
                    else:
                        bin_vectors[j, i] = cat_map.get(value, 0)
        
        return bin_vectors
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("Bin Dictionary Summary")
        lines.append("=" * 60)
        lines.append(f"Features: {self.n_features}")
        lines.append(f"Total Bins: {self.total_bins}")
        lines.append("")
        lines.append("Feature Details:")
        lines.append(f"{'Feature':<20} {'Type':<12} {'Bins':>6}")
        lines.append("-" * 38)
        
        for attr in self.feature_order:
            if attr in self.numeric_boundaries:
                n_bins = len(self.numeric_boundaries[attr]) - 1
                ftype = "Numeric"
            else:
                n_bins = len(self.categorical_maps[attr])
                ftype = "Categorical"
            
            lines.append(f"{attr:<20} {ftype:<12} {n_bins:>6}")
        
        return "\n".join(lines)


@dataclass
class TrainingData:
    """Training data for learned index model."""
    X: np.ndarray  # Shape: (n_samples, n_features) - bin vectors
    Y: np.ndarray  # Shape: (n_samples, n_metrics) - quality vectors
    feature_names: List[str]
    metric_names: List[str]
    n_samples: int = 0
    n_features: int = 0
    n_metrics: int = 0
    
    def __post_init__(self):
        self.n_samples = len(self.X)
        self.n_features = self.X.shape[1] if len(self.X.shape) > 1 else 0
        self.n_metrics = self.Y.shape[1] if len(self.Y.shape) > 1 else 0


def generate_training_data(data: pd.DataFrame,
                          bin_dictionary: BinDictionary,
                          metric_names: Optional[List[str]] = None) -> TrainingData:
    """
    Generate (X, Y) training data for learned index model.
    
    Process:
    1. For each tuple t in dataset
    2. X = bin_dictionary.get_bin_vector(t) 
    3. Y = aggregated quality vector from assigned bins
    4. Collect into training arrays
    
    Args:
        data: Full dataset
        bin_dictionary: BinDictionary from WP 7
        metric_names: Quality metrics to include (if None, use all from dictionary)
        
    Returns:
        TrainingData with X (bin vectors) and Y (quality vectors)
    """
    n_samples = len(data)
    n_features = bin_dictionary.n_features
    
    # Infer metric names from dictionary
    if metric_names is None:
        # Get metric names from first quality vector
        sample_key = next(iter(bin_dictionary.bin_quality_vectors.keys()))
        metric_names = list(bin_dictionary.bin_quality_vectors[sample_key].keys())
    
    n_metrics = len(metric_names)
    
    # Initialize arrays
    X = np.zeros((n_samples, n_features), dtype=np.int32)
    Y = np.zeros((n_samples, n_metrics), dtype=np.float32)
    
    # Efficient batch processing
    X = bin_dictionary.batch_get_bin_vectors(data)
    
    # Get quality vectors for each row
    for i in range(n_samples):
        bin_vector = X[i]
        quality_vector = bin_dictionary.get_quality_vector(bin_vector)
        
        for j, metric in enumerate(metric_names):
            Y[i, j] = quality_vector.get(metric, 0.0)
    
    return TrainingData(
        X=X,
        Y=Y,
        feature_names=bin_dictionary.feature_order,
        metric_names=metric_names
    )


def save_training_data(training_data: TrainingData, path: str):
    """
    Save training data to disk.
    
    Args:
        training_data: TrainingData to save
        path: Base path (without extension)
    """
    np.savez(
        f"{path}.npz",
        X=training_data.X,
        Y=training_data.Y,
        feature_names=training_data.feature_names,
        metric_names=training_data.metric_names
    )


def load_training_data(path: str) -> TrainingData:
    """
    Load training data from disk.
    
    Args:
        path: Path to .npz file
        
    Returns:
        TrainingData object
    """
    data = np.load(path, allow_pickle=True)
    
    return TrainingData(
        X=data['X'],
        Y=data['Y'],
        feature_names=list(data['feature_names']),
        metric_names=list(data['metric_names'])
    )
