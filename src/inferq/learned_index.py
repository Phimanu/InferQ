"""
WP 8: Stage 3 - Model Training and Index Class

Implements the learned index: trains Random Forest model and provides
unified QualityIndex interface for quality predictions.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from .bin_dictionary import BinDictionary, TrainingData


@dataclass
class ModelMetrics:
    """Evaluation metrics for trained model."""
    train_mse: float
    test_mse: float
    train_r2: float
    test_r2: float
    train_mae: float
    test_mae: float
    per_metric_r2: Dict[str, float]
    per_metric_mse: Dict[str, float]


def train_model(training_data: TrainingData,
                test_size: float = 0.2,
                n_estimators: int = 100,
                max_depth: Optional[int] = None,
                random_state: int = 42,
                n_jobs: int = -1,
                verbose: bool = True) -> Tuple[RandomForestRegressor, ModelMetrics]:
    """
    Train Random Forest regressor for quality prediction.
    
    Args:
        training_data: TrainingData from WP 7
        test_size: Fraction of data for testing
        n_estimators: Number of trees in forest
        max_depth: Maximum depth of trees (None = unlimited)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = use all cores)
        verbose: Print training progress
        
    Returns:
        (trained_model, metrics)
    """
    if verbose:
        print(f"Training Random Forest model...")
        print(f"  Training data: {training_data.X.shape}")
        print(f"  Features: {training_data.n_features}")
        print(f"  Target metrics: {training_data.n_metrics}")
        print(f"  Estimators: {n_estimators}, Max depth: {max_depth}")
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        training_data.X,
        training_data.Y,
        test_size=test_size,
        random_state=random_state
    )
    
    if verbose:
        print(f"  Split: {len(X_train)} train, {len(X_test)} test")
    
    # Initialize Random Forest with multi-output support
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1 if verbose else 0
    )
    
    # Train model
    model.fit(X_train, Y_train)
    
    if verbose:
        print(f"  Training complete!")
    
    # Evaluate
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    
    # Overall metrics
    train_mse = mean_squared_error(Y_train, Y_train_pred)
    test_mse = mean_squared_error(Y_test, Y_test_pred)
    train_r2 = r2_score(Y_train, Y_train_pred)
    test_r2 = r2_score(Y_test, Y_test_pred)
    train_mae = mean_absolute_error(Y_train, Y_train_pred)
    test_mae = mean_absolute_error(Y_test, Y_test_pred)
    
    # Per-metric evaluation
    per_metric_r2 = {}
    per_metric_mse = {}
    
    for i, metric_name in enumerate(training_data.metric_names):
        r2 = r2_score(Y_test[:, i], Y_test_pred[:, i])
        mse = mean_squared_error(Y_test[:, i], Y_test_pred[:, i])
        per_metric_r2[metric_name] = r2
        per_metric_mse[metric_name] = mse
    
    metrics = ModelMetrics(
        train_mse=train_mse,
        test_mse=test_mse,
        train_r2=train_r2,
        test_r2=test_r2,
        train_mae=train_mae,
        test_mae=test_mae,
        per_metric_r2=per_metric_r2,
        per_metric_mse=per_metric_mse
    )
    
    if verbose:
        print(f"\n  Model Performance:")
        print(f"    Test R²:  {test_r2:.4f}")
        print(f"    Test MSE: {test_mse:.6f}")
        print(f"    Test MAE: {test_mae:.6f}")
    
    return model, metrics


def save_model(model: RandomForestRegressor, 
               path: str,
               bin_dictionary: Optional[BinDictionary] = None,
               metrics: Optional[ModelMetrics] = None):
    """
    Save trained model to disk.
    
    Args:
        model: Trained Random Forest model
        path: Path to save (without extension)
        bin_dictionary: Optional BinDictionary to save with model
        metrics: Optional ModelMetrics to save
    """
    model_data = {
        'model': model,
        'bin_dictionary': bin_dictionary,
        'metrics': metrics
    }
    
    with open(f"{path}.pkl", 'wb') as f:
        pickle.dump(model_data, f)


def load_model(path: str) -> Tuple[RandomForestRegressor, Optional[BinDictionary], Optional[ModelMetrics]]:
    """
    Load trained model from disk.
    
    Args:
        path: Path to model file
        
    Returns:
        (model, bin_dictionary, metrics)
    """
    with open(path, 'rb') as f:
        model_data = pickle.load(f)
    
    return (
        model_data['model'],
        model_data.get('bin_dictionary'),
        model_data.get('metrics')
    )


class QualityIndex:
    """
    Learned index for quality prediction.
    
    Combines BinDictionary for fast lookups and trained Random Forest
    for quality predictions. Provides unified interface for quality monitoring.
    """
    
    def __init__(self,
                 bin_dictionary: BinDictionary,
                 model: RandomForestRegressor,
                 metrics: Optional[ModelMetrics] = None):
        """
        Initialize QualityIndex.
        
        Args:
            bin_dictionary: BinDictionary from WP 7
            model: Trained Random Forest model
            metrics: Optional training metrics
        """
        self.bin_dictionary = bin_dictionary
        self.model = model
        self.metrics = metrics
        
        # Infer metric names from dictionary
        sample_key = next(iter(bin_dictionary.bin_quality_vectors.keys()))
        self.metric_names = list(bin_dictionary.bin_quality_vectors[sample_key].keys())
    
    @classmethod
    def from_training_data(cls,
                          training_data: TrainingData,
                          bin_dictionary: BinDictionary,
                          **train_kwargs) -> 'QualityIndex':
        """
        Build QualityIndex by training model on training data.
        
        Args:
            training_data: TrainingData from WP 7
            bin_dictionary: BinDictionary from WP 7
            **train_kwargs: Arguments for train_model()
            
        Returns:
            QualityIndex ready for predictions
        """
        model, metrics = train_model(training_data, **train_kwargs)
        return cls(bin_dictionary, model, metrics)
    
    @classmethod
    def load(cls, path: str) -> 'QualityIndex':
        """
        Load QualityIndex from disk.
        
        Args:
            path: Path to saved index
            
        Returns:
            QualityIndex
        """
        model, bin_dictionary, metrics = load_model(path)
        
        if bin_dictionary is None:
            raise ValueError("Saved model does not include BinDictionary")
        
        return cls(bin_dictionary, model, metrics)
    
    def save(self, path: str):
        """
        Save QualityIndex to disk.
        
        Args:
            path: Path to save (without extension)
        """
        save_model(self.model, path, self.bin_dictionary, self.metrics)
    
    def predict(self, row: Union[pd.Series, Dict]) -> Dict[str, float]:
        """
        Predict quality metrics for a single data row.
        
        Args:
            row: Data row as Series or dict
            
        Returns:
            Dict mapping metric names to predicted values
        """
        # Get bin vector
        bin_vector = self.bin_dictionary.get_bin_vector(row)
        
        # Predict quality
        quality_prediction = self.model.predict([bin_vector])[0]
        
        # Map to metric names
        return {
            metric: float(value)
            for metric, value in zip(self.metric_names, quality_prediction)
        }
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict quality metrics for multiple rows efficiently.
        
        Args:
            data: DataFrame with multiple rows
            
        Returns:
            DataFrame with predicted quality metrics
        """
        # Get bin vectors
        bin_vectors = self.bin_dictionary.batch_get_bin_vectors(data)
        
        # Predict quality
        quality_predictions = self.model.predict(bin_vectors)
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(
            quality_predictions,
            columns=self.metric_names,
            index=data.index
        )
        
        return predictions_df
    
    def predict_with_bin_info(self, row: Union[pd.Series, Dict]) -> Dict[str, Any]:
        """
        Predict quality with additional bin assignment information.
        
        Args:
            row: Data row as Series or dict
            
        Returns:
            Dict with 'prediction', 'bin_vector', and 'features'
        """
        bin_vector = self.bin_dictionary.get_bin_vector(row)
        quality_prediction = self.model.predict([bin_vector])[0]
        
        return {
            'prediction': {
                metric: float(value)
                for metric, value in zip(self.metric_names, quality_prediction)
            },
            'bin_vector': bin_vector.tolist(),
            'features': self.bin_dictionary.feature_order
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from Random Forest.
        
        Returns:
            Dict mapping feature names to importance scores
        """
        importances = self.model.feature_importances_
        
        return {
            feature: float(importance)
            for feature, importance in zip(
                self.bin_dictionary.feature_order,
                importances
            )
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("Quality Index Summary")
        lines.append("=" * 60)
        lines.append(f"Features: {self.bin_dictionary.n_features}")
        lines.append(f"Total Bins: {self.bin_dictionary.total_bins}")
        lines.append(f"Quality Metrics: {len(self.metric_names)}")
        lines.append(f"Model: Random Forest ({self.model.n_estimators} estimators)")
        
        if self.metrics:
            lines.append("")
            lines.append("Model Performance:")
            lines.append(f"  Test R²:  {self.metrics.test_r2:.4f}")
            lines.append(f"  Test MSE: {self.metrics.test_mse:.6f}")
            lines.append(f"  Test MAE: {self.metrics.test_mae:.6f}")
        
        lines.append("")
        lines.append("Feature Importance:")
        importances = self.get_feature_importance()
        for feature, importance in sorted(importances.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True):
            lines.append(f"  {feature:<20} {importance:.4f}")
        
        return "\n".join(lines)
    
    def evaluate(self, data: pd.DataFrame, true_quality: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate index performance on test data.
        
        Args:
            data: Test data features
            true_quality: True quality values
            
        Returns:
            Dict with evaluation metrics
        """
        predictions = self.predict_batch(data)
        
        # Align columns
        common_metrics = [m for m in self.metric_names if m in true_quality.columns]
        
        mse = mean_squared_error(
            true_quality[common_metrics],
            predictions[common_metrics]
        )
        r2 = r2_score(
            true_quality[common_metrics],
            predictions[common_metrics]
        )
        mae = mean_absolute_error(
            true_quality[common_metrics],
            predictions[common_metrics]
        )
        
        return {
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'n_samples': len(data),
            'n_metrics': len(common_metrics)
        }
