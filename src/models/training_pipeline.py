import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.models.model_factory import ModelFactory, TextClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """
    Tracks and manages experiment metrics and state during model training.

    This class handles:
    - Model parameter tracking
    - Training and validation metrics
    - Feature importance tracking
    - Error analysis results
    """

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = {
            'training_metrics': [],
            'validation_metrics': [],
            'feature_importance': [],
            'error_analysis': []
        }
        self.logger = get_logger(self.__class__.__name__)

    def log_metrics(self,
                    metrics: Dict[str, Any],
                    step: int,
                    metric_type: str = 'training_metrics') -> None:
        """Log metrics for current training step."""
        metrics['step'] = step
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history[metric_type].append(metrics)

    def log_feature_importance(self,
                               feature_importance: Dict[str, float],
                               step: int) -> None:
        """Log feature importance scores."""
        self.log_metrics(
            {'feature_importance': feature_importance},
            step,
            'feature_importance'
        )

    def log_error_analysis(self,
                           error_analysis: Dict[str, Any],
                           step: int) -> None:
        """Log error analysis results."""
        self.log_metrics(
            error_analysis,
            step,
            'error_analysis'
        )

    def save_experiment_state(self, experiment_id: str) -> None:
        """Save current experiment state to disk."""
        state_path = self.experiment_dir / f"{experiment_id}_state.json"
        with open(state_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def load_experiment_state(self, experiment_id: str) -> None:
        """Load experiment state from disk."""
        state_path = self.experiment_dir / f"{experiment_id}_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                self.metrics_history = json.load(f)


class ValidationManager:
    """
    Manages model validation and evaluation during training.

    This class handles:
    - Cross-validation setup and execution
    - Model evaluation and metric calculation
    - Validation result aggregation
    """

    def __init__(self,
                 n_splits: int = 5,
                 random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        self.logger = get_logger(self.__class__.__name__)

    def get_cv_splits(self,
                      X: np.ndarray,
                      y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate cross-validation splits."""
        return list(self.cv.split(X, y))

    def evaluate_fold(self,
                      model: TextClassifier,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluate model performance on a single fold."""
        # Train model on fold
        model.fit(X_train, y_train)

        # Get predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        val_metrics = self._calculate_metrics(y_val, y_pred_val)

        return train_metrics, val_metrics

    def _calculate_metrics(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred)
        }


class FeatureAnalyzer:
    """
    Analyzes and tracks feature importance during training.

    This class handles:
    - Feature importance calculation
    - Feature selection
    - Feature ranking and visualization
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def get_feature_importance(self,
                               model: TextClassifier,
                               feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get feature importance scores from model."""
        importance_scores = {}

        try:
            # Get raw importance scores
            if hasattr(model.model, 'feature_importances_'):
                scores = model.model.feature_importances_
            elif hasattr(model.model, 'coef_'):
                scores = np.abs(model.model.coef_[0])
            else:
                return {}

            # Map scores to feature names
            for idx, score in enumerate(scores):
                feature_name = (
                    feature_names[idx] if feature_names
                    else f'feature_{idx}'
                )
                importance_scores[feature_name] = float(score)

            # Sort by importance
            importance_scores = dict(
                sorted(
                    importance_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")

        return importance_scores

    def analyze_feature_correlations(self,
                                     X: np.ndarray,
                                     feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze correlations between features."""
        try:
            df = pd.DataFrame(
                X,
                columns=feature_names if feature_names else [f'feature_{i}' for i in range(X.shape[1])]
            )
            correlations = df.corr().to_dict()
            return {'feature_correlations': correlations}
        except Exception as e:
            self.logger.error(f"Error analyzing feature correlations: {str(e)}")
            return {}


class TrainingPipeline:
    """
    Complete training pipeline for text classification models.

    This pipeline manages:
    - Model training and validation
    - Experiment tracking
    - Feature analysis
    - Error analysis
    - Model persistence
    """

    def __init__(self,
                 model_type: str = 'svm',
                 model_params: Optional[Dict[str, Any]] = None,
                 experiment_dir: Optional[str] = None,
                 n_splits: int = 5,
                 random_state: int = 42):
        """Initialize training pipeline."""
        self.model_type = model_type
        self.model_params = model_params
        self.experiment_dir = Path(experiment_dir or 'experiments')
        self.n_splits = n_splits
        self.random_state = random_state

        # Initialize components
        self.model = None
        self.experiment_tracker = ExperimentTracker(self.experiment_dir)
        self.validation_manager = ValidationManager(n_splits, random_state)
        self.feature_analyzer = FeatureAnalyzer()
        self.logger = get_logger(self.__class__.__name__)

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              feature_names: Optional[List[str]] = None,
              experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Train model with full pipeline including validation and analysis."""
        try:
            start_time = time.time()

            # Generate experiment ID if not provided
            if experiment_id is None:
                experiment_id = f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.logger.info(f"Starting training pipeline for experiment {experiment_id}")
            self.logger.info(f"Model type: {self.model_type}")
            self.logger.info(f"Training data shape: X={X.shape}, y={y.shape}")

            # Create and initialize model
            self.model = ModelFactory.create_model(
                model_type=self.model_type,
                model_config=self.model_params
            )

            # Get cross-validation splits
            cv_splits = self.validation_manager.get_cv_splits(X, y)

            # Train and evaluate on each fold
            fold_results = []
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                self.logger.info(f"\nTraining fold {fold + 1}/{self.n_splits}")

                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train and evaluate
                train_metrics, val_metrics = self.validation_manager.evaluate_fold(
                    self.model, X_train, y_train, X_val, y_val
                )

                # Get feature importance
                feature_importance = self.feature_analyzer.get_feature_importance(
                    self.model, feature_names
                )

                # Log metrics
                self.experiment_tracker.log_metrics(train_metrics, fold, 'training_metrics')
                self.experiment_tracker.log_metrics(val_metrics, fold, 'validation_metrics')
                self.experiment_tracker.log_feature_importance(feature_importance, fold)

                fold_results.append({
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'feature_importance': feature_importance
                })

                # Log fold results
                self.logger.info(f"Fold {fold + 1} Results:")
                self.logger.info(f"Train accuracy: {train_metrics['accuracy']:.4f}")
                self.logger.info(f"Validation accuracy: {val_metrics['accuracy']:.4f}")

            # Train final model on full dataset
            self.logger.info("\nTraining final model on full dataset")
            self.model.fit(X, y)

            # Calculate final metrics
            final_predictions = self.model.predict(X)
            final_metrics = self.validation_manager._calculate_metrics(
                y, final_predictions
            )

            # Get final feature importance
            final_feature_importance = self.feature_analyzer.get_feature_importance(
                self.model, feature_names
            )

            # Analyze feature correlations
            feature_correlations = self.feature_analyzer.analyze_feature_correlations(
                X, feature_names
            )

            # Prepare final results
            training_time = time.time() - start_time
            final_results = {
                'experiment_id': experiment_id,
                'model_type': self.model_type,
                'training_time': training_time,
                'final_metrics': final_metrics,
                'feature_importance': final_feature_importance,
                'feature_correlations': feature_correlations,
                'cross_validation': {
                    'n_splits': self.n_splits,
                    'fold_results': fold_results
                }
            }

            # Save experiment state
            self.experiment_tracker.save_experiment_state(experiment_id)

            # Save final model
            model_path = self.experiment_dir / f"{experiment_id}_model.joblib"
            self.model.save(model_path)

            self.logger.info("\nTraining pipeline completed successfully")
            self.logger.info(f"Total training time: {training_time:.2f} seconds")
            self.logger.info(f"Final accuracy: {final_metrics['accuracy']:.4f}")

            return final_results

        except Exception as e:
            self.logger.error(f"Error in training pipeline: {str(e)}")
            raise RuntimeError(f"Training pipeline failed: {str(e)}")

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load saved experiment state and model."""
        try:
            # Load experiment state
            self.experiment_tracker.load_experiment_state(experiment_id)

            # Load model
            model_path = self.experiment_dir / f"{experiment_id}_model.joblib"
            self.model = ModelFactory.load_model(model_path)

            return self.experiment_tracker.metrics_history

        except Exception as e:
            self.logger.error(f"Error loading experiment: {str(e)}")
            raise RuntimeError(f"Failed to load experiment: {str(e)}")

    def get_feature_rankings(self,
                             top_k: Optional[int] = None) -> Dict[str, float]:
        """Get feature importance rankings."""
        if self.model is None:
            raise RuntimeError("Model must be trained first")

        importance_scores = self.feature_analyzer.get_feature_importance(self.model)

        if top_k:
            importance_scores = dict(
                list(importance_scores.items())[:top_k]
            )

        return importance_scores
