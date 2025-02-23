import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from src.models.model_factory import ModelFactory
from src.models.model_evaluation import ModelEvaluator
from src.models.training_pipeline import TrainingPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelExperiments:
    """
    Conducts comprehensive model experiments and hyperparameter tuning.

    Features:
    - Data preprocessing and scaling
    - Model evaluation with different configurations
    - Hyperparameter tuning
    - Performance comparison with confusion matrices
    - Result analysis and visualization
    """

    def __init__(self,
                 data_dir: str = '../../data/prepared_data',
                 cache_dir: str = '../../data/cache',
                 experiment_dir: str = 'experiments',
                 random_state: int = 42):
        """Initialize experiment infrastructure."""
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Initialize components
        self.evaluator = ModelEvaluator()
        self.experiment_results = {}
        self.logger = get_logger(self.__class__.__name__)

        # Load and preprocess data
        self._load_data()
        self._preprocess_data()

    def _load_data(self):
        """Load prepared data and metadata."""
        try:
            # Load train/val/test data
            self.X_train = np.load(self.data_dir / 'train_features.npy')
            self.y_train = np.load(self.data_dir / 'train_labels.npy')
            self.X_val = np.load(self.data_dir / 'val_features.npy')
            self.y_val = np.load(self.data_dir / 'val_labels.npy')
            self.X_test = np.load(self.data_dir / 'test_features.npy')
            self.y_test = np.load(self.data_dir / 'test_labels.npy')

            # Load metadata
            self.metadata = joblib.load(self.data_dir / 'metadata.joblib')

            self.logger.info("Data loaded successfully")
            self.logger.info(f"Training data shape: {self.X_train.shape}")
            self.logger.info(f"Validation data shape: {self.X_val.shape}")
            self.logger.info(f"Test data shape: {self.X_test.shape}")

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def _preprocess_data(self):
        """Preprocess data for model training."""
        try:
            # Standard scaling for most models
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_val_scaled = self.scaler.transform(self.X_val)
            self.X_test_scaled = self.scaler.transform(self.X_test)

            # MinMax scaling for Naive Bayes (ensures non-negative values)
            self.minmax = MinMaxScaler()
            self.X_train_minmax = self.minmax.fit_transform(self.X_train)
            self.X_val_minmax = self.minmax.transform(self.X_val)
            self.X_test_minmax = self.minmax.transform(self.X_test)

            self.logger.info("Data preprocessing completed")

        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise RuntimeError(f"Data preprocessing failed: {str(e)}")

    def get_model_data(self, model_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get appropriate preprocessed data for model type."""
        if model_type == 'naive_bayes':
            return self.X_train_minmax, self.X_val_minmax, self.X_test_minmax
        else:
            return self.X_train_scaled, self.X_val_scaled, self.X_test_scaled

    def run_initial_experiments(self):
        """Run initial experiments with default configurations."""
        model_types = ModelFactory.get_supported_models().keys()

        for model_type in model_types:
            self.logger.info(f"\nRunning initial experiment for {model_type}")
            try:
                # Get appropriate data
                X_train, X_val, X_test = self.get_model_data(model_type)

                # Create and train model
                model = ModelFactory.create_model(model_type)
                model.fit(X_train, self.y_train)

                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                val_metrics = {
                    'accuracy': accuracy_score(self.y_val, val_predictions),
                    'precision': precision_score(self.y_val, val_predictions, average='weighted'),
                    'recall': recall_score(self.y_val, val_predictions, average='weighted'),
                    'f1': f1_score(self.y_val, val_predictions, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_val, val_predictions)
                }

                # Evaluate on test set
                test_predictions = model.predict(X_test)
                test_metrics = {
                    'accuracy': accuracy_score(self.y_test, test_predictions),
                    'precision': precision_score(self.y_test, test_predictions, average='weighted'),
                    'recall': recall_score(self.y_test, test_predictions, average='weighted'),
                    'f1': f1_score(self.y_test, test_predictions, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_test, test_predictions)
                }

                # Store results
                self.experiment_results[model_type] = {
                    'initial_results': {
                        'validation_metrics': val_metrics,
                        'test_metrics': test_metrics,
                        'model_config': model.get_params()
                    }
                }

                # Plot confusion matrices
                self._plot_confusion_matrices(
                    model_type,
                    val_metrics['confusion_matrix'],
                    test_metrics['confusion_matrix'],
                    'initial'
                )

                self.logger.info(f"\nResults for {model_type}:")
                self.logger.info(f"Validation Metrics:")
                self._log_metrics(val_metrics)
                self.logger.info(f"\nTest Metrics:")
                self._log_metrics(test_metrics)

            except Exception as e:
                self.logger.error(f"Error in experiment for {model_type}: {str(e)}")
                continue

    def tune_hyperparameters(self):
        """Perform hyperparameter tuning for each model type."""
        # Define parameter grids for each model type
        param_grids = {
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'fit_prior': [True, False]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [2000],
                'solver': ['lbfgs', 'saga'],
                'multi_class': ['multinomial'],
                'class_weight': ['balanced']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear'],
                'probability': [True],
                'class_weight': ['balanced']
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [None],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced']
            },
            'bagging': {
                'n_estimators': [50, 100],
                'max_samples': [0.5, 0.7, 1.0],
                'max_features': [0.5, 0.7, 1.0]
            }
        }

        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1': make_scorer(f1_score, average='weighted')
        }

        for model_type, param_grid in param_grids.items():
            self.logger.info(f"\nTuning hyperparameters for {model_type}")
            try:
                # Get appropriate data
                X_train, X_val, X_test = self.get_model_data(model_type)

                # Create base model
                base_model_obj = ModelFactory.create_model(model_type)
                base_model = base_model_obj.model

                # Perform grid search
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    scoring=scoring,
                    refit='f1',
                    cv=5,
                    n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, self.y_train)

                # Get best parameters
                best_params = grid_search.best_params_

                # Convert best_params into a serializable dictionary
                best_params_serializable = {}
                for key, value in best_params.items():
                    try:
                        json.dumps(value)
                        best_params_serializable[key] = value
                    except TypeError:
                        best_params_serializable[key] = repr(value)

                # Create best model using the best parameters from grid search
                best_model = ModelFactory.create_model(
                    model_type=model_type,
                    model_config=best_params
                )
                best_model.fit(X_train, self.y_train)

                # Evaluate on validation set
                val_predictions = best_model.predict(X_val)
                val_metrics = {
                    'accuracy': accuracy_score(self.y_val, val_predictions),
                    'precision': precision_score(self.y_val, val_predictions, average='weighted'),
                    'recall': recall_score(self.y_val, val_predictions, average='weighted'),
                    'f1': f1_score(self.y_val, val_predictions, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_val, val_predictions)
                }

                # Evaluate on test set
                test_predictions = best_model.predict(X_test)
                test_metrics = {
                    'accuracy': accuracy_score(self.y_test, test_predictions),
                    'precision': precision_score(self.y_test, test_predictions, average='weighted'),
                    'recall': recall_score(self.y_test, test_predictions, average='weighted'),
                    'f1': f1_score(self.y_test, test_predictions, average='weighted'),
                    'confusion_matrix': confusion_matrix(self.y_test, test_predictions)
                }

                # Now store tuning results with the serializable best parameters
                self.experiment_results[model_type]['tuned_results'] = {
                    'best_params': best_params_serializable,
                    'cv_results': grid_search.cv_results_,
                    'validation_metrics': val_metrics,
                    'test_metrics': test_metrics
                }

                # Plot confusion matrices
                self._plot_confusion_matrices(
                    model_type,
                    val_metrics['confusion_matrix'],
                    test_metrics['confusion_matrix'],
                    'tuned'
                )

                self.logger.info(f"\nBest parameters for {model_type}:")
                self.logger.info(json.dumps(best_params_serializable, indent=2))
                self.logger.info(f"\nValidation Metrics:")
                self._log_metrics(val_metrics)
                self.logger.info(f"\nTest Metrics:")
                self._log_metrics(test_metrics)

            except Exception as e:
                self.logger.error(f"Error tuning {model_type}: {str(e)}")
                continue

    def _plot_confusion_matrices(self,
                                 model_type: str,
                                 val_cm: np.ndarray,
                                 test_cm: np.ndarray,
                                 stage: str):
        """Plot confusion matrices for validation and test sets."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'{model_type} Validation Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title(f'{model_type} Test Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.experiment_dir / f'{model_type}_{stage}_confusion_matrices.png')
        plt.close()

    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics."""
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"F1 Score: {metrics['f1']:.4f}")

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results and select best model."""
        analysis = {
            'model_comparisons': {},
            'best_model': None,
            'selection_rationale': {}
        }
        for model_type, results in self.experiment_results.items():
            initial_metrics = results['initial_results']['test_metrics']
            if 'tuned_results' in results:
                tuned_metrics = results['tuned_results']['test_metrics']
            else:
                tuned_metrics = None

            analysis['model_comparisons'][model_type] = {
                'initial_performance': {
                    'accuracy': initial_metrics['accuracy'],
                    'precision': initial_metrics['precision'],
                    'recall': initial_metrics['recall'],
                    'f1': initial_metrics['f1'],
                    'confusion_matrix': initial_metrics['confusion_matrix'].tolist()
                }
            }
            if tuned_metrics:
                analysis['model_comparisons'][model_type]['tuned_performance'] = {
                    'accuracy': tuned_metrics['accuracy'],
                    'precision': tuned_metrics['precision'],
                    'recall': tuned_metrics['recall'],
                    'f1': tuned_metrics['f1'],
                    'confusion_matrix': tuned_metrics['confusion_matrix'].tolist(),
                    'improvement': {
                        'accuracy': tuned_metrics['accuracy'] - initial_metrics['accuracy'],
                        'precision': tuned_metrics['precision'] - initial_metrics['precision'],
                        'recall': tuned_metrics['recall'] - initial_metrics['recall'],
                        'f1': tuned_metrics['f1'] - initial_metrics['f1']
                    }
                }
        best_f1 = 0
        best_model = None
        best_is_tuned = False
        for model_type, comparison in analysis['model_comparisons'].items():
            initial_f1 = comparison['initial_performance']['f1']
            tuned_f1 = comparison.get('tuned_performance', {}).get('f1', 0)
            if tuned_f1 > best_f1:
                best_f1 = tuned_f1
                best_model = model_type
                best_is_tuned = True
            elif initial_f1 > best_f1:
                best_f1 = initial_f1
                best_model = model_type
                best_is_tuned = False
        analysis['best_model'] = {
            'model_type': best_model,
            'is_tuned': best_is_tuned,
            'f1_score': best_f1,
            'configuration': (
                self.experiment_results[best_model]['tuned_results']['best_params']
                if best_is_tuned else
                self.experiment_results[best_model]['initial_results']['model_config']
            )
        }
        analysis['selection_rationale'] = {
            'metric_used': 'F1 Score (weighted)',
            'best_score': best_f1,
            'considerations': [
                'Overall performance across all metrics',
                'Performance stability between validation and test sets',
                'Improvement from hyperparameter tuning',
                'Confusion matrix analysis'
            ],
            'confusion_matrix_analysis': self._analyze_confusion_matrix(
                self.experiment_results[best_model][
                    'tuned_results' if best_is_tuned else 'initial_results'
                ]['test_metrics']['confusion_matrix']
            )
        }
        analysis_path = self.experiment_dir / 'model_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        self.logger.info("\nModel Analysis Results:")
        self.logger.info(f"Best Model: {best_model} ({'tuned' if best_is_tuned else 'initial'})")
        self.logger.info(f"Best F1 Score: {best_f1:.4f}")
        return analysis

    def _analyze_confusion_matrix(self, cm: np.ndarray) -> Dict[str, Any]:
        """Perform detailed analysis of confusion matrix."""
        n_classes = cm.shape[0]
        per_class_metrics = {}
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            per_class_metrics[f'class_{i}'] = {
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn),
                'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0
            }
        confused_pairs = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                confused_count = cm[i, j] + cm[j, i]
                if confused_count > 0:
                    confused_pairs.append({
                        'classes': (f'class_{i}', f'class_{j}'),
                        'confusion_count': int(confused_count)
                    })
        confused_pairs.sort(key=lambda x: x['confusion_count'], reverse=True)
        return {
            'per_class_metrics': per_class_metrics,
            'most_confused_pairs': confused_pairs[:3],
            'overall_accuracy': float(np.trace(cm) / np.sum(cm))
        }

    def save_experiment_results(self):
        """Save all experiment results."""
        save_path = self.experiment_dir / 'experiment_results.joblib'
        joblib.dump(self.experiment_results, save_path)
        self.logger.info(f"Experiment results saved to {save_path}")

    def load_experiment_results(self):
        """Load saved experiment results."""
        load_path = self.experiment_dir / 'experiment_results.joblib'
        if load_path.exists():
            self.experiment_results = joblib.load(load_path)
            self.logger.info(f"Experiment results loaded from {load_path}")
        else:
            self.logger.warning("No saved experiment results found")


if __name__ == "__main__":
    experiments = ModelExperiments()
    experiments.run_initial_experiments()
    experiments.tune_hyperparameters()
    analysis = experiments.analyze_results()
    experiments.save_experiment_results()
