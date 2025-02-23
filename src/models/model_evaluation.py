import numpy as np
import pandas as pd
import psutil
import tracemalloc
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from src.models.model_factory import ModelFactory, TextClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation system with detailed metrics and analysis.

    Features:
    - Cross-validation metrics calculation
    - Confusion matrix analysis
    - ROC curves per class
    - Detailed error analysis
    - Resource usage tracking
    """

    def __init__(self,
                 n_splits: int = 5,
                 random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = get_logger(self.__class__.__name__)

    def evaluate_model(self,
                       model: TextClassifier,
                       X: np.ndarray,
                       y: np.ndarray,
                       feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""
        evaluation_results = {
            'cross_validation_metrics': self._perform_cross_validation(model, X, y),
            'confusion_matrix_analysis': self._analyze_confusion_matrix(model, X, y),
            'roc_analysis': self._perform_roc_analysis(model, X, y),
            'error_analysis': self._perform_error_analysis(model, X, y, feature_names),
            'performance_metrics': self._calculate_performance_metrics(model, X, y)
        }

        return evaluation_results

    def _perform_cross_validation(self,
                                  model: TextClassifier,
                                  X: np.ndarray,
                                  y: np.ndarray) -> Dict[str, Any]:
        """Perform k-fold cross-validation."""
        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        # Initialize metric trackers
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Calculate metrics
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred, average='weighted'))
            metrics['recall'].append(recall_score(y_val, y_pred, average='weighted'))
            metrics['f1'].append(f1_score(y_val, y_pred, average='weighted'))

        # Calculate statistics
        cv_results = {
            'mean': {
                metric: float(np.mean(scores))
                for metric, scores in metrics.items()
            },
            'std': {
                metric: float(np.std(scores))
                for metric, scores in metrics.items()
            },
            'confidence_intervals': {
                metric: (
                    float(np.mean(scores) - 1.96 * np.std(scores) / np.sqrt(self.n_splits)),
                    float(np.mean(scores) + 1.96 * np.std(scores) / np.sqrt(self.n_splits))
                )
                for metric, scores in metrics.items()
            }
        }

        return cv_results

    def _analyze_confusion_matrix(self,
                                  model: TextClassifier,
                                  X: np.ndarray,
                                  y: np.ndarray) -> Dict[str, Any]:
        """Perform detailed confusion matrix analysis."""
        # Get predictions
        y_pred = model.predict(X)

        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Calculate per-class metrics
        class_metrics = {}
        for i, class_label in enumerate(model.classes_):
            true_pos = cm[i, i]
            false_pos = cm[:, i].sum() - true_pos
            false_neg = cm[i, :].sum() - true_pos
            true_neg = cm.sum() - (true_pos + false_pos + false_neg)

            class_metrics[str(class_label)] = {
                'true_positives': int(true_pos),
                'false_positives': int(false_pos),
                'false_negatives': int(false_neg),
                'true_negatives': int(true_neg),
                'precision': float(precision_score(y, y_pred, labels=[class_label], average='weighted')),
                'recall': float(recall_score(y, y_pred, labels=[class_label], average='weighted')),
                'f1_score': float(f1_score(y, y_pred, labels=[class_label], average='weighted'))
            }

        return {
            'confusion_matrix': cm.tolist(),
            'class_metrics': class_metrics,
            'overall_accuracy': float(accuracy_score(y, y_pred)),
            'classification_report': classification_report(y, y_pred)
        }

    def _perform_roc_analysis(self,
                              model: TextClassifier,
                              X: np.ndarray,
                              y: np.ndarray) -> Dict[str, Any]:
        """Perform ROC curve analysis per class."""
        # Binarize the labels for ROC curve calculation
        y_bin = label_binarize(y, classes=model.classes_)
        n_classes = len(model.classes_)

        # Get probability predictions
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X)
        else:
            # For models without probability scores, use decision function if available
            if hasattr(model, 'decision_function'):
                y_score = model.decision_function(X)
                if y_score.ndim == 1:
                    y_score = np.column_stack([1 - y_score, y_score])
            else:
                return {
                    'error': 'Model does not support probability predictions or decision function'
                }

        # Calculate ROC curve and ROC area for each class
        roc_metrics = {}
        for i, class_label in enumerate(model.classes_):
            fpr, tpr, thresholds = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)

            roc_metrics[str(class_label)] = {
                'false_positive_rate': fpr.tolist(),
                'true_positive_rate': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(roc_auc)
            }

        # Calculate micro-average ROC curve and ROC area
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        return {
            'per_class_roc': roc_metrics,
            'micro_average': {
                'false_positive_rate': fpr_micro.tolist(),
                'true_positive_rate': tpr_micro.tolist(),
                'auc': float(roc_auc_micro)
            }
        }

    def _perform_error_analysis(self,
                                model: TextClassifier,
                                X: np.ndarray,
                                y: np.ndarray,
                                feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform detailed error analysis."""
        # Get predictions
        y_pred = model.predict(X)

        # Identify error cases
        error_mask = y_pred != y
        error_indices = np.where(error_mask)[0]

        # Analyze error cases
        error_analysis = {
            'total_errors': int(error_mask.sum()),
            'error_rate': float(error_mask.sum() / len(y)),
            'error_cases': []
        }

        # Analyze each error case
        for idx in error_indices:
            error_case = {
                'index': int(idx),
                'true_label': str(y[idx]),
                'predicted_label': str(y_pred[idx]),
                'confidence': None  # Will be updated if available
            }

            # Add prediction confidence if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X[idx].reshape(1, -1))
                error_case['confidence'] = float(np.max(proba))

            # Add feature importance if available
            if feature_names is not None and hasattr(model.model, 'feature_importances_'):
                top_features = np.argsort(model.model.feature_importances_)[-5:]
                error_case['important_features'] = [
                    {
                        'feature': feature_names[i],
                        'importance': float(model.model.feature_importances_[i])
                    }
                    for i in top_features
                ]

            error_analysis['error_cases'].append(error_case)

        # Calculate error patterns
        error_patterns = {}
        for true_label in model.classes_:
            for pred_label in model.classes_:
                if true_label != pred_label:
                    mask = (y == true_label) & (y_pred == pred_label)
                    pattern = f"{true_label}->{pred_label}"
                    error_patterns[pattern] = int(mask.sum())

        error_analysis['error_patterns'] = error_patterns

        return error_analysis

    def _calculate_performance_metrics(self,
                                       model: TextClassifier,
                                       X: np.ndarray,
                                       y: np.ndarray) -> Dict[str, Any]:
        """Calculate model performance metrics including timing and resource usage."""
        tracemalloc.start()
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        # Perform predictions
        y_pred = model.predict(X)

        # Calculate timing
        prediction_time = time.time() - start_time

        # Calculate memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_usage = psutil.Process().memory_info().rss - start_memory

        return {
            'prediction_time': float(prediction_time),
            'predictions_per_second': float(len(X) / prediction_time),
            'memory_usage': {
                'current': float(current),
                'peak': float(peak),
                'total': float(memory_usage)
            }
        }


class PerformanceBenchmark:
    """
    Benchmarks and compares performance across different models.

    Features:
    - Model comparison
    - Performance tracking
    - Resource usage analysis
    - Statistical significance testing
    """

    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.logger = get_logger(self.__class__.__name__)

    def benchmark_models(self,
                         models: List[TextClassifier],
                         X: np.ndarray,
                         y: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Benchmark multiple models and compare their performance."""
        benchmark_results = {
            'individual_results': {},
            'comparisons': {},
            'rankings': {}
        }

        # Evaluate each model
        for model in models:
            model_type = model.model_type
            self.logger.info(f"\nEvaluating {model_type}")

            evaluation = self.evaluator.evaluate_model(
                model, X, y, feature_names
            )

            benchmark_results['individual_results'][model_type] = evaluation

        # Compare models
        benchmark_results['comparisons'] = self._compare_models(
            benchmark_results['individual_results']
        )

        # Rank models
        benchmark_results['rankings'] = self._rank_models(
            benchmark_results['individual_results']
        )

        return benchmark_results

    def _compare_models(self,
                        individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance between models."""
        comparisons = {
            'accuracy': {},
            'timing': {},
            'resource_usage': {}
        }

        # Compare accuracy metrics
        for model_type, results in individual_results.items():
            cv_metrics = results['cross_validation_metrics']['mean']
            comparisons['accuracy'][model_type] = {
                'accuracy': cv_metrics['accuracy'],
                'precision': cv_metrics['precision'],
                'recall': cv_metrics['recall'],
                'f1': cv_metrics['f1']
            }

        # Compare timing and resource usage
        for model_type, results in individual_results.items():
            perf_metrics = results['performance_metrics']
            comparisons['timing'][model_type] = {
                'prediction_time': perf_metrics['prediction_time'],
                'predictions_per_second': perf_metrics['predictions_per_second']
            }
            comparisons['resource_usage'][model_type] = perf_metrics['memory_usage']

        return comparisons

    def _rank_models(self,
                     individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Rank models based on different metrics."""
        rankings = {
            'accuracy': [],
            'f1_score': [],
            'speed': [],
            'memory_efficiency': []
        }

        # Rank by accuracy
        model_accuracies = [
            (model_type, results['cross_validation_metrics']['mean']['accuracy'])
            for model_type, results in individual_results.items()
        ]
        rankings['accuracy'] = [
            model_type for model_type, _ in sorted(
                model_accuracies,
                key=lambda x: x[1],
                reverse=True
            )
        ]

        # Rank by F1 score
        model_f1_scores = [
            (model_type, results['cross_validation_metrics']['mean']['f1'])
            for model_type, results in individual_results.items()
        ]
        rankings['f1_score'] = [
            model_type for model_type, _ in sorted(
                model_f1_scores,
                key=lambda x: x[1],
                reverse=True
            )
        ]

        # Rank by speed (predictions per second)
        model_speeds = [
            (model_type, results['performance_metrics']['predictions_per_second'])
            for model_type, results in individual_results.items()
        ]
        rankings['speed'] = [
            model_type for model_type, _ in sorted(
                model_speeds,
                key=lambda x: x[1],
                reverse=True
            )
        ]

        # Rank by memory efficiency (inverse of peak memory usage)
        model_memory = [
            (model_type, -results['performance_metrics']['memory_usage']['peak'])
            for model_type, results in individual_results.items()
        ]
        rankings['memory_efficiency'] = [
            model_type for model_type, _ in sorted(
                model_memory,
                key=lambda x: x[1],
                reverse=True
            )
        ]

        return rankings

    def generate_summary_report(self,
                                benchmark_results: Dict[str, Any]) -> str:
        """Generate a detailed summary report of benchmark results."""
        summary = []

        # Add header
        summary.append("Model Benchmarking Summary Report")
        summary.append("=" * 30 + "\n")

        # Performance rankings
        summary.append("Performance Rankings")
        summary.append("-" * 20)
        for metric, ranked_models in benchmark_results['rankings'].items():
            summary.append(f"\n{metric.replace('_', ' ').title()}:")
            for i, model in enumerate(ranked_models, 1):
                summary.append(f"{i}. {model}")

        # Detailed metrics
        summary.append("\nDetailed Performance Metrics")
        summary.append("-" * 25)
        for model_type, results in benchmark_results['individual_results'].items():
            summary.append(f"\n{model_type.upper()}")
            cv_metrics = results['cross_validation_metrics']['mean']
            summary.append(f"Accuracy: {cv_metrics['accuracy']:.4f}")
            summary.append(f"F1 Score: {cv_metrics['f1']:.4f}")
            summary.append(f"Prediction Speed: {results['performance_metrics']['predictions_per_second']:.2f} pred/sec")
            summary.append(
                f"Peak Memory Usage: {results['performance_metrics']['memory_usage']['peak'] / 1024 / 1024:.2f} MB")

        # Return formatted summary
        return "\n".join(summary)

    def visualize_results(self,
                          benchmark_results: Dict[str, Any],
                          output_dir: Path) -> None:
        """Generate visualization plots for benchmark results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        accuracies = [
            results['cross_validation_metrics']['mean']['accuracy']
            for results in benchmark_results['individual_results'].values()
        ]
        model_types = list(benchmark_results['individual_results'].keys())

        plt.bar(model_types, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Model Type')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_comparison.png')
        plt.close()

        # Plot ROC curves
        plt.figure(figsize=(10, 6))
        for model_type, results in benchmark_results['individual_results'].items():
            roc_data = results['roc_analysis']['micro_average']
            plt.plot(
                roc_data['false_positive_rate'],
                roc_data['true_positive_rate'],
                label=f"{model_type} (AUC = {roc_data['auc']:.2f})"
            )

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_comparison.png')
        plt.close()

        # Plot resource usage
        plt.figure(figsize=(10, 6))
        memory_usage = [
            results['performance_metrics']['memory_usage']['peak'] / 1024 / 1024
            for results in benchmark_results['individual_results'].values()
        ]
        prediction_times = [
            results['performance_metrics']['prediction_time']
            for results in benchmark_results['individual_results'].values()
        ]

        plt.scatter(prediction_times, memory_usage)
        for i, model_type in enumerate(model_types):
            plt.annotate(
                model_type,
                (prediction_times[i], memory_usage[i]),
                xytext=(5, 5),
                textcoords='offset points'
            )

        plt.title('Resource Usage Comparison')
        plt.xlabel('Prediction Time (seconds)')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.tight_layout()
        plt.savefig(output_dir / 'resource_usage.png')
        plt.close()
