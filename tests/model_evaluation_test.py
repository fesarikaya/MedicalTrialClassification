import os
import shutil
import tempfile
import unittest
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from src.models.model_evaluation import ModelEvaluator, PerformanceBenchmark
from src.models.model_factory import ModelFactory, TextClassifier


class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        # Create synthetic data with three classes.
        X, y = make_classification(
            n_samples=100, n_features=20, n_informative=10, n_classes=3,
            random_state=42
        )
        self.X = X.astype(np.float32)
        self.y = y
        self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]
        # Create and fit a classifier model using ModelFactory.
        self.model = ModelFactory.create_model(model_type='logistic_regression')
        self.model.fit(self.X, self.y)
        # Manually ensure that classes_ is assigned if not set during fit.
        if self.model.classes_ is None:
            self.model.classes_ = np.unique(self.y)
        # Instantiate the evaluator with 3-fold CV.
        self.evaluator = ModelEvaluator(n_splits=3, random_state=42)

    def test_perform_cross_validation(self):
        cv_metrics = self.evaluator._perform_cross_validation(self.model, self.X, self.y)
        self.assertIn('mean', cv_metrics)
        self.assertIn('std', cv_metrics)
        self.assertIn('confidence_intervals', cv_metrics)
        self.assertIsInstance(cv_metrics['mean']['accuracy'], float)

    def test_analyze_confusion_matrix(self):
        cm_analysis = self.evaluator._analyze_confusion_matrix(self.model, self.X, self.y)
        self.assertIn('confusion_matrix', cm_analysis)
        self.assertIn('overall_accuracy', cm_analysis)
        self.assertIsInstance(cm_analysis['confusion_matrix'], list)
        self.assertIsInstance(cm_analysis['overall_accuracy'], float)

    def test_perform_roc_analysis(self):
        # For logistic regression, predict_proba is available.
        roc_analysis = self.evaluator._perform_roc_analysis(self.model, self.X, self.y)
        # If the model supports predict_proba, roc_analysis should have these keys.
        self.assertIn('per_class_roc', roc_analysis)
        self.assertIn('micro_average', roc_analysis)
        self.assertIsInstance(roc_analysis['micro_average']['auc'], float)

    def test_perform_error_analysis(self):
        error_analysis = self.evaluator._perform_error_analysis(
            self.model, self.X, self.y, self.feature_names
        )
        self.assertIn('total_errors', error_analysis)
        self.assertIn('error_cases', error_analysis)
        # Even if the model is well-trained errors may be zero.
        self.assertIsInstance(error_analysis['total_errors'], int)

    def test_calculate_performance_metrics(self):
        perf_metrics = self.evaluator._calculate_performance_metrics(self.model, self.X, self.y)
        self.assertIn('prediction_time', perf_metrics)
        self.assertIn('predictions_per_second', perf_metrics)
        self.assertIn('memory_usage', perf_metrics)

    def test_evaluate_model(self):
        # Evaluate the model using the public interface.
        results = self.evaluator.evaluate_model(self.model, self.X, self.y, self.feature_names)
        expected_keys = [
            'cross_validation_metrics',
            'confusion_matrix_analysis',
            'roc_analysis',
            'error_analysis',
            'performance_metrics'
        ]
        for key in expected_keys:
            self.assertIn(key, results, f"Expected key '{key}' is missing in evaluation results")


class TestPerformanceBenchmark(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data.
        X, y = make_classification(
            n_samples=100, n_features=10, n_informative=5, n_classes=3,
            random_state=123
        )
        self.X = X.astype(np.float32)
        self.y = y
        self.feature_names = [f"feat_{i}" for i in range(self.X.shape[1])]
        # Create two models with different types.
        self.model1 = ModelFactory.create_model('logistic_regression')
        self.model1.fit(self.X, self.y)
        if self.model1.classes_ is None:
            self.model1.classes_ = np.unique(self.y)
        self.model2 = ModelFactory.create_model('svm')
        self.model2.fit(self.X, self.y)
        if self.model2.classes_ is None:
            self.model2.classes_ = np.unique(self.y)
        self.models = [self.model1, self.model2]
        # Instantiate the PerformanceBenchmark.
        self.benchmark = PerformanceBenchmark()

    def test_benchmark_models(self):
        benchmark_results = self.benchmark.benchmark_models(
            self.models, self.X, self.y, self.feature_names
        )
        self.assertIn('individual_results', benchmark_results)
        self.assertIn('comparisons', benchmark_results)
        self.assertIn('rankings', benchmark_results)
        for model_type, results in benchmark_results['individual_results'].items():
            self.assertIsInstance(results, dict)

    def test_generate_summary_report(self):
        benchmark_results = self.benchmark.benchmark_models(
            self.models, self.X, self.y, self.feature_names
        )
        summary = self.benchmark.generate_summary_report(benchmark_results)
        self.assertIsInstance(summary, str)
        self.assertIn("Model Benchmarking Summary Report", summary)

    def test_visualize_results(self):
        # Create a temporary directory for saving plots.
        output_dir = Path(tempfile.mkdtemp())
        benchmark_results = self.benchmark.benchmark_models(
            self.models, self.X, self.y, self.feature_names
        )
        self.benchmark.visualize_results(benchmark_results, output_dir)
        self.assertTrue((output_dir / 'accuracy_comparison.png').exists())
        self.assertTrue((output_dir / 'roc_comparison.png').exists())
        self.assertTrue((output_dir / 'resource_usage.png').exists())
        shutil.rmtree(output_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
