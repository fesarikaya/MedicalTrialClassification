import os
import shutil
import tempfile
import unittest
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.datasets import make_classification
from src.models.training_pipeline import (
    ExperimentTracker,
    ValidationManager,
    FeatureAnalyzer,
    TrainingPipeline
)
from src.models.model_factory import ModelFactory

class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for experiments
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tracker = ExperimentTracker(self.temp_dir)

    def tearDown(self):
        # Remove temporary directory after test
        shutil.rmtree(self.temp_dir)

    def test_log_and_save_load(self):
        step = 0
        train_metrics = {'accuracy': 0.9, 'loss': 0.1}
        val_metrics = {'accuracy': 0.85, 'loss': 0.15}
        feat_importance = {'feature_0': 0.5, 'feature_1': 0.3}

        self.tracker.log_metrics(train_metrics, step, metric_type='training_metrics')
        self.tracker.log_metrics(val_metrics, step, metric_type='validation_metrics')
        self.tracker.log_feature_importance(feat_importance, step)
        self.tracker.log_error_analysis({'misclassified_count': 5}, step)

        # Save experiment state
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracker.save_experiment_state(experiment_id)
        # Clear local metrics and then reload
        self.tracker.metrics_history = {}
        self.tracker.load_experiment_state(experiment_id)

        # Verify that keys are present
        for key in ['training_metrics', 'validation_metrics', 'feature_importance', 'error_analysis']:
            self.assertIn(key, self.tracker.metrics_history, f"Missing key: {key}")


class TestValidationManager(unittest.TestCase):
    def setUp(self):
        self.n_splits = 3
        self.val_manager = ValidationManager(n_splits=self.n_splits, random_state=42)
        # Create synthetic dataset
        X, y = make_classification(n_samples=50, n_features=10, n_informative=8, n_classes=3, random_state=42)
        self.X = X.astype(np.float32)
        self.y = y

    def test_get_cv_splits(self):
        splits = self.val_manager.get_cv_splits(self.X, self.y)
        self.assertEqual(len(splits), self.n_splits, "Number of CV splits does not match n_splits")

    def test_evaluate_fold(self):
        # Instead of using the pipeline.model (which is not created in __init__),
        # straightaway create a model instance via ModelFactory.
        model = ModelFactory.create_model(model_type='svm')
        splits = self.val_manager.get_cv_splits(self.X, self.y)
        train_idx, val_idx = splits[0]
        X_train, X_val = self.X[train_idx], self.X[val_idx]
        y_train, y_val = self.y[train_idx], self.y[val_idx]

        # Evaluate one fold using the non-None model
        train_metrics, val_metrics = self.val_manager.evaluate_fold(model, X_train, y_train, X_val, y_val)
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            self.assertIn(metric, train_metrics)
            self.assertIn(metric, val_metrics)


class TestFeatureAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = FeatureAnalyzer()
        # Create a dummy classifier object with feature importances.
        class DummyModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.1, 0.3, 0.6])
        # Create a dummy TextClassifier-like object.
        DummyClassifier = type("DummyClassifier", (), {})  # create a new type
        dummy = DummyClassifier()
        dummy.model = DummyModel()
        dummy.model_type = 'random_forest'
        # Set attributes needed by the analyzer if any.
        self.dummy_classifier = dummy

    def test_get_feature_importance(self):
        importance_scores = self.analyzer.get_feature_importance(self.dummy_classifier)
        self.assertIsInstance(importance_scores, dict)
        # Verify order: scores should be sorted in descending order.
        sorted_scores = sorted(importance_scores.values(), reverse=True)
        self.assertEqual(list(importance_scores.values()), sorted_scores)


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        X, y = make_classification(n_samples=200, n_features=10, n_informative=8, n_classes=3, random_state=42)
        self.X = X.astype(np.float32)
        self.y = y
        self.experiment_dir = Path(tempfile.mkdtemp())
        self.pipeline = TrainingPipeline(
            model_type='svm',
            model_params={'C': 1.5, 'kernel': 'linear'},
            experiment_dir=str(self.experiment_dir),
            n_splits=3,
            random_state=42
        )

    def tearDown(self):
        shutil.rmtree(self.experiment_dir)

    def test_train(self):
        feature_names = [f"feat_{i}" for i in range(self.X.shape[1])]
        results = self.pipeline.train(self.X, self.y, feature_names=feature_names)
        expected_keys = [
            'experiment_id', 'model_type', 'training_time',
            'final_metrics', 'feature_importance',
            'feature_correlations', 'cross_validation'
        ]
        for key in expected_keys:
            self.assertIn(key, results, f"Key '{key}' missing in final results")
        model_path = self.experiment_dir / f"{results['experiment_id']}_model.joblib"
        self.assertTrue(model_path.exists(), "Model file was not saved")

    def test_load_experiment(self):
        results = self.pipeline.train(self.X, self.y)
        exp_id = results['experiment_id']
        metrics_history = self.pipeline.load_experiment(exp_id)
        self.assertIsInstance(metrics_history, dict)
        self.assertTrue(any(key in metrics_history for key in ['training_metrics', 'validation_metrics']))

if __name__ == '__main__':
    unittest.main(verbosity=2)
