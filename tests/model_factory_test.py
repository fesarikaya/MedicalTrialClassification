import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from sklearn.datasets import make_classification
from src.models.model_factory import ModelFactory, TextClassifier


class TestModelFactory(unittest.TestCase):
    """Test suite for ModelFactory implementation"""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all test methods"""
        # Create synthetic dataset for testing
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=5,
            random_state=42
        )
        # Convert to float32 for better numerical stability
        cls.X = X.astype(np.float32)
        cls.y = y

        # Create temporary directory for model saving/loading tests
        cls.temp_dir = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are done"""
        # Remove temporary directory and its contents
        shutil.rmtree(cls.temp_dir)

    def test_supported_models(self):
        """Test getting supported model types"""
        supported_models = ModelFactory.get_supported_models()

        # Verify all required model types are supported
        expected_models = {
            'naive_bayes', 'logistic_regression', 'svm',
            'random_forest', 'bagging'
        }
        self.assertEqual(
            set(supported_models.keys()),
            expected_models,
            "Not all required model types are supported"
        )

        # Verify each model has required configuration
        for model_info in supported_models.values():
            self.assertIn('description', model_info)
            self.assertIn('default_params', model_info)

    def test_model_creation(self):
        """Test creating models of each type"""
        for model_type in ModelFactory.get_supported_models():
            # Create model with default configuration
            model = ModelFactory.create_model(model_type)

            # Verify model properties
            self.assertIsInstance(model, TextClassifier)
            self.assertEqual(model.model_type, model_type)

            # Verify model can be trained
            model.fit(self.X, self.y)

            # Verify predictions
            predictions = model.predict(self.X)
            self.assertEqual(predictions.shape, (len(self.X),))

            # Verify unique classes
            unique_classes = np.unique(predictions)
            self.assertEqual(
                len(unique_classes),
                len(np.unique(self.y)),
                f"Incorrect number of classes for {model_type}"
            )

    def test_custom_configuration(self):
        """Test model creation with custom configurations"""
        test_configs = {
            'naive_bayes': {'alpha': 0.5},
            'logistic_regression': {'C': 0.1, 'max_iter': 500},
            'svm': {'C': 2.0, 'kernel': 'linear'},
            'random_forest': {'n_estimators': 50, 'max_depth': 10},
            'bagging': {'n_estimators': 20, 'max_samples': 0.8}
        }

        for model_type, config in test_configs.items():
            # Create model with custom config
            model = ModelFactory.create_model(
                model_type=model_type,
                model_config=config
            )

            # Verify configuration was applied
            model_params = model.get_params()
            for param, value in config.items():
                self.assertIn(param, model_params['model_params'])
                self.assertEqual(
                    model_params['model_params'][param],
                    value,
                    f"Parameter {param} not set correctly for {model_type}"
                )

    def test_model_validation(self):
        """Test model configuration validation"""
        # Test valid configurations
        valid_config = {
            'naive_bayes': {'alpha': 1.0},
            'logistic_regression': {'C': 1.0, 'max_iter': 1000},
            'svm': {'C': 1.0, 'kernel': 'linear'},
            'random_forest': {'n_estimators': 100},
            'bagging': {'n_estimators': 100}
        }

        for model_type, config in valid_config.items():
            self.assertTrue(
                ModelFactory.validate_model_config(model_type, config),
                f"Valid configuration rejected for {model_type}"
            )

        # Test invalid configurations
        with self.assertRaises(ValueError):
            ModelFactory.validate_model_config(
                'naive_bayes',
                {'invalid_param': 1.0}
            )

        with self.assertRaises(ValueError):
            ModelFactory.validate_model_config(
                'invalid_model_type',
                {'param': 1.0}
            )

    def test_model_persistence(self):
        """Test model saving and loading"""
        for model_type in ModelFactory.get_supported_models():
            # Create and train model
            model = ModelFactory.create_model(model_type)
            model.fit(self.X, self.y)

            # Generate predictions before saving
            predictions_before = model.predict(self.X)

            # Save model
            save_path = self.temp_dir / f"{model_type}_model.joblib"
            model.save(save_path)

            # Load model
            loaded_model = ModelFactory.load_model(save_path)

            # Verify predictions match
            predictions_after = loaded_model.predict(self.X)
            np.testing.assert_array_equal(
                predictions_before,
                predictions_after,
                f"Predictions changed after save/load for {model_type}"
            )

    def test_ensemble_creation(self):
        """Test creating model ensembles"""
        # Test creating ensemble with default configurations
        model_types = ['naive_bayes', 'svm', 'random_forest']
        ensemble = ModelFactory.create_ensemble(model_types)

        # Verify ensemble properties
        self.assertEqual(len(ensemble), len(model_types))
        for model, expected_type in zip(ensemble, model_types):
            self.assertEqual(model.model_type, expected_type)

        # Test with custom configurations
        configs = [
            {'alpha': 0.5},
            {'C': 2.0, 'kernel': 'linear'},
            {'n_estimators': 50}
        ]
        ensemble = ModelFactory.create_ensemble(model_types, configs)

        # Verify configurations were applied
        for model, config in zip(ensemble, configs):
            model_params = model.get_params()
            for param, value in config.items():
                self.assertEqual(
                    model_params['model_params'][param],
                    value
                )

    def test_error_handling(self):
        """Test error handling in model factory"""
        # Test invalid model type
        with self.assertRaises(ValueError):
            ModelFactory.create_model('invalid_model_type')

        # Test invalid configuration
        with self.assertRaises(ValueError):
            ModelFactory.create_model(
                'naive_bayes',
                {'invalid_param': 1.0}
            )

        # Test invalid ensemble configuration
        with self.assertRaises(ValueError):
            ModelFactory.create_ensemble(
                ['naive_bayes', 'svm'],
                [{'alpha': 0.5}]  # Mismatched number of configurations
            )

        # Test loading non-existent model
        with self.assertRaises(FileNotFoundError):
            ModelFactory.load_model('non_existent_model.joblib')

    def test_model_evaluation(self):
        """Test model evaluation functionality"""
        # Split data into train and test sets
        train_size = int(0.8 * len(self.X))
        X_train = self.X[:train_size]
        y_train = self.y[:train_size]
        X_test = self.X[train_size:]
        y_test = self.y[train_size:]

        for model_type in ModelFactory.get_supported_models():
            # Create and train model
            model = ModelFactory.create_model(model_type)
            model.fit(X_train, y_train)

            # Evaluate model
            evaluation = model.evaluate(X_test, y_test)

            # Verify evaluation metrics
            self.assertIn('overall_metrics', evaluation)
            self.assertIn('per_class_metrics', evaluation)
            self.assertIn('confusion_matrix', evaluation)
            self.assertIn('error_analysis', evaluation)

            # Verify metric values are valid
            self.assertGreaterEqual(evaluation['overall_metrics']['accuracy'], 0.0)
            self.assertLessEqual(evaluation['overall_metrics']['accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)