import numpy as np
import joblib
import time
from typing import Dict, Optional, Tuple, Any, List
from abc import ABC, abstractmethod
from pathlib import Path
from src.utils.logger import get_logger
from joblib import dump, load
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import KFold

logger = get_logger(__name__)



class BaseModel(ABC):
    """
    Abstract base class for text classification models.

    This class defines the interface that all model implementations must follow.
    It includes basic methods for training, prediction, and model persistence,
    as well as evaluation methods specific to text classification.

    Attributes:
        model: The underlying model instance
        classes_: List of unique class labels
        logger: Logger instance for tracking model operations
    """

    def __init__(self):
        """Initialize base model attributes"""
        self.model = None
        self.classes_ = None
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> \
    Dict[str, Any]:
        """
        Train the model on input data with comprehensive performance evaluation.

        The method supports multiple classification algorithms:
        - Naive Bayes
        - Logistic Regression
        - Support Vector Machines (SVM)
        - Random Forests
        - Bagging (Bootstrap Aggregating)

        Performance metrics tracked:
        - Accuracy
        - Precision (per class and weighted average)
        - Recall (per class and weighted average)
        - F1 Score (per class and weighted average)
        - Confusion Matrix

        Args:
            X: Feature matrix of shape (n_samples, n_features)
                Training features from preprocessed text data
            y: Target labels of shape (n_samples,)
                Disease classification labels
            validation_data: Optional tuple of (X_val, y_val) for validation
                X_val: Validation features of shape (n_val_samples, n_features)
                y_val: Validation labels of shape (n_val_samples,)

        Returns:
            Dict containing training metrics and performance evaluation:
            {
                'training_metrics': {
                    'accuracy': float,
                    'precision': Dict[str, float],  # Per class
                    'recall': Dict[str, float],     # Per class
                    'f1_score': Dict[str, float],   # Per class
                    'confusion_matrix': np.ndarray,
                    'avg_metrics': {
                        'precision': float,  # Weighted average
                        'recall': float,     # Weighted average
                        'f1_score': float    # Weighted average
                    }
                },
                'validation_metrics': {
                    # Same structure as training_metrics
                } if validation_data else None,
                'model_info': {
                    'model_type': str,
                    'training_time': float,
                    'n_features': int,
                    'n_classes': int,
                    'class_distribution': Dict[str, int]
                }
            }

        Raises:
            ValueError: If input data format is invalid
            RuntimeError: If training fails
        """

        try:
            # Log training start
            self.logger.info("Starting model training...")
            self.logger.info(f"Training data shape: X={X.shape}, y={y.shape}")

            # Validate input data
            self._validate_training_data(X, y)

            # Store unique classes
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            self.logger.info(f"Number of classes: {n_classes}")

            # Calculate class distribution
            class_distribution = {
                str(label): int(np.sum(y == label))
                for label in self.classes_
            }

            # Initialize performance metrics dictionary
            performance_metrics = {
                'training_metrics': {},
                'validation_metrics': None,
                'model_info': {
                    'model_type': self.model_type if hasattr(self, 'model_type') else 'base',
                    'training_time': 0,
                    'n_features': X.shape[1],
                    'n_classes': n_classes,
                    'class_distribution': class_distribution
                }
            }

            # Record training start time
            start_time = time.time()

            # Train model using _train_model method
            training_history = self._train_model(X, y, validation_data)

            # Make predictions on training data
            y_pred_train = self.model.predict(X)

            # Calculate comprehensive training metrics
            performance_metrics['training_metrics'] = {
                'accuracy': accuracy_score(y, y_pred_train),
                'precision': {
                    str(label): precision_score(y, y_pred_train, labels=[label], average='weighted')
                    for label in self.classes_
                },
                'recall': {
                    str(label): recall_score(y, y_pred_train, labels=[label], average='weighted')
                    for label in self.classes_
                },
                'f1_score': {
                    str(label): f1_score(y, y_pred_train, labels=[label], average='weighted')
                    for label in self.classes_
                },
                'confusion_matrix': confusion_matrix(y, y_pred_train).tolist(),
                'avg_metrics': {
                    'precision': precision_score(y, y_pred_train, average='weighted'),
                    'recall': recall_score(y, y_pred_train, average='weighted'),
                    'f1_score': f1_score(y, y_pred_train, average='weighted')
                }
            }

            # Calculate validation metrics if validation data provided
            if validation_data is not None:
                X_val, y_val = validation_data

                # Validate validation data
                self._validate_training_data(X_val, y_val, is_validation=True)

                # Make predictions on validation data
                y_pred_val = self.model.predict(X_val)

                # Calculate comprehensive validation metrics
                performance_metrics['validation_metrics'] = {
                    'accuracy': accuracy_score(y_val, y_pred_val),
                    'precision': {
                        str(label): precision_score(y_val, y_pred_val, labels=[label], average='weighted')
                        for label in self.classes_
                    },
                    'recall': {
                        str(label): recall_score(y_val, y_pred_val, labels=[label], average='weighted')
                        for label in self.classes_
                    },
                    'f1_score': {
                        str(label): f1_score(y_val, y_pred_val, labels=[label], average='weighted')
                        for label in self.classes_
                    },
                    'confusion_matrix': confusion_matrix(y_val, y_pred_val).tolist(),
                    'avg_metrics': {
                        'precision': precision_score(y_val, y_pred_val, average='weighted'),
                        'recall': recall_score(y_val, y_pred_val, average='weighted'),
                        'f1_score': f1_score(y_val, y_pred_val, average='weighted')
                    }
                }

            # Record training end time
            performance_metrics['model_info']['training_time'] = time.time() - start_time

            # Log training completion and metrics
            self.logger.info("\nTraining completed successfully")
            self.logger.info("\nTraining Metrics:")
            self._log_metrics(performance_metrics['training_metrics'])

            if validation_data is not None:
                self.logger.info("\nValidation Metrics:")
                self._log_metrics(performance_metrics['validation_metrics'])

            return performance_metrics

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Helper method to log performance metrics."""
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")

        self.logger.info("\nPer-class Metrics:")
        for label in metrics['precision'].keys():
            self.logger.info(f"\nClass {label}:")
            self.logger.info(f"Precision: {metrics['precision'][label]:.4f}")
            self.logger.info(f"Recall: {metrics['recall'][label]:.4f}")
            self.logger.info(f"F1-score: {metrics['f1_score'][label]:.4f}")

        self.logger.info("\nWeighted Averages:")
        self.logger.info(f"Precision: {metrics['avg_metrics']['precision']:.4f}")
        self.logger.info(f"Recall: {metrics['avg_metrics']['recall']:.4f}")
        self.logger.info(f"F1-score: {metrics['avg_metrics']['f1_score']:.4f}")

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray, is_validation: bool = False) -> None:
        """
        Validate training data format and contents.

        Args:
            X: Feature matrix
            y: Target labels
            is_validation: Whether this is validation data

        Raises:
            ValueError: If data validation fails
        """
        data_type = "Validation" if is_validation else "Training"

        # Check data types
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError(f"{data_type} data must be numpy arrays")

        # Check dimensions
        if X.ndim != 2:
            raise ValueError(f"{data_type} features must be 2-dimensional")
        if y.ndim != 1:
            raise ValueError(f"{data_type} labels must be 1-dimensional")

        # Check matching lengths
        if len(X) != len(y):
            raise ValueError(f"{data_type} features and labels must have same length")

        # Check for NaN or infinite values
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError(f"{data_type} features contain NaN or infinite values")

        # Check label validity
        if not is_validation and len(np.unique(y)) < 2:
            raise ValueError("Training data must contain at least 2 classes")

    @abstractmethod
    def _train_model(self, X: np.ndarray, y: np.ndarray,
                     validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Model-specific training implementation with comprehensive performance metrics.

        This implementation supports:
        - Naive Bayes
        - Logistic Regression
        - Support Vector Machines (SVM)
        - Random Forests
        - Bagging (Bootstrap Aggregating)

        Performance metrics tracked:
        - Accuracy
        - Precision (per class and weighted average)
        - Recall (per class and weighted average)
        - F1 Score (per class and weighted average)
        - Confusion Matrix

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            validation_data: Optional tuple of (X_val, y_val) for validation

        Returns:
            Dict containing training history and detailed performance metrics
        """

        # Initialize training history with extended metrics
        training_history = {
            'metrics': {
                'train': {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'confusion_matrix': []
                },
                'validation': {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'confusion_matrix': []
                }
            },
            'training_time': 0,
            'best_iteration': 0,
            'model_params': {},
            'convergence_info': {}
        }

        start_time = time.time()

        try:
            # Initialize model based on type
            if hasattr(self, 'model_type'):
                if self.model_type == 'naive_bayes':
                    self.model = MultinomialNB()
                elif self.model_type == 'logistic_regression':
                    self.model = LogisticRegression(
                        multi_class='multinomial',
                        max_iter=1000,
                        solver='lbfgs'
                    )
                elif self.model_type == 'svm':
                    self.model = SVC(
                        kernel='linear',
                        probability=True
                    )
                elif self.model_type == 'random_forest':
                    self.model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        random_state=42
                    )
                elif self.model_type == 'bagging':
                    base_estimator = DecisionTreeClassifier(max_depth=None)
                    self.model = BaggingClassifier(
                        base_estimator=base_estimator,
                        n_estimators=100,
                        random_state=42
                    )
                else:
                    raise ValueError(f"Unsupported model type: {self.model_type}")

            def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
                """Calculate all performance metrics for a set of predictions."""
                return {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted'),
                    'recall': recall_score(y_true, y_pred, average='weighted'),
                    'f1': f1_score(y_true, y_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                    'per_class_metrics': classification_report(y_true, y_pred, output_dict=True)
                }

            # Implement k-fold cross-validation
            n_splits = 5
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_metrics = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Train model on fold
                self.model.fit(X_train_fold, y_train_fold)

                # Calculate predictions
                train_pred = self.model.predict(X_train_fold)
                val_pred = self.model.predict(X_val_fold)

                # Calculate metrics
                train_metrics = calculate_metrics(y_train_fold, train_pred)
                val_metrics = calculate_metrics(y_val_fold, val_pred)

                # Store metrics
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']:
                    training_history['metrics']['train'][metric].append(train_metrics[metric])
                    training_history['metrics']['validation'][metric].append(val_metrics[metric])

                fold_metrics.append(val_metrics)

                # Log fold metrics
                self.logger.info(f"\nFold {fold + 1}/{n_splits} Results:")
                self.logger.info(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
                self.logger.info(f"Train Precision: {train_metrics['precision']:.4f}")
                self.logger.info(f"Train Recall: {train_metrics['recall']:.4f}")
                self.logger.info(f"Train F1: {train_metrics['f1']:.4f}")
                self.logger.info(f"\nValidation Accuracy: {val_metrics['accuracy']:.4f}")
                self.logger.info(f"Validation Precision: {val_metrics['precision']:.4f}")
                self.logger.info(f"Validation Recall: {val_metrics['recall']:.4f}")
                self.logger.info(f"Validation F1: {val_metrics['f1']:.4f}")

            # Final training on full dataset
            self.model.fit(X, y)

            # If validation data provided, calculate final validation metrics
            if validation_data is not None:
                X_val, y_val = validation_data
                val_pred = self.model.predict(X_val)
                final_val_metrics = calculate_metrics(y_val, val_pred)

                training_history['final_validation_metrics'] = final_val_metrics

                self.logger.info("\nFinal Validation Metrics:")
                self.logger.info(f"Accuracy: {final_val_metrics['accuracy']:.4f}")
                self.logger.info(f"Precision: {final_val_metrics['precision']:.4f}")
                self.logger.info(f"Recall: {final_val_metrics['recall']:.4f}")
                self.logger.info(f"F1 Score: {final_val_metrics['f1']:.4f}")

            # Calculate average metrics across folds
            training_history['cross_validation'] = {
                'mean_metrics': {},
                'std_metrics': {}
            }

            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                values = [m[metric] for m in fold_metrics]
                training_history['cross_validation']['mean_metrics'][metric] = float(np.mean(values))
                training_history['cross_validation']['std_metrics'][metric] = float(np.std(values))

            # Record training time
            training_history['training_time'] = time.time() - start_time

            # Record model parameters
            training_history['model_params'] = self.model.get_params()

            # Determine best iteration based on validation F1 score
            training_history['best_iteration'] = np.argmax(
                training_history['metrics']['validation']['f1']
            )

            # Log final summary
            self.logger.info("\nTraining Summary:")
            self.logger.info("Cross-validation metrics (mean ± std):")
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                mean = training_history['cross_validation']['mean_metrics'][metric]
                std = training_history['cross_validation']['std_metrics'][metric]
                self.logger.info(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")

            self.logger.info(f"\nTraining time: {training_history['training_time']:.2f} seconds")

            return training_history

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Model training failed: {str(e)}")

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on medical text data using trained classification model.

        This method handles predictions for different classifier types (Naive Bayes,
        Logistic Regression, SVM, Random Forests, and Bagging) while ensuring proper
        input validation and error handling. It processes feature vectors from medical
        text descriptions to predict disease classifications.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
               Each row represents a medical text sample's feature vector

        Returns:
            np.ndarray: Predicted disease labels of shape (n_samples,)
                       Contains predicted class labels matching training labels

        Raises:
            ValueError: If input features have invalid shape or type
            RuntimeError: If model is not trained or prediction fails
        """
        try:
            # Validate model is trained
            if self.model is None:
                raise RuntimeError("Model must be trained before making predictions")

            # Validate input data
            if not isinstance(X, np.ndarray):
                raise ValueError("Input must be a numpy array")

            if X.ndim != 2:
                raise ValueError(f"Input must be 2D array, got shape {X.shape}")

            # Handle feature dimensionality
            n_features_expected = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else X.shape[1]
            if X.shape[1] != n_features_expected:
                raise ValueError(f"Expected {n_features_expected} features, got {X.shape[1]}")

            # Make predictions
            predictions = self.model.predict(X)

            # Validate predictions
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)

            if predictions.shape[0] != X.shape[0]:
                raise RuntimeError(f"Expected {X.shape[0]} predictions, got {predictions.shape[0]}")

            # Ensure predictions match training classes
            if not np.all(np.isin(predictions, self.classes_)):
                raise RuntimeError("Predictions contain invalid class labels")

            return predictions

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Error during prediction: {str(e)}")

    def predict_single(self, x: np.ndarray) -> str:
        """
        Make prediction for a single medical text input through the Flask API endpoint.

        This method handles single-sample prediction requests for the deployed model,
        supporting all classifier types (Naive Bayes, Logistic Regression, SVM, Random
        Forests, and Bagging). It ensures proper input formatting and validation before
        passing to the underlying model.

        Args:
            x: Single feature vector of shape (n_features,) representing the processed
               text features from a single medical text description

        Returns:
            str: Predicted disease class label

        Raises:
            ValueError: If input features have invalid shape or type
            RuntimeError: If model is not trained or prediction fails
        """
        try:
            # Validate input type
            if not isinstance(x, np.ndarray):
                raise ValueError("Input must be a numpy array")

            # Validate model is trained
            if self.model is None:
                raise RuntimeError("Model must be trained before making predictions")

            # Validate input dimensions
            n_features_expected = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else x.shape[-1]
            if x.size != n_features_expected:
                raise ValueError(f"Expected {n_features_expected} features, got {x.size}")

            # Reshape input to 2D if needed
            if x.ndim == 1:
                x = x.reshape(1, -1)
            elif x.ndim != 2:
                raise ValueError(f"Input must be 1D or 2D array, got {x.ndim} dimensions")

            # Get prediction
            prediction = self.predict(x)[0]

            # Validate prediction is valid class label
            if prediction not in self.classes_:
                raise RuntimeError("Invalid prediction class")

            return prediction

        except Exception as e:
            self.logger.error(f"Single prediction failed: {str(e)}")
            raise RuntimeError(f"Error during single prediction: {str(e)}")

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters and configuration information for medical text classifier.

        The method returns essential parameters needed for the disease classification model,
        including model settings, current state, and classification details. This is crucial
        for model persistence and serving through the Flask API.

        Returns:
            Dict containing model parameters and state information

        Raises:
            RuntimeError: If model is not properly initialized
            ValueError: If required parameters are missing
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        try:

            return {
                'model_info': {
                    'model_type': self.__class__.__name__,
                    'n_classes': len(self.classes_) if self.classes_ is not None else 0,
                    'input_dim': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None,
                    'classes': self.classes_.tolist() if self.classes_ is not None else None,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'model_params': self.model.get_params() if hasattr(self.model, 'get_params') else {},
                'preprocessing_params': {
                    'max_length': getattr(self, 'max_length', 5000),
                    'vocabulary_size': getattr(self, 'vocabulary_size', None)
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get model parameters: {str(e)}")
            raise RuntimeError(f"Error retrieving model parameters: {str(e)}")

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk using joblib.

        This method saves the trained model along with essential metadata required
        for model restoration and validation. The saved file includes the model state,
        class labels, and model parameters.

        Args:
            path: Path where the model should be saved
                 Example: 'models/medical_classifier.joblib'

        Raises:
            ValueError: If the model hasn't been trained or path is invalid
            RuntimeError: If saving fails due to I/O or serialization errors
        """
        try:
            if self.model is None:
                raise ValueError("Cannot save unintialized model. Model must be trained first.")

            # Create directory if it doesn't exist
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Prepare model data for saving
            model_data = {
                'model': self.model,
                'classes': self.classes_,
                'parameters': self.get_params(),
                'model_type': self.__class__.__name__
            }

            # Save the model data
            joblib.dump(model_data, path)
            self.logger.info(f"Model successfully saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving model to {path}: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from disk using joblib.

        This method loads a previously saved model and validates its compatibility
        with the current classification task. It restores the model state, class
        labels, and model parameters.

        Args:
            path: Path from which to load the model
                 Example: 'models/medical_classifier.joblib'

        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the loaded model is incompatible or corrupted
            RuntimeError: If loading fails due to I/O or deserialization errors
        """
        try:
            # Check if file exists
            if not Path(path).is_file():
                raise FileNotFoundError(f"Model file not found at {path}")

            # Load model data
            model_data = joblib.load(path)

            # Validate model data
            required_keys = {'model', 'classes', 'parameters', 'model_type'}
            if not all(key in model_data for key in required_keys):
                raise ValueError("Loaded model data is missing required components")

            # Validate model type
            if model_data['model_type'] != self.__class__.__name__:
                raise ValueError(f"Model type mismatch. Expected {self.__class__.__name__}, "
                                 f"got {model_data['model_type']}")

            # Restore model state
            self.model = model_data['model']
            self.classes_ = model_data['classes']

            self.logger.info(f"Model successfully loaded from {path}")
            self.logger.info(f"Model type: {model_data['model_type']}")
            self.logger.info(f"Number of classes: {len(self.classes_)}")

        except Exception as e:
            self.logger.error(f"Error loading model from {path}: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance with comprehensive metrics calculation.

        This method provides thorough performance evaluation for all supported classification algorithms:
        - Naive Bayes
        - Logistic Regression
        - Support Vector Machines (SVM)
        - Random Forests
        - Bagging (Bootstrap Aggregating)

        The evaluation includes the following metrics:
        - Accuracy: Overall classification accuracy
        - Precision: Both per-class and weighted average
        - Recall: Both per-class and weighted average
        - F1 Score: Both per-class and weighted average
        - Confusion Matrix: Detailed error analysis

        Args:
            X: Feature matrix of shape (n_samples, n_features)
               Test features from preprocessed medical text data
            y: True labels of shape (n_samples,)
               Actual disease classification labels

        Returns:
            Dict containing comprehensive evaluation metrics:
            {
                'overall_metrics': {
                    'accuracy': float,
                    'weighted_precision': float,
                    'weighted_recall': float,
                    'weighted_f1': float
                },
                'per_class_metrics': {
                    class_name: {
                        'precision': float,
                        'recall': float,
                        'f1_score': float,
                        'support': int
                    }
                },
                'confusion_matrix': {
                    'matrix': np.ndarray,
                    'display_labels': List[str]
                },
                'classification_report': str,
                'evaluation_time': float,
                'error_analysis': {
                    'misclassified_count': int,
                    'most_confused_pairs': List[Tuple[str, str, int]]
                }
            }

        Raises:
            ValueError: If input data format is invalid
            RuntimeError: If evaluation fails
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, classification_report
        )
        import time

        try:
            # Validate model is trained
            if self.model is None:
                raise RuntimeError("Model must be trained before evaluation")

            # Validate input data
            self.validate_input(X)
            if not isinstance(y, np.ndarray):
                raise ValueError("Labels must be a numpy array")

            if len(X) != len(y):
                raise ValueError("Features and labels must have the same length")

            evaluation_start = time.time()

            # Make predictions
            y_pred = self.predict(X)

            # Initialize evaluation results dictionary
            evaluation_results = {
                'overall_metrics': {},
                'per_class_metrics': {},
                'confusion_matrix': {},
                'error_analysis': {}
            }

            # Calculate overall metrics with weighted averaging
            evaluation_results['overall_metrics'] = {
                'accuracy': accuracy_score(y, y_pred),
                'weighted_precision': precision_score(y, y_pred, average='weighted'),
                'weighted_recall': recall_score(y, y_pred, average='weighted'),
                'weighted_f1': f1_score(y, y_pred, average='weighted')
            }

            # Calculate per-class metrics
            for class_label in self.classes_:
                class_metrics = {
                    'precision': precision_score(y, y_pred, labels=[class_label], average='weighted'),
                    'recall': recall_score(y, y_pred, labels=[class_label], average='weighted'),
                    'f1_score': f1_score(y, y_pred, labels=[class_label], average='weighted'),
                    'support': np.sum(y == class_label)
                }
                evaluation_results['per_class_metrics'][str(class_label)] = class_metrics

            # Calculate and store confusion matrix
            conf_matrix = confusion_matrix(y, y_pred)
            evaluation_results['confusion_matrix'] = {
                'matrix': conf_matrix.tolist(),
                'display_labels': [str(label) for label in self.classes_]
            }

            # Generate classification report
            evaluation_results['classification_report'] = classification_report(y, y_pred)

            # Perform error analysis
            misclassified = y != y_pred
            evaluation_results['error_analysis'] = {
                'misclassified_count': np.sum(misclassified),
                'misclassification_rate': float(np.sum(misclassified)) / len(y)
            }

            # Find most confused class pairs
            confused_pairs = []
            for i in range(len(self.classes_)):
                for j in range(i + 1, len(self.classes_)):
                    confused_count = conf_matrix[i, j] + conf_matrix[j, i]
                    if confused_count > 0:
                        confused_pairs.append((
                            str(self.classes_[i]),
                            str(self.classes_[j]),
                            int(confused_count)
                        ))

            # Sort pairs by confusion count and take top 3
            confused_pairs.sort(key=lambda x: x[2], reverse=True)
            evaluation_results['error_analysis']['most_confused_pairs'] = confused_pairs[:3]

            # Calculate evaluation time
            evaluation_results['evaluation_time'] = time.time() - evaluation_start

            # Log evaluation results
            self.logger.info("\nModel Evaluation Results:")

            self.logger.info("\nOverall Metrics:")
            for metric, value in evaluation_results['overall_metrics'].items():
                self.logger.info(f"{metric}: {value:.4f}")

            self.logger.info("\nPer-class Metrics:")
            for class_label, metrics in evaluation_results['per_class_metrics'].items():
                self.logger.info(f"\nClass: {class_label}")
                for metric, value in metrics.items():
                    self.logger.info(f"{metric}: {value:.4f}")

            self.logger.info("\nError Analysis:")
            self.logger.info(f"Total misclassified: {evaluation_results['error_analysis']['misclassified_count']}")
            self.logger.info(
                f"Misclassification rate: {evaluation_results['error_analysis']['misclassification_rate']:.4f}")

            self.logger.info("\nMost Confused Class Pairs:")
            for class1, class2, count in evaluation_results['error_analysis']['most_confused_pairs']:
                self.logger.info(f"{class1} - {class2}: {count} instances")

            self.logger.info(f"\nEvaluation completed in {evaluation_results['evaluation_time']:.2f} seconds")

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise RuntimeError(f"Model evaluation failed: {str(e)}")

    def validate_input(self, X: np.ndarray) -> bool:
        """
        Validate input data format.

        Args:
            X: Input features to validate

        Returns:
            bool: True if input is valid

        Raises:
            ValueError: If input format is invalid
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if X.ndim not in [1, 2]:
            raise ValueError("Input must be 1D or 2D array")

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dict containing model information
        """
        return {
            'model_type': self.__class__.__name__,
            'num_classes': len(self.classes_) if self.classes_ is not None else None,
            'classes': self.classes_,
            'parameters': self.get_params()
        }


class TextClassifier(BaseModel):
    """
    Text classifier implementing multiple classification algorithms for medical text classification.

    This classifier supports five classification methods:
    - Naive Bayes
    - Logistic Regression
    - Support Vector Machines (SVM)
    - Random Forests
    - Bagging (Bootstrap Aggregating)

    The classifier inherits from BaseModel and implements all necessary methods for training,
    prediction, and evaluation while maintaining comprehensive performance tracking.
    """

    def __init__(self,
                 model_type: str = 'svm',
                 model_params: Optional[Dict[str, Any]] = None):
        """Initialize the text classifier."""
        super().__init__()
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.logger = get_logger(self.__class__.__name__)
        self._nb_shift = 0
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the specified classification model with appropriate parameters."""
        try:
            if self.model_type == 'naive_bayes':
                from sklearn.naive_bayes import MultinomialNB
                default_params = {
                    'alpha': 1.0,
                    'fit_prior': True
                }
                params = {**default_params, **self.model_params}
                self.model = MultinomialNB(**params)
            elif self.model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                default_params = {
                    'multi_class': 'multinomial',
                    'max_iter': 1000,
                    'solver': 'lbfgs',
                    'C': 1.0,
                    'class_weight': 'balanced'
                }
                params = {**default_params, **self.model_params}
                self.model = LogisticRegression(**params)
            elif self.model_type == 'svm':
                from sklearn.svm import SVC
                default_params = {
                    'kernel': 'linear',
                    'C': 1.0,
                    'probability': True,
                    'class_weight': 'balanced'
                }
                params = {**default_params, **self.model_params}
                self.model = SVC(**params)
            elif self.model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                default_params = {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'class_weight': 'balanced',
                    'random_state': 42
                }
                params = {**default_params, **self.model_params}
                self.model = RandomForestClassifier(**params)
            elif self.model_type == 'bagging':
                from sklearn.ensemble import BaggingClassifier
                from sklearn.tree import DecisionTreeClassifier
                # Using "estimator" per updated scikit-learn API.
                default_params = {
                    'estimator': DecisionTreeClassifier(max_depth=None),
                    'n_estimators': 100,
                    'max_samples': 1.0,
                    'max_features': 1.0,
                    'random_state': 42
                }
                params = {**default_params, **self.model_params}
                self.model = BaggingClassifier(**params)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            self.logger.info(f"Initialized {self.model_type} classifier with parameters:")
            self.logger.info(str(self.model.get_params()))
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """Fit the model using the provided training data."""
        return self._train_model(X, y, validation_data)

    def _train_model(self, X: np.ndarray, y: np.ndarray,
                     validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """Train the classification model with the specified algorithm."""
        try:
            if self.model_type == 'naive_bayes':
                min_val = np.min(X)
                if min_val < 0:
                    self._nb_shift = -min_val
                    X = X + self._nb_shift
                else:
                    self._nb_shift = 0

            training_history = {
                'model_type': self.model_type,
                'train_metrics': {},
                'val_metrics': {},
                'training_time': 0
            }
            start_time = time.time()
            self.model.fit(X, y)
            y_pred_train = self.model.predict(X)
            training_history['train_metrics'] = self._calculate_metrics(y, y_pred_train)

            if validation_data is not None:
                X_val, y_val = validation_data
                if self.model_type == 'naive_bayes':
                    X_val = X_val + self._nb_shift
                y_pred_val = self.model.predict(X_val)
                training_history['val_metrics'] = self._calculate_metrics(y_val, y_pred_val)

            training_history['training_time'] = time.time() - start_time

            if self.model_type == 'random_forest':
                training_history['feature_importances'] = self.model.feature_importances_.tolist()
            elif self.model_type == 'svm' and self.model.kernel == 'linear':
                training_history['feature_weights'] = self.model.coef_[0].tolist()

            return training_history

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise RuntimeError(f"Model training failed: {str(e)}")

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Helper method to calculate performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained classifier."""
        try:
            if self.model is None:
                raise RuntimeError("Model must be trained before making predictions")
            if self.model_type == 'naive_bayes':
                X = X + self._nb_shift
            return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities for each class."""
        try:
            if self.model is None:
                raise RuntimeError("Model must be trained before predicting probabilities")

            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                raise NotImplementedError(
                    f"Probability prediction not supported for {self.model_type}"
                )

        except Exception as e:
            self.logger.error(f"Error during probability prediction: {str(e)}")
            raise RuntimeError(f"Probability prediction failed: {str(e)}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if available for the model."""
        try:
            if self.model_type == 'random_forest':
                return {
                    f'feature_{i}': importance
                    for i, importance in enumerate(self.model.feature_importances_)
                }
            elif self.model_type == 'svm' and self.model.kernel == 'linear':
                return {
                    f'feature_{i}': weight
                    for i, weight in enumerate(self.model.coef_[0])
                }
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return None

    def get_params(self) -> Dict[str, Any]:
        """Get the model's parameters."""
        try:
            return {
                'model_type': self.model_type,
                'model_params': self.model.get_params(),
                'classes': self.classes_.tolist() if self.classes_ is not None else None
            }

        except Exception as e:
            self.logger.error(f"Error getting model parameters: {str(e)}")
            raise RuntimeError(f"Failed to get model parameters: {str(e)}")

    def save(self, path: str) -> None:
        """Save the model to disk."""
        try:
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'classes': self.classes_,
                'params': self.get_params()
            }

            joblib.dump(model_data, path)
            self.logger.info(f"Model saved successfully to {path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise RuntimeError(f"Failed to save model: {str(e)}")

    def load(self, path: str) -> None:
        """Load the model from disk."""
        try:
            model_data = joblib.load(path)

            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.classes_ = model_data['classes']

            self.logger.info(f"Model loaded successfully from {path}")
            self.logger.info(f"Model type: {self.model_type}")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")


class ModelFactory:
    """
    Factory class for creating and managing text classification models.

    This factory provides centralized model creation with standardized configuration
    and validation. It supports all five classification methods:
    - Naive Bayes
    - Logistic Regression
    - Support Vector Machines (SVM)
    - Random Forests
    - Bagging (Bootstrap Aggregating)

    The factory ensures consistent model initialization and configuration while
    providing flexibility for different use cases and requirements.
    """

    # Define supported model types and their default configurations
    _SUPPORTED_MODELS = {
        'naive_bayes': {
            'description': 'Multinomial Naive Bayes classifier for text classification',
            'default_params': {
                'alpha': 1.0,
                'fit_prior': True
            }
        },
        'logistic_regression': {
            'description': 'Logistic Regression with multinomial setting for multi-class classification',
            'default_params': {
                'multi_class': 'multinomial',
                'max_iter': 1000,
                'solver': 'lbfgs',
                'C': 1.0,
                'class_weight': 'balanced'
            }
        },
        'svm': {
            'description': 'Support Vector Machine with linear kernel for text classification',
            'default_params': {
                'kernel': 'linear',
                'C': 1.0,
                'probability': True,
                'class_weight': 'balanced'
            }
        },
        'random_forest': {
            'description': 'Random Forest ensemble for robust text classification',
            'default_params': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'class_weight': 'balanced',
                'random_state': 42
            }
        },
        'bagging': {
            'description': 'Bagging ensemble with decision trees for text classification',
            'default_params': {
                'n_estimators': 100,
                'max_samples': 1.0,
                'max_features': 1.0,
                'random_state': 42
            }
        }
    }

    @classmethod
    def create_model(cls,
                     model_type: str = 'svm',
                     model_config: Optional[Dict[str, Any]] = None,
                     **kwargs) -> TextClassifier:
        """Create and configure a text classification model."""
        logger = get_logger(__name__)

        try:
            model_type = model_type.lower()
            if model_type not in cls._SUPPORTED_MODELS:
                raise ValueError(
                    f"Unsupported model type: {model_type}. "
                    f"Supported types: {list(cls._SUPPORTED_MODELS.keys())}"
                )

            default_config = cls._SUPPORTED_MODELS[model_type]['default_params'].copy()

            if model_config:
                cls.validate_model_config(model_type, model_config)
                default_config.update(model_config)

            default_config.update(kwargs)

            logger.info(f"Creating {model_type} model with configuration:")
            logger.info(str(default_config))

            return TextClassifier(model_type=model_type, model_params=default_config)

        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            if isinstance(e, ValueError):
                raise e
            raise RuntimeError(f"Model creation failed: {str(e)}")

    @classmethod
    def get_supported_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about supported model types and their default configurations."""
        return cls._SUPPORTED_MODELS

    @classmethod
    def validate_model_config(cls, model_type: str, config: Dict[str, Any]) -> bool:
        """Validate model configuration parameters."""
        logger = get_logger(__name__)

        try:
            # Check model type
            if model_type not in cls._SUPPORTED_MODELS:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Get default configuration
            default_config = cls._SUPPORTED_MODELS[model_type]['default_params']

            # Check for invalid parameters
            invalid_params = set(config.keys()) - set(default_config.keys())
            if invalid_params:
                raise ValueError(
                    f"Invalid parameters for {model_type}: {invalid_params}"
                )

            # Validate parameter types
            for param, value in config.items():
                expected_type = type(default_config[param])
                if not isinstance(value, expected_type) and value is not None:
                    raise ValueError(
                        f"Invalid type for parameter '{param}'. "
                        f"Expected {expected_type.__name__}, got {type(value).__name__}"
                    )

            logger.info(f"Configuration validated successfully for {model_type}")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

    @classmethod
    def load_model(cls, path: str) -> TextClassifier:
        """Load a saved model from disk."""
        logger = get_logger(__name__)

        try:
            # Create a new classifier instance
            classifier = TextClassifier()

            # Load the model
            classifier.load(path)

            logger.info(f"Model loaded successfully from {path}")
            return classifier

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    @classmethod
    def create_ensemble(cls,
                        model_types: List[str],
                        configs: Optional[List[Dict[str, Any]]] = None) -> List[TextClassifier]:
        """Create an ensemble of different models for potential voting or stacking."""
        logger = get_logger(__name__)

        try:
            if configs and len(model_types) != len(configs):
                raise ValueError(
                    "Number of configurations must match number of models"
                )

            # Create models
            models = []
            for i, model_type in enumerate(model_types):
                config = configs[i] if configs else None
                model = cls.create_model(model_type, config)
                models.append(model)

            logger.info(f"Created ensemble with {len(models)} models")
            return models

        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
            raise RuntimeError(f"Ensemble creation failed: {str(e)}")


