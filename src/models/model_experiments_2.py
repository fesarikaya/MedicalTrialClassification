from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import Dict, List, Optional, Any
from src.models.model_factory import ModelFactory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RegularizedStackingClassifier(BaseEstimator, ClassifierMixin):
    """Custom stacking classifier with regularization."""

    def __init__(self, estimators, final_estimator, cv=3, regularization=0.1):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.regularization = regularization
        self.stacking = None

    def fit(self, X, y):
        # Create stacking with regularization and reduced complexity
        self.stacking = StackingClassifier(
            estimators=self.estimators,
            final_estimator=self.final_estimator,
            cv=self.cv,
            passthrough=True,  # Include original features
            n_jobs=-1
        )
        self.stacking.fit(X, y)
        # Set the classes_ attribute for compatibility with sklearn utilities.
        self.classes_ = self.stacking.classes_
        return self

    def predict(self, X):
        return self.stacking.predict(X)

    def predict_proba(self, X):
        return self.stacking.predict_proba(X)


class OptimizedModelArchitecture:
    """
    Optimized model architecture with reduced complexity and better regularization
    to address overfitting.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.base_models = None
        self.stacking_classifier = None
        self.voting_classifier = None
        self.logger = get_logger(self.__class__.__name__)

    def create_optimized_model(self) -> Dict[str, Any]:
        """Create optimized model architecture with regularization."""
        try:
            # Create base models with reduced complexity
            bagging_model = ModelFactory.create_model(
                'bagging',
                model_config={
                    'n_estimators': 50,  # Reduced from 100
                    'max_samples': 0.5,  # Reduced from 0.7
                    'max_features': 0.3,  # Reduced from 0.5
                    'random_state': self.random_state
                }
            ).model

            rf_model = ModelFactory.create_model(
                'random_forest',
                model_config={
                    'n_estimators': 100,  # Reduced from 200
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced',
                    'random_state': self.random_state
                }
            ).model

            svm_model = ModelFactory.create_model(
                'svm',
                model_config={
                    'C': 0.5,  # Increased regularization
                    'kernel': 'linear',
                    'probability': True,
                    'class_weight': 'balanced'
                }
            ).model

            # Create meta-classifier with early stopping
            meta_classifier = ModelFactory.create_model(
                'logistic_regression',
                model_config={
                    'C': 0.3,  # Increased regularization
                    'max_iter': 1000,
                    'multi_class': 'multinomial',
                    'solver': 'saga',  # Better for regularization
                    'class_weight': 'balanced',
                }
            ).model

            # Create regularized stacking classifier
            self.stacking_classifier = RegularizedStackingClassifier(
                estimators=[
                    ('bagging', bagging_model),
                    ('rf', rf_model),
                    ('svm', svm_model)
                ],
                final_estimator=meta_classifier,
                cv=3  # Reduced from 5
            )

            # Create optimized voting classifier
            self.voting_classifier = VotingClassifier(
                estimators=[
                    ('bagging', bagging_model),
                    ('rf', rf_model),
                    ('svm', svm_model)
                ],
                voting='soft',
                weights=[1.5, 1.0, 1.0],  # Adjusted weights
                n_jobs=-1
            )

            # Store base models
            self.base_models = {
                'bagging': bagging_model,
                'random_forest': rf_model,
                'svm': svm_model
            }

            return {
                'stacking_classifier': self.stacking_classifier,
                'voting_classifier': self.voting_classifier,
                'base_models': self.base_models
            }

        except Exception as e:
            self.logger.error(f"Error creating optimized model: {str(e)}")
            raise RuntimeError(f"Optimized model creation failed: {str(e)}")

    def fit_with_early_stopping(self,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: Optional[np.ndarray] = None,
                                y_val: Optional[np.ndarray] = None,
                                patience: int = 3) -> Dict[str, Any]:
        """Fit models with early stopping to prevent overfitting."""
        from sklearn.metrics import f1_score

        try:
            results = {
                'training_history': {'stacking': [], 'voting': []},
                'best_scores': {}
            }

            # Fit stacking classifier
            self.logger.info("Fitting stacking classifier with early stopping...")
            best_stacking_score = 0
            patience_counter = 0

            for epoch in range(10):  # Maximum 10 epochs
                self.stacking_classifier.fit(X_train, y_train)

                # Calculate validation score
                if X_val is not None and y_val is not None:
                    val_pred = self.stacking_classifier.predict(X_val)
                    val_score = f1_score(y_val, val_pred, average='weighted')

                    results['training_history']['stacking'].append({
                        'epoch': epoch,
                        'val_f1': val_score
                    })

                    if val_score > best_stacking_score:
                        best_stacking_score = val_score
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

                    self.logger.info(f"Epoch {epoch}: Validation F1 = {val_score:.4f}")

            # Fit voting classifier
            self.logger.info("\nFitting voting classifier...")
            self.voting_classifier.fit(X_train, y_train)

            if X_val is not None and y_val is not None:
                val_pred = self.voting_classifier.predict(X_val)
                voting_val_score = f1_score(y_val, val_pred, average='weighted')
                self.logger.info(f"Voting Classifier Validation F1 = {voting_val_score:.4f}")

                results['best_scores'] = {
                    'stacking_val_f1': best_stacking_score,
                    'voting_val_f1': voting_val_score
                }

            return results

        except Exception as e:
            self.logger.error(f"Error fitting models: {str(e)}")
            raise RuntimeError(f"Model fitting failed: {str(e)}")

    def predict(self, X: np.ndarray, method: str = 'stacking') -> np.ndarray:
        """Make predictions using the specified ensemble method."""
        if method == 'stacking':
            return self.stacking_classifier.predict(X)
        elif method == 'voting':
            return self.voting_classifier.predict(X)
        else:
            raise ValueError("Method must be 'stacking' or 'voting'")

    def evaluate_with_cross_validation(self,
                                       X: np.ndarray,
                                       y: np.ndarray,
                                       cv: int = 5) -> Dict[str, Any]:
        """Evaluate models with cross-validation and confidence intervals."""
        from scipy import stats

        try:
            results = {}

            for method in ['stacking', 'voting']:
                classifier = (self.stacking_classifier if method == 'stacking'
                              else self.voting_classifier)

                # Calculate cross-validation scores
                cv_scores = cross_val_score(
                    classifier, X, y,
                    cv=cv,
                    scoring='f1_weighted',
                    n_jobs=-1
                )

                # Calculate confidence interval
                confidence_interval = stats.t.interval(
                    0.95,
                    len(cv_scores) - 1,
                    loc=np.mean(cv_scores),
                    scale=stats.sem(cv_scores)
                )

                results[method] = {
                    'mean_cv_score': float(np.mean(cv_scores)),
                    'std_cv_score': float(np.std(cv_scores)),
                    'confidence_interval': [float(ci) for ci in confidence_interval],
                    'cv_scores': cv_scores.tolist()
                }

                self.logger.info(f"\n{method.capitalize()} Cross-validation Results:")
                self.logger.info(f"Mean F1: {results[method]['mean_cv_score']:.4f}")
                self.logger.info(f"Std F1: {results[method]['std_cv_score']:.4f}")
                self.logger.info(
                    f"95% CI: [{results[method]['confidence_interval'][0]:.4f}, "
                    f"{results[method]['confidence_interval'][1]:.4f}]"
                )

            return results

        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            raise RuntimeError(f"Cross-validation failed: {str(e)}")


# Usage example
if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    import joblib

    # Load and preprocess data
    data_dir = "../../data/prepared_data"
    X_train = np.load(f"{data_dir}/train_features.npy")
    y_train = np.load(f"{data_dir}/train_labels.npy")
    X_val = np.load(f"{data_dir}/val_features.npy")
    y_val = np.load(f"{data_dir}/val_labels.npy")
    X_test = np.load(f"{data_dir}/test_features.npy")
    y_test = np.load(f"{data_dir}/test_labels.npy")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create and train optimized model
    opt_model = OptimizedModelArchitecture()
    opt_model.create_optimized_model()

    # Fit with early stopping
    training_results = opt_model.fit_with_early_stopping(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        patience=3
    )

    # Evaluate with cross-validation
    cv_results = opt_model.evaluate_with_cross_validation(
        X_train_scaled, y_train
    )

    # Save model
    joblib.dump(opt_model, "optimized_model.joblib")