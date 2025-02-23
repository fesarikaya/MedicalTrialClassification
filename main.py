from flask import Flask, jsonify, request
import numpy as np
import joblib
import traceback
from pathlib import Path
from src.preprocessing.preprocessing import create_ordered_medical_pipeline
from src.utils.logger import get_logger
from typing import Dict, Any

# Initialize Flask app
app = Flask(__name__)


class PredictionService:
    """Service for handling model predictions"""

    def __init__(self, model_path: str = 'src/models/model.joblib'):
        self.model_path = Path(model_path)
        self.pipeline = None
        self.preprocessor = None
        self.logger = get_logger(__name__)
        try:
            self.initialize()
        except Exception as e:
            error_traceback = traceback.format_exc()
            self.logger.error(f"Initialization failed with error: {str(e)}")
            self.logger.error(f"Traceback: {error_traceback}")
            raise

    def initialize(self):
        """Initialize model pipeline"""
        try:
            self.logger.info(f"Looking for model at: {self.model_path.absolute()}")

            # Load model pipeline
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path.absolute()}")

            # Load the pipeline
            self.pipeline = joblib.load(self.model_path)
            self.logger.info("Model pipeline loaded successfully")

            # Log pipeline contents
            self.logger.info("Pipeline contents:")
            for key in self.pipeline.keys():
                self.logger.info(f"- Found component: {key}")

            # Initialize preprocessor
            self.logger.info("Initializing preprocessor...")
            self.preprocessor = create_ordered_medical_pipeline()
            self.logger.info("Preprocessor initialized successfully")

            # Log feature dimensions
            self.logger.info(f"Expected feature dimension: {self.pipeline['feature_dim']}")
            self.logger.info(f"Vectorizer vocabulary size: {len(self.pipeline['vectorizer'].vocabulary_)}")

        except Exception as e:
            error_traceback = traceback.format_exc()
            self.logger.error(f"Initialization failed with error: {str(e)}")
            self.logger.error(f"Traceback: {error_traceback}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction on input text"""
        try:
            # Validate input
            if not isinstance(text, str):
                raise ValueError("Input must be a string")
            if not text.strip():
                raise ValueError("Input text cannot be empty")

            # Preprocess text
            processed_text = self.preprocessor.process(text)
            if isinstance(processed_text, tuple):
                processed_text = processed_text[0]

            # Extract features using vectorizer
            features = self.pipeline['vectorizer'].transform([processed_text])
            features = features.toarray()

            # Verify feature dimension
            if features.shape[1] != self.pipeline['feature_dim']:
                raise ValueError(
                    f"Feature dimension mismatch: got {features.shape[1]}, "
                    f"expected {self.pipeline['feature_dim']}"
                )

            # Scale features
            features_scaled = self.pipeline['scaler'].transform(features)

            # Get the model - the model itself is a VotingClassifier
            model = self.pipeline['model']

            # Make prediction
            prediction = model.predict(features_scaled)[0]

            # Get prediction probability if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = float(np.max(probabilities))

            # Convert prediction using label encoder if available
            if 'metadata' in self.pipeline and 'label_encoder' in self.pipeline['metadata']:
                prediction = self.pipeline['metadata']['label_encoder'].inverse_transform([prediction])[0]

            return {
                'status': 'success',
                'prediction': prediction,
                'confidence': confidence
            }

        except Exception as e:
            error_traceback = traceback.format_exc()
            self.logger.error(f"Prediction failed: {str(e)}")
            self.logger.error(f"Traceback: {error_traceback}")
            raise


# Initialize service with error handling
try:
    prediction_service = PredictionService()
    app.logger.info("PredictionService initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize PredictionService: {str(e)}")
    traceback.print_exc()
    prediction_service = None


@app.route("/health")
def health_check():
    """Health check endpoint"""
    if prediction_service is None:
        return jsonify({
            'status': 'unhealthy',
            'error': 'Prediction service failed to initialize'
        }), 500

    health_info = {
        'status': 'healthy',
        'components': {
            'model': prediction_service.pipeline is not None and 'model' in prediction_service.pipeline,
            'vectorizer': prediction_service.pipeline is not None and 'vectorizer' in prediction_service.pipeline,
            'scaler': prediction_service.pipeline is not None and 'scaler' in prediction_service.pipeline,
            'preprocessor': prediction_service.preprocessor is not None
        }
    }
    return jsonify(health_info)


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    if prediction_service is None:
        return jsonify({
            'status': 'error',
            'message': 'Prediction service is not available'
        }), 503

    try:
        # Get request data
        data = request.get_json()

        # Validate request data
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        if 'description' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No description provided'
            }), 400

        # Get prediction
        result = prediction_service.predict(data['description'])

        return jsonify(result)

    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        error_traceback = traceback.format_exc()
        app.logger.error(f"Prediction failed: {str(e)}")
        app.logger.error(f"Traceback: {error_traceback}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': error_traceback
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
