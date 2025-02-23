import os
import numpy as np
import pandas as pd
import joblib
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the optimized model architecture
from src.models.model_experiments_2 import OptimizedModelArchitecture


def main():
    # Paths to the prepared data
    data_dir = Path("../../data/prepared_data")

    print("Loading prepared data...")
    # Load features and labels
    X_train = np.load(data_dir / "train_features.npy")
    X_val = np.load(data_dir / "val_features.npy")
    X_test = np.load(data_dir / "test_features.npy")

    y_train = np.load(data_dir / "train_labels.npy")
    y_val = np.load(data_dir / "val_labels.npy")
    y_test = np.load(data_dir / "test_labels.npy")

    # Load metadata
    metadata = joblib.load(data_dir / "metadata.joblib")

    print("Data shapes:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create and initialize the optimized model architecture
    print("Creating and training model...")
    opt_model = OptimizedModelArchitecture(random_state=42)
    model_dict = opt_model.create_optimized_model()
    voting_model = opt_model.voting_classifier

    # Train the voting classifier
    voting_model.fit(X_train_scaled, y_train)
    print("Voting classifier model successfully trained.")

    # Validate the model
    y_val_pred = voting_model.predict(X_val_scaled)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    print(f"Validation F1 Score: {val_f1:.4f}")

    # Test the model
    y_test_pred = voting_model.predict(X_test_scaled)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Test F1 Score: {test_f1:.4f}")

    # Save the model pipeline
    print("Saving model pipeline...")
    model_pipeline = {
        'model': voting_model,
        'scaler': scaler,
        'metadata': metadata,  # Include metadata for reference
        'feature_dim': X_train.shape[1]  # Store feature dimensionality
    }

    # Create models directory if it doesn't exist
    models_dir = Path("../../src/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save the pipeline
    joblib.dump(model_pipeline, models_dir / "model.joblib")
    print("Model pipeline saved as 'model.joblib'")

    train_features = np.load(data_dir / "train_features.npy")
    print(f"Training features shape: {train_features.shape}")

    # Load metadata
    metadata = joblib.load(data_dir / "metadata.joblib")
    print("Feature types from metadata:", metadata['feature_types'])

    model_path = models_dir / "model.joblib"

    # Load the existing model
    if model_path.exists():
        model_dict = joblib.load(model_path)
        # Get actual voting classifier from potentially nested structure
        while isinstance(model_dict.get('model', None), dict):
            model_dict = model_dict['model']
        voting_model = model_dict.get('model', None)
        if voting_model is None:
            raise ValueError("Could not find voting classifier in model file")
        print("Loaded existing voting classifier")
    else:
        raise FileNotFoundError("Original model file not found!")

    # Initialize vectorizer with parameters matching data preparation
    vectorizer = TfidfVectorizer(
        max_features=train_features.shape[1],  # Match the training feature dimension
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )

    # Load raw data
    df = pd.read_csv("../../data/trials.csv")
    texts = df['description'].tolist()
    print(f"Loaded {len(texts)} training texts")

    # Fit vectorizer
    X = vectorizer.fit_transform(texts)
    print(f"TF-IDF features shape: {X.shape}")

    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(train_features)  # Fit on original training features
    print("Fitted scaler on training features")

    # Create pipeline with simplified structure
    model_pipeline = {
        'model': voting_model,  # Save the voting classifier directly
        'vectorizer': vectorizer,
        'scaler': scaler,
        'feature_dim': train_features.shape[1],
        'metadata': {
            'feature_types': metadata['feature_types'],
            'label_encoder': metadata['label_encoder']
        }
    }

    # Save pipeline safely using a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        print(f"Saving pipeline to temporary file: {tmp_file.name}")
        joblib.dump(model_pipeline, tmp_file.name)

        # Verify the saved file
        print("Verifying saved pipeline...")
        loaded_test = joblib.load(tmp_file.name)

        # Check structure
        required_components = ['model', 'vectorizer', 'scaler', 'feature_dim', 'metadata']
        if not all(k in loaded_test for k in required_components):
            raise ValueError("Saved pipeline verification failed!")

        if not hasattr(loaded_test['model'], 'predict'):
            raise ValueError("Saved model does not have predict method!")

        # If verification passes, move the file to final location
        if model_path.exists():
            backup_path = model_path.with_suffix('.backup')
            print(f"Creating backup of existing model: {backup_path}")
            os.rename(model_path, backup_path)

        print(f"Moving verified pipeline to final location: {model_path}")
        os.rename(tmp_file.name, model_path)
        print("Pipeline saved successfully")

        # Final verification
        final_pipeline = joblib.load(model_path)
        print("\nFinal verification of saved pipeline:")
        print("Components found:")
        for key, value in final_pipeline.items():
            print(f"- {key}: {type(value)}")

        # Verify model has predict method
        if not hasattr(final_pipeline['model'], 'predict'):
            raise ValueError("Final model verification failed: model has no predict method")

        print("Pipeline verification completed successfully!")

if __name__ == "__main__":
    main()
