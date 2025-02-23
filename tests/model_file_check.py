import joblib
from pathlib import Path
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def check_model_file():
    """Check if model file exists and examine its contents"""
    model_path = Path("../src/models/model.joblib")

    # Check if file exists
    if not model_path.exists():
        print(f"Model file not found at {model_path}")
        return

    print(f"Model file found at {model_path}")

    try:
        # Load the model
        model_data = joblib.load(model_path)
        print("\nModel data type:", type(model_data))

        # Check if it's a dictionary or direct model
        if isinstance(model_data, dict):
            print("\nModel is saved as a dictionary with components:")
            for key, value in model_data.items():
                print(f"- {key}: {type(value)}")
        else:
            print("\nModel is saved directly as:", type(model_data))

        # Check for VotingClassifier
        if isinstance(model_data, VotingClassifier) or (
                isinstance(model_data, dict) and isinstance(model_data.get('model'), VotingClassifier)):
            print("\nVotingClassifier found in model")
            voting_clf = model_data if isinstance(model_data, VotingClassifier) else model_data['model']
            print("- Estimators:", [est[0] for est in voting_clf.estimators])
            print("- Voting:", voting_clf.voting)

        # Check for scaler
        if isinstance(model_data, dict) and 'scaler' in model_data:
            print("\nScaler found in model:")
            print("- Type:", type(model_data['scaler']))
            if isinstance(model_data['scaler'], StandardScaler):
                print("- Feature count:", model_data['scaler'].n_features_in_)

        # Check for vectorizer
        if isinstance(model_data, dict) and 'vectorizer' in model_data:
            print("\nVectorizer found in model:")
            print("- Type:", type(model_data['vectorizer']))
            if isinstance(model_data['vectorizer'], TfidfVectorizer):
                print("- Vocabulary size:", len(model_data['vectorizer'].vocabulary_) if model_data[
                    'vectorizer'].vocabulary_ else "Not fitted")
        else:
            print("\nNo vectorizer found in model!")

    except Exception as e:
        print(f"\nError examining model file: {str(e)}")


def check_model_structure():
    """Print detailed model structure"""
    model_path = Path("../src/models/model.joblib")

    # Load the model
    pipeline = joblib.load(model_path)

    print("\nModel Pipeline Structure:")
    print("=======================")

    # Check all top-level keys
    print("\nTop level keys:", list(pipeline.keys()))

    # Check model structure
    print("\nModel structure:")
    model = pipeline['model']
    if isinstance(model, dict):
        print("Model is a dictionary containing:", list(model.keys()))
    else:
        print(f"Model is of type: {type(model)}")
        print("Model attributes:", dir(model))

    # Print other components
    print("\nOther components:")
    for key, value in pipeline.items():
        if key != 'model':
            print(f"{key}: {type(value)}")
            if hasattr(value, 'get_params'):
                print(f"Parameters: {value.get_params()}")


if __name__ == "__main__":
    check_model_file()
    check_model_structure()

