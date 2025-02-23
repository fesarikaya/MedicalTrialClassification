import os
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.preprocessing.preprocessing import create_ordered_medical_pipeline
from src.features.tfidf_features import MedicalTextFeatureExtractor
from src.features.word_embeddings import MedicalWordEmbeddings
from src.features.entity_recognition import MedicalEntityRecognizer
from src.features.text_statistics import TextStatisticsExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreparator:
    """Prepare data for model development"""

    def __init__(self,
                 test_size: float = 0.15,
                 val_size: float = 0.15,
                 random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.logger = get_logger(self.__class__.__name__)

        # Initialize feature extractors
        self.feature_extractors = {
            'tfidf': MedicalTextFeatureExtractor(),
            'embeddings': MedicalWordEmbeddings(model_type='fasttext'),
            'entities': MedicalEntityRecognizer(),
            'statistics': TextStatisticsExtractor()
        }

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate the dataset"""
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded dataset with shape: {df.shape}")

            # Validate required columns
            required_columns = ['description', 'label']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns: {required_columns}")

            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        # First split: separate test set
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df['label'],
            random_state=self.random_state
        )

        # Second split: separate validation set
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val['label'],
            random_state=self.random_state
        )

        self.logger.info(f"Data split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test

    def extract_features(self, texts: List[str], feature_types: List[str]) -> np.ndarray:
        """Extract and combine features"""
        all_features = []

        for feature_type in feature_types:
            if feature_type not in self.feature_extractors:
                raise ValueError(f"Unknown feature type: {feature_type}")

            extractor = self.feature_extractors[feature_type]

            if feature_type == 'tfidf':
                # Check if the TF-IDF vectorizer is already fitted. If not, fit it.
                if not hasattr(extractor.vectorizer, 'vocabulary_'):
                    features, _ = extractor.fit_transform(texts)
                else:
                    features, _ = extractor.transform(texts)
            elif feature_type == 'embeddings':
                features = np.vstack([
                    extractor.get_document_embedding(text, method='weighted')
                    for text in texts
                ])
            elif feature_type == 'entities':
                features = np.vstack([
                    list(extractor.get_entity_features(text).values())
                    for text in texts
                ])
            else:  # statistics
                features = np.vstack([
                    extractor.get_feature_vector(text)
                    for text in texts
                ])

            all_features.append(features)

        return np.hstack(all_features)

    def _prepare_data_internal(self, file_path: str, feature_types: List[str]) -> Dict:
        """Internal method to load, split, and prepare data."""
        # Load data
        df = self.load_data(file_path)

        # Split data
        train_df, val_df, test_df = self.split_data(df)

        # Extract features
        train_features = self.extract_features(train_df['description'].tolist(), feature_types)
        val_features = self.extract_features(val_df['description'].tolist(), feature_types)
        test_features = self.extract_features(test_df['description'].tolist(), feature_types)

        # Prepare labels
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_df['label'])
        val_labels = label_encoder.transform(val_df['label'])
        test_labels = label_encoder.transform(test_df['label'])

        prepared_data = {
            'train': (train_features, train_labels),
            'val': (val_features, val_labels),
            'test': (test_features, test_labels),
            'label_encoder': label_encoder,
            'feature_types': feature_types
        }

        self.logger.info("Data preparation completed successfully")
        return prepared_data

    def save_prepared_data(self, prepared_data: Dict, output_dir: str):
        """Save prepared data to files"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save features and labels for each split
        for split_name in ['train', 'val', 'test']:
            features, labels = prepared_data[split_name]

            # Save features
            np.save(
                os.path.join(output_dir, f'{split_name}_features.npy'),
                features
            )

            # Save labels
            np.save(
                os.path.join(output_dir, f'{split_name}_labels.npy'),
                labels
            )

        # Save metadata
        metadata = {
            'label_encoder': prepared_data['label_encoder'],
            'feature_types': prepared_data['feature_types']
        }
        joblib.dump(
            metadata,
            os.path.join(output_dir, 'metadata.joblib')
        )

        self.logger.info(f"Saved prepared data to {output_dir}")

    def load_prepared_data(self, input_dir: str) -> Dict:
        """Load prepared data from files"""
        prepared_data = {}

        # Load features and labels for each split
        for split_name in ['train', 'val', 'test']:
            features = np.load(
                os.path.join(input_dir, f'{split_name}_features.npy')
            )
            labels = np.load(
                os.path.join(input_dir, f'{split_name}_labels.npy')
            )
            prepared_data[split_name] = (features, labels)

        # Load metadata
        metadata = joblib.load(
            os.path.join(input_dir, 'metadata.joblib')
        )
        prepared_data.update(metadata)

        self.logger.info(f"Loaded prepared data from {input_dir}")
        return prepared_data

    def prepare_data(self,
                     file_path: str,
                     output_dir: str,
                     feature_types: List[str] = ['tfidf', 'statistics']) -> Dict:
        """Main method to prepare data for modeling"""
        # Load and prepare data internally
        prepared_data = self._prepare_data_internal(file_path, feature_types)

        # Save prepared data
        self.save_prepared_data(prepared_data, output_dir)

        return prepared_data


if __name__ == "__main__":
    # Data preparation
    preparator = DataPreparator()

    # Prepare and save data
    prepared_data = preparator.prepare_data(
        file_path='trials.csv',
        output_dir='prepared_data',
        feature_types=['tfidf', 'statistics']
    )

    # Load prepared data
    loaded_data = preparator.load_prepared_data('prepared_data')

    # Verify data
    for split_name in ['train', 'val', 'test']:
        original_features, original_labels = prepared_data[split_name]
        loaded_features, loaded_labels = loaded_data[split_name]

        assert np.array_equal(original_features, loaded_features)
        assert np.array_equal(original_labels, loaded_labels)

        logger.info(rf"\n{split_name.capitalize()} set loaded successfully:")
        logger.info(rf"Features shape: {loaded_features.shape}")
        logger.info(rf"Labels shape: {loaded_labels.shape}")