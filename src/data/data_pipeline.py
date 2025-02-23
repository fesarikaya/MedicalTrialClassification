import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Generator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import joblib
from pathlib import Path
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPipeline:
    """
    Data pipeline for model training with advanced data handling capabilities

    Features:
    - Stratified sampling
    - Cross-validation splits
    - Data caching
    - Batch generation
    """

    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 n_splits: int = 5,
                 cache_dir: Optional[str] = None,
                 random_state: int = 42):
        # Initialize data pipeline
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.random_state = random_state
        self.logger = get_logger(self.__class__.__name__)

        # Load prepared data
        self.load_data()

        # Initialize cross-validation splitter
        self.cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

    def load_data(self):
        """Load prepared data from files"""
        try:
            # Load features and labels for each split
            self.data = {}
            for split in ['train', 'val', 'test']:
                features_path = self.data_dir / f'{split}_features.npy'
                labels_path = self.data_dir / f'{split}_labels.npy'
                features = np.load(features_path)
                labels = np.load(labels_path)
                self.data[split] = (features, labels)

            # Load metadata
            metadata_path = self.data_dir / 'metadata.joblib'
            self.metadata = joblib.load(metadata_path)

            self.logger.info("Data loaded successfully")
            self._log_data_info()

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _log_data_info(self):
        """Log information about loaded data"""
        for split, (features, labels) in self.data.items():
            self.logger.info(f"\n{split.capitalize()} set:")
            self.logger.info(f"Features shape: {features.shape}")
            self.logger.info(f"Labels shape: {labels.shape}")
            self.logger.info(f"Classes: {np.unique(labels)}")

    def get_cv_splits(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate cross-validation splits"""
        features, labels = self.data['train']
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(features, labels)):
            self.logger.info(f"Generating split for fold {fold + 1}/{self.n_splits}")
            yield train_idx, val_idx

    def get_batch_generator(self,
                            split: str,
                            batch_size: Optional[int] = None,
                            shuffle_data: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate batches of data"""
        features, labels = self.data[split]
        batch_size = batch_size if batch_size is not None else self.batch_size

        # Create cache key if caching is enabled
        if self.cache_dir:
            cache_key = f"{split}_batch_{batch_size}"
            if not self._check_cache(cache_key):
                # Save the original, unshuffled data to cache.
                self._save_to_cache(cache_key, features, labels)
            else:
                features, labels = self._load_from_cache(cache_key)

        # Shuffle if requested (the caching preserves the original order)
        if shuffle_data:
            features, labels = shuffle(features, labels, random_state=self.random_state)

        num_samples = len(features)
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            yield features[start_idx:end_idx], labels[start_idx:end_idx]

    def get_all_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get all data for a split"""
        return self.data[split]

    def _check_cache(self, key: str) -> bool:
        """Check if data exists in cache"""
        if not self.cache_dir:
            return False

        features_path = self.cache_dir / f"{key}_features.npy"
        labels_path = self.cache_dir / f"{key}_labels.npy"
        return features_path.exists() and labels_path.exists()

    def _load_from_cache(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from cache"""
        features = np.load(self.cache_dir / f"{key}_features.npy")
        labels = np.load(self.cache_dir / f"{key}_labels.npy")
        self.logger.info(f"Loaded cache with key: {key}")
        return features, labels

    def _save_to_cache(self, key: str, features: np.ndarray, labels: np.ndarray):
        """Save data to cache"""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{key}_features.npy", features)
            np.save(self.cache_dir / f"{key}_labels.npy", labels)
            self.logger.info(f"Saved cache with key: {key}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline - adjust paths as needed
    pipeline = DataPipeline(
        data_dir='../../data/prepared_data',
        batch_size=32,
        n_splits=5,
        cache_dir='../../data/cache'
    )

    # Test cross-validation splits
    logger.info("\nTesting cross-validation splits:")
    for fold, (train_idx, val_idx) in enumerate(pipeline.get_cv_splits()):
        logger.info(f"Fold {fold + 1}:")
        logger.info(f"Training samples: {len(train_idx)}")
        logger.info(f"Validation samples: {len(val_idx)}")

    # Test batch generator
    logger.info("\nTesting batch generator:")
    for split in ['train', 'val', 'test']:
        batch_gen = pipeline.get_batch_generator(split)
        num_batches = 0
        for features_batch, labels_batch in batch_gen:
            num_batches += 1
        logger.info(f"{split.capitalize()} batches generated: {num_batches}")
