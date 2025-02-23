import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from gensim.models import Word2Vec, FastText
from gensim.models.callbacks import CallbackAny2Vec
from src.preprocessing.preprocessing import create_ordered_medical_pipeline
from src.utils.logger import get_logger
from collections import defaultdict

logger = get_logger(__name__)


class EpochLogger(CallbackAny2Vec):
    """Callback to log training progress"""

    def __init__(self):
        self.epoch = 0
        self.logger = get_logger(self.__class__.__name__)

    def on_epoch_end(self, model):
        self.logger.info(f"Finished epoch {self.epoch}")
        self.epoch += 1


class MedicalWordEmbeddings:
    """
    Domain-specific word embeddings for medical text.
    Supports both Word2Vec and FastText models.
    """

    def __init__(self,
                 model_type: str = 'word2vec',
                 embedding_dim: int = 100,
                 window_size: int = 5,
                 min_count: int = 2,
                 disease_category: Optional[str] = None):
        """Initialize word embeddings model."""
        self.model_type = model_type.lower()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.disease_category = disease_category
        self.model = None
        self.logger = get_logger(self.__class__.__name__)

        # Initialize preprocessing pipeline
        self.preprocessor = create_ordered_medical_pipeline(
            disease_category=disease_category
        )

    def preprocess_texts(self, texts: List[str]) -> List[List[str]]:
        """Preprocess texts into tokens."""
        processed_texts = []
        for text in texts:
            # Apply preprocessing
            result = self.preprocessor.process(text)
            processed_text = result[0] if isinstance(result, tuple) else result
            # Tokenize
            tokens = processed_text.split()
            processed_texts.append(tokens)
        return processed_texts

    def train(self, texts: List[str], **kwargs):
        """Train word embeddings model."""
        # Preprocess texts
        processed_texts = self.preprocess_texts(texts)

        # Initialize model
        epoch_logger = EpochLogger()
        if self.model_type == 'word2vec':
            self.model = Word2Vec(
                sentences=processed_texts,
                vector_size=self.embedding_dim,
                window=self.window_size,
                min_count=self.min_count,
                workers=4,
                callbacks=[epoch_logger],
                **kwargs
            )
        elif self.model_type == 'fasttext':
            self.model = FastText(
                sentences=processed_texts,
                vector_size=self.embedding_dim,
                window=self.window_size,
                min_count=self.min_count,
                workers=4,
                callbacks=[epoch_logger],
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.logger.info(f"Trained {self.model_type} model with vocabulary size: {len(self.model.wv.key_to_index)}")

    def get_document_embedding(self, text: str, method: str = 'mean') -> np.ndarray:
        """Get embedding for a document."""
        if hasattr(text, 'text'):
            text = text.text

        # Preprocess text
        processed = self.preprocessor.process(text)
        if isinstance(processed, tuple):
            processed = processed[0]
        tokens = processed.split()

        # Get word vectors
        vectors = []
        weights = []
        for token in tokens:
            try:
                if self.model_type == 'word2vec':
                    vector = self.model.wv[token]
                else:
                    vector = self.model.wv.get_vector(token)
                vectors.append(vector)

                # Calculate weight based on medical term importance
                if token.lower() in self.preprocessor.context.preserved_terms:
                    weights.append(2.0)  # Higher weight for medical terms
                else:
                    weights.append(1.0)

            except KeyError:
                continue

        if not vectors:
            return np.zeros(self.embedding_dim)

        # Aggregate vectors
        if method == 'weighted':
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            return np.average(vectors, axis=0, weights=weights)
        else:
            return np.mean(vectors, axis=0)

    def get_similar_terms(self, term: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get similar terms for a given medical term."""
        try:
            return self.model.wv.most_similar(term, topn=n)
        except KeyError:
            self.logger.warning(f"Term not found in vocabulary: {term}")
            return []

    def save_model(self, path: str):
        """Save the model."""
        if self.model is not None:
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
        else:
            raise ValueError("No model to save")

    def load_model(self, path: str):
        """Load a saved model."""
        if self.model_type == 'word2vec':
            self.model = Word2Vec.load(path)
        else:
            self.model = FastText.load(path)
        self.logger.info(f"Model loaded from {path}")


# Example usage and testing
if __name__ == "__main__":
    # Test texts
    test_texts = [
        """Patient with ALS showing respiratory decline. FVC = 65% ± 5%. 
           ALSFRS-R score decreased from 42 to 38 over 3 months.""",
        """Subject with severe ALS symptoms. Respiratory function declined.
           Motor function significantly impaired.""",
        """ALS patient showing bulbar symptoms. FVC measurements indicate
           respiratory weakness. ALSFRS-R score: 35."""
    ]

    # Create and train embeddings model
    embeddings = MedicalWordEmbeddings(
        model_type='fasttext',
        embedding_dim=100,
        disease_category='ALS'
    )

    # Train model
    embeddings.train(test_texts)

    # Test document embedding
    test_doc = "New ALS patient with respiratory symptoms"
    doc_embedding = embeddings.get_document_embedding(test_doc, method='weighted')

    # Print results
    logger.info(f"\nDocument embedding shape: {doc_embedding.shape}")

    # Test similar terms
    similar_terms = embeddings.get_similar_terms('respiratory')
    logger.info("\nSimilar terms to 'respiratory':")
    for term, similarity in similar_terms:
        logger.info(f"- {term}: {similarity:.4f}")

    # Test model saving/loading
    embeddings.save_model('medical_embeddings.model')
    embeddings.load_model('medical_embeddings.model')