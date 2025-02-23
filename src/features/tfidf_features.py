from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import logging
import re
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing.preprocessing import create_ordered_medical_pipeline
from src.utils.logger import get_logger
from collections import defaultdict

logger = get_logger(__name__)


# Medical terms and patterns based on EDA findings
MEDICAL_TERMS = {
    # General medical terms
    'disease', 'syndrome', 'disorder', 'condition', 'symptoms',
    'treatment', 'therapy', 'medication', 'diagnosis', 'prognosis',
    'clinical', 'medical', 'therapeutic', 'assessment', 'evaluation'
}

# Disease-specific terms from n-gram analysis
DISEASE_TERMS = {
    'ALS': {
        'amyotrophic', 'lateral', 'sclerosis', 'bulbar', 'respiratory',
        'muscle', 'weakness', 'motor', 'function', 'strength',
        'vital', 'capacity', 'decline', 'progression', 'fvc'
    },
    'OCD': {
        'obsessive', 'compulsive', 'anxiety', 'ritual', 'behavior',
        'intrusive', 'thoughts', 'repetitive', 'severity', 'ybocs',
        'cognitive', 'behavioral', 'therapy', 'exposure', 'response'
    },
    'Parkinson': {
        'motor', 'tremor', 'rigidity', 'bradykinesia', 'balance',
        'gait', 'movement', 'dopamine', 'levodopa', 'dyskinesia',
        'updrs', 'hoehn', 'yahr', 'stage', 'progression'
    },
    'Dementia': {
        'cognitive', 'memory', 'alzheimer', 'decline', 'mental',
        'behavioral', 'function', 'caregivers', 'activities', 'mmse',
        'cdr', 'impairment', 'deterioration', 'confusion', 'awareness'
    },
    'Scoliosis': {
        'spine', 'curve', 'spinal', 'surgical', 'correction',
        'thoracic', 'lumbar', 'idiopathic', 'cobb', 'angle',
        'fusion', 'deformity', 'degree', 'brace', 'curvature'
    }
}

# Measurement patterns from EDA
MEASUREMENT_PATTERNS = {
    'numeric': r'\d+(?:\.\d+)?',
    'percentage': r'\d+(?:\.\d+)?\s*%',
    'range': r'\d+(?:\.\d+)?\s*(?:-|to)\s*\d+(?:\.\d+)?',
    'units': r'\d+(?:\.\d+)?\s*(?:mg|kg|ml|cm|mm|units)',
    'scores': r'\d+(?:\.\d+)?\s*(?:points?|score)',
    'plus_minus': r'\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?'
}

# Combined measurement pattern
MEASUREMENT_PATTERN = '|'.join(f'(?:{pattern})' for pattern in MEASUREMENT_PATTERNS.values())


class MedicalTfidfVectorizer(BaseEstimator, TransformerMixin):
    """Custom TF-IDF vectorizer with medical term weighting"""

    def __init__(self,
                 disease_category: Optional[str] = None,
                 max_features: Optional[int] = None,
                 min_df: Union[int, float] = 2,
                 max_df: Union[int, float] = 0.95):

        self.disease_category = disease_category
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.logger = get_logger(self.__class__.__name__)

        # Base TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            norm='l2',
            use_idf=True
        )

        # Term importance weights from EDA
        self.term_weights = {
            'disease_specific': 2.0,  # Disease-specific terms
            'measurements': 1.5,  # Medical measurements
            'medical_general': 1.2,  # General medical terms
            'common': 1.0  # Other terms
        }

    def _get_term_weight(self, term: str) -> float:
        """Determine weight for a specific term"""
        term = term.lower()

        # Check disease-specific terms
        if self.disease_category and term in DISEASE_TERMS.get(self.disease_category, set()):
            return self.term_weights['disease_specific']

        # Check medical terms
        if term in MEDICAL_TERMS:
            return self.term_weights['medical_general']

        # Check measurement patterns
        if re.search(MEASUREMENT_PATTERN, term):
            return self.term_weights['measurements']

        return self.term_weights['common']

    def _weight_matrix(self, X):
        """Apply term weights to TF-IDF matrix"""
        feature_names = self.vectorizer.get_feature_names_out()
        weights = np.array([self._get_term_weight(term) for term in feature_names])
        return X.multiply(weights)

    def fit(self, texts: List[str], y=None):
        """Fit the vectorizer"""
        self.logger.info("Fitting TF-IDF vectorizer")
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: List[str]):
        """Transform texts to weighted TF-IDF matrix"""
        self.logger.info("Transforming texts to TF-IDF features")
        X = self.vectorizer.transform(texts)
        return self._weight_matrix(X)

    def fit_transform(self, texts: List[str], y=None):
        """Fit and transform texts"""
        return self.fit(texts).transform(texts)

    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()

class MedicalTextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Complete pipeline for medical text preprocessing and feature extraction"""

    def __init__(self,
                 disease_category: Optional[str] = None,
                 config: Optional[Dict] = None):

        self.disease_category = disease_category
        # Default configuration
        self.config = {
            'max_length': 5000,
            'preserve_case': True,
            'include_scores': True,
            'standardize_terms': True,
            'handle_stopwords': True,
            'max_features': 1000,
            'min_df': 2,
            'max_df': 0.95
        }
        if config:
            self.config.update(config)

        # Initialize preprocessing pipeline
        self.preprocessor = create_ordered_medical_pipeline(
            disease_category=disease_category,
            config=self.config
        )

        # Initialize TF-IDF vectorizer
        self.vectorizer = MedicalTfidfVectorizer(
            disease_category=disease_category,
            max_features=self.config['max_features'],
            min_df=self.config['min_df'],
            max_df=self.config['max_df']
        )

        # Initialize text statistics calculator
        self.text_stats = TextStatisticsExtractor(disease_category)

        self.logger = get_logger(self.__class__.__name__)

    def fit(self, texts: List[str], y=None):
        """Fit the feature extraction pipeline"""
        self.logger.info("Starting feature extraction pipeline fitting")

        # Preprocess texts
        processed_texts = []
        for text in texts:
            result = self.preprocessor.process(text)
            processed_texts.append(result[0] if isinstance(result, tuple) else result)

        # Fit vectorizer
        self.vectorizer.fit(processed_texts)

        return self

    def transform(self, texts: List[str]) -> Tuple[np.ndarray, Dict]:
        """Transform texts to features"""
        self.logger.info("Transforming texts to features")

        # Preprocess texts
        processed_texts = []
        text_statistics = []

        for text in texts:
            # Apply preprocessing
            result = self.preprocessor.process(text)
            processed_text = result[0] if isinstance(result, tuple) else result
            processed_texts.append(processed_text)

            # Calculate text statistics
            stats = self.text_stats.extract_statistics(processed_text)
            text_statistics.append(stats)

        # Get TF-IDF features
        tfidf_features = self.vectorizer.transform(processed_texts)

        # Combine with text statistics
        combined_features = self._combine_features(tfidf_features, text_statistics)

        return combined_features, {
            'tfidf_shape': tfidf_features.shape,
            'statistics_features': len(text_statistics[0])
        }

    def _combine_features(self, tfidf_features: np.ndarray, text_statistics: List[Dict]) -> np.ndarray:
        """Combine TF-IDF features with text statistics"""
        # Convert text statistics to array
        stats_array = np.array([[
            stats['length'],
            stats['medical_term_density'],
            stats['measurement_density'],
            stats['disease_term_density']
        ] for stats in text_statistics])

        # Combine features
        if isinstance(tfidf_features, np.ndarray):
            return np.hstack((tfidf_features, stats_array))
        else:
            return np.hstack((tfidf_features.toarray(), stats_array))


class TextStatisticsExtractor:
    """Extract statistical features from medical texts"""

    def __init__(self, disease_category: Optional[str] = None):
        self.disease_category = disease_category
        self.disease_terms = set()
        if disease_category and disease_category in DISEASE_TERMS:
            self.disease_terms = DISEASE_TERMS[disease_category]

    def extract_statistics(self, text: str) -> Dict:
        """Extract statistical features from text"""
        # Basic text statistics
        words = text.split()
        total_words = len(words)

        # Medical term density
        medical_terms = sum(1 for word in words if word.lower() in MEDICAL_TERMS)
        medical_term_density = medical_terms / total_words if total_words > 0 else 0

        # Measurement density
        measurements = sum(1 for word in words if bool(re.search(MEASUREMENT_PATTERN, word)))
        measurement_density = measurements / total_words if total_words > 0 else 0

        # Disease-specific term density
        disease_terms = sum(1 for word in words if word.lower() in self.disease_terms)
        disease_term_density = disease_terms / total_words if total_words > 0 else 0

        return {
            'length': total_words,
            'medical_term_density': medical_term_density,
            'measurement_density': measurement_density,
            'disease_term_density': disease_term_density
        }


# Test and example usage
if __name__ == "__main__":
    # Test texts
    test_texts = [
        """Patient with ALS (amyotrophic lateral sclerosis) showing respiratory decline. 
           FVC = 65% ± 5%. ALSFRS-R score decreased from 42 to 38 over 3 months.""",
        """Subject with OCD experiencing severe anxiety. Y-BOCS score: 28. 
           Cognitive behavioral therapy initiated with exposure treatment.""",
        """Parkinson's disease patient showing increased tremor. UPDRS score of 45. 
           Levodopa dosage: 100mg/day."""
    ]

    # Test full pipeline
    extractor = MedicalTextFeatureExtractor(
        disease_category='ALS',
        config={
            'max_features': 1000,
            'min_df': 1,
            'max_df': 0.95
        }
    )

    # Fit and transform
    logger.info("Testing feature extraction pipeline...")

    # Fit the pipeline
    extractor.fit(test_texts)

    # Transform texts
    features, feature_info = extractor.transform(test_texts)

    # Print results
    logger.info(f"\nFeature Matrix Shape: {features.shape}")
    logger.info("\nFeature Information:")
    logger.info(f"- TF-IDF features: {feature_info['tfidf_shape'][1]}")
    logger.info(f"- Statistical features: {feature_info['statistics_features']}")

    # Print detailed feature analysis for first text
    logger.info("\nDetailed analysis of first text:")
    feature_names = extractor.vectorizer.get_feature_names()

    # Print top TF-IDF features
    logger.info("\nTop TF-IDF features:")
    non_zero = features[0].nonzero()[0]
    for idx in non_zero[:5]:
        if idx < len(feature_names):
            logger.info(f"- {feature_names[idx]}: {features[0][idx]:.4f}")

    # Print statistical features
    logger.info("\nStatistical features:")
    stats = extractor.text_stats.extract_statistics(test_texts[0])
    for key, value in stats.items():
        logger.info(f"- {key}: {value:.4f}")

    logger.info("\nFeature extraction test completed successfully!")