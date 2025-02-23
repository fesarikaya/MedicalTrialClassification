import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from src.preprocessing.preprocessing import create_ordered_medical_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextStatisticsExtractor:
    """Extract statistical features from medical texts based on EDA findings"""

    def __init__(self, disease_category: Optional[str] = None):
        self.disease_category = disease_category
        self.logger = get_logger(self.__class__.__name__)

        # Initialize preprocessing pipeline
        self.preprocessor = create_ordered_medical_pipeline(
            disease_category=disease_category
        )

        # Measurement patterns from EDA
        self.measurement_patterns = {
            'scores': r'\d+(?:\s*(?:points?|score))',
            'percentages': r'\d+(?:\.\d+)?\s*%',
            'ranges': r'\d+\s*(?:-|to)\s*\d+',
            'units': r'\d+(?:\.\d+)?\s*(?:mg|kg|ml|cm|mm)',
            'plus_minus': r'\d+\s*±\s*\d+'
        }

        # Disease-specific patterns from EDA
        self.disease_patterns = {
            'ALS': {
                'scores': r'(?:ALSFRS-R|FVC)',
                'measurements': r'\d+\s*(?:fvc|alsfrs)',
                'time_patterns': r'(?:months?|years?)\s*(?:decline|progression)'
            },
            'OCD': {
                'scores': r'(?:Y-BOCS|severity)',
                'measurements': r'\d+\s*(?:ybocs|severity)',
                'time_patterns': r'(?:frequency|duration)\s*of\s*(?:symptoms|behaviors)'
            },
            'Parkinson': {
                'scores': r'(?:UPDRS|Hoehn)',
                'measurements': r'\d+\s*(?:updrs|stage)',
                'time_patterns': r'(?:onset|progression|duration)'
            },
            'Dementia': {
                'scores': r'(?:MMSE|CDR)',
                'measurements': r'\d+\s*(?:mmse|cdr)',
                'time_patterns': r'(?:months?|years?)\s*(?:decline|progression)'
            },
            'Scoliosis': {
                'scores': r'(?:Cobb|curve)',
                'measurements': r'\d+\s*(?:degree|angle)',
                'time_patterns': r'(?:growth|progression|correction)'
            }
        }

    def extract_basic_statistics(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics"""
        # Preprocess text
        processed = self.preprocessor.process(text)
        if isinstance(processed, tuple):
            processed = processed[0]

        words = processed.split()
        sentences = [s.strip() for s in processed.split('.') if s.strip()]

        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0
        }

    def extract_measurement_statistics(self, text: str) -> Dict[str, float]:
        """Extract measurement-related statistics"""
        stats = {}

        # Count measurements by type
        for name, pattern in self.measurement_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            stats[f'{name}_count'] = sum(1 for _ in matches)

        # Calculate measurement density
        total_measurements = sum(stats.values())
        words = text.split()
        stats['measurement_density'] = total_measurements / len(words) if words else 0

        return stats

    def extract_disease_specific_statistics(self, text: str) -> Dict[str, float]:
        """Extract disease-specific statistics"""
        stats = {}

        if self.disease_category and self.disease_category in self.disease_patterns:
            patterns = self.disease_patterns[self.disease_category]

            for name, pattern in patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                stats[f'{self.disease_category.lower()}_{name}_count'] = sum(1 for _ in matches)

        return stats

    def extract_readability_statistics(self, text: str) -> Dict[str, float]:
        """Extract readability statistics"""
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        # Count syllables (simple approximation)
        def count_syllables(word):
            return len(re.findall(r'[aeiou]+', word.lower())) + 1

        syllable_counts = [count_syllables(w) for w in words]

        stats = {
            'avg_syllables_per_word': sum(syllable_counts) / len(words) if words else 0,
            'complex_words_ratio': sum(1 for c in syllable_counts if c > 2) / len(words) if words else 0
        }

        # Approximate Flesch Reading Ease
        if words and sentences:
            stats['flesch_reading_ease'] = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (
                    sum(syllable_counts) / len(words))
        else:
            stats['flesch_reading_ease'] = 0

        return stats

    def extract_all_statistics(self, text: str) -> Dict[str, float]:
        """Extract all statistical features"""
        # Collect all statistics
        stats = {}

        # Basic statistics
        stats.update(self.extract_basic_statistics(text))

        # Measurement statistics
        stats.update(self.extract_measurement_statistics(text))

        # Disease-specific statistics
        stats.update(self.extract_disease_specific_statistics(text))

        # Readability statistics
        stats.update(self.extract_readability_statistics(text))

        return stats

    def get_feature_vector(self, text: str) -> np.ndarray:
        """Convert statistics to feature vector"""
        stats = self.extract_all_statistics(text)
        return np.array(list(stats.values()))

    def get_feature_names(self) -> List[str]:
        """Get names of statistical features"""
        # Extract features from a sample text to get all feature names
        stats = self.extract_all_statistics("Sample text")
        return list(stats.keys())


# Example usage and testing
if __name__ == "__main__":
    # Test texts
    test_texts = [
        """Patient with ALS showing respiratory decline. FVC = 65% ± 5%. 
           ALSFRS-R score decreased from 42 to 38 over 3 months.""",
        """Subject with severe OCD symptoms. Y-BOCS score: 28. 
           Treatment includes cognitive behavioral therapy with daily monitoring.""",
        """Parkinson's disease patient showing increased tremor. UPDRS score of 45. 
           Started on levodopa 100mg/day with three-month follow-up."""
    ]

    # Test for different disease categories
    for disease in ['ALS', 'OCD', 'Parkinson']:
        logger.info(f"\nAnalyzing {disease} text:")

        # Create statistics extractor
        extractor = TextStatisticsExtractor(disease_category=disease)

        # Get relevant test text
        text = test_texts[['ALS', 'OCD', 'Parkinson'].index(disease)]

        # Extract all statistics
        stats = extractor.extract_all_statistics(text)

        # Print results
        logger.info("\nText Statistics:")
        for feature, value in stats.items():
            logger.info(f"{feature}: {value:.4f}")

        # Get feature vector
        feature_vector = extractor.get_feature_vector(text)
        logger.info(f"\nFeature vector shape: {feature_vector.shape}")

        # Get feature names
        feature_names = extractor.get_feature_names()
        logger.info(f"Number of features: {len(feature_names)}")