import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from src.preprocessing.preprocessing import create_ordered_medical_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MedicalEntityRecognizer:
    """Medical entity recognition for clinical texts using a rule-based approach."""

    def __init__(self, disease_category: Optional[str] = None):
        """Initialize the medical entity recognizer."""
        self.disease_category = disease_category
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info("Initialized rule-based medical entity recognizer (spaCy and thinc disabled)")

        # Define entity categories based on exploratory data analysis (EDA)
        self.entity_categories = {
            'DISEASE': ['disease', 'syndrome', 'disorder', 'condition'],
            'SYMPTOM': ['symptom', 'manifestation', 'sign', 'indication'],
            'ANATOMY': ['muscle', 'nerve', 'brain', 'spine', 'respiratory'],
            'MEDICATION': ['drug', 'medication', 'treatment', 'therapy'],
            'MEASUREMENT': ['score', 'scale', 'rating', 'assessment']
        }

        # Disease-specific entities from EDA
        self.disease_entities = {
            'ALS': {
                'symptoms': ['respiratory decline', 'muscle weakness', 'bulbar dysfunction'],
                'measurements': ['FVC', 'ALSFRS-R'],
                'anatomy': ['motor neurons', 'respiratory muscles']
            },
            'OCD': {
                'symptoms': ['intrusive thoughts', 'compulsions', 'anxiety'],
                'measurements': ['Y-BOCS', 'severity scale'],
                'behaviors': ['ritual', 'repetitive behavior']
            },
            'Parkinson': {
                'symptoms': ['tremor', 'rigidity', 'bradykinesia'],
                'measurements': ['UPDRS', 'Hoehn and Yahr'],
                'anatomy': ['substantia nigra', 'basal ganglia']
            },
            'Dementia': {
                'symptoms': ['memory loss', 'cognitive decline', 'confusion'],
                'measurements': ['MMSE', 'CDR'],
                'domains': ['memory', 'executive function', 'behavior']
            },
            'Scoliosis': {
                'anatomy': ['spine', 'vertebrae', 'thoracic', 'lumbar'],
                'measurements': ['Cobb angle', 'curve degree'],
                'procedures': ['fusion', 'correction', 'brace']
            }
        }

        # Initialize preprocessing pipeline
        self.preprocessor = create_ordered_medical_pipeline(disease_category)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text using a simple rule-based approach.
        The method searches for keywords within the preprocessed text.
        """
        # Preprocess text
        processed = self.preprocessor.process(text)
        if isinstance(processed, tuple):
            processed = processed[0]
        processed_lower = processed.lower()

        # Extract entities based on keyword matching
        entities = defaultdict(list)
        for category, terms in self.entity_categories.items():
            for term in terms:
                if term in processed_lower:
                    entities[category].append(term)

        # Get disease-specific entities
        if self.disease_category:
            disease_specific = self._extract_disease_specific_entities(processed_lower)
            for category, terms in disease_specific.items():
                entities[category].extend(terms)

        return dict(entities)

    def _extract_disease_specific_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract disease-specific entities using keyword matching."""
        entities = defaultdict(list)
        if self.disease_category in self.disease_entities:
            disease_terms = self.disease_entities[self.disease_category]
            for category, terms in disease_terms.items():
                for term in terms:
                    if term.lower() in text:
                        entities[f"{self.disease_category}_{category}"].append(term)
        return dict(entities)

    def get_entity_features(self, text: str) -> Dict[str, float]:
        """Get numerical features based on entity analysis."""
        entities = self.extract_entities(text)
        total_entities = sum(len(e) for e in entities.values())

        features = {
            'total_entities': total_entities,
            'unique_entity_types': len(entities)
        }

        # Calculate density for each category
        words = text.split()
        total_words = len(words)
        for category in self.entity_categories:
            if category in entities:
                density = len(entities[category]) / total_words if total_words > 0 else 0.0
                features[f'{category.lower()}_density'] = density
            else:
                features[f'{category.lower()}_density'] = 0.0

        # Add disease-specific features
        if self.disease_category:
            disease_entities = {k: v for k, v in entities.items() if k.startswith(self.disease_category)}
            features['disease_specific_entities'] = sum(len(e) for e in disease_entities.values())

        return features


# Example usage and testing
if __name__ == "__main__":
    # Test texts
    test_texts = [
        """Patient with ALS showing respiratory decline. FVC = 65% ± 5%. 
           ALSFRS-R score decreased from 42 to 38 over 3 months.""",
        """Subject with severe ALS symptoms. Respiratory function declined.
           Motor function significantly impaired. Bulbar onset observed."""
    ]

    # Create entity recognizer
    recognizer = MedicalEntityRecognizer(disease_category='ALS')

    # Test entity extraction
    logger.info("\nTesting entity extraction:")
    for i, text in enumerate(test_texts, 1):
        logger.info(f"\nText {i}:")
        entities = recognizer.extract_entities(text)
        for category, terms in entities.items():
            logger.info(f"{category}: {', '.join(terms)}")

    # Test feature extraction
    logger.info("\nTesting feature extraction:")
    features = recognizer.get_entity_features(test_texts[0])
    for feature, value in features.items():
        logger.info(f"{feature}: {value:.4f}")
