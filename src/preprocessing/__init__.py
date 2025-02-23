"""
Preprocessing module for medical text processing
Contains classes and functions for text preprocessing and standardization
"""

from .preprocessing import (
    # Main Classes
    TextPreprocessor,
    PreprocessingPipeline,
    ProcessingResult,
    PreprocessingStats,
    PreprocessingContext,

    # Preprocessor Classes
    LengthNormalizer,
    MedicalTermPreprocessor,
    SpecialCharacterHandler,
    DiseaseCategoryPreprocessor,
    MedicalScorePreprocessor,
    MedicalAbbreviationNormalizer,
    MedicalTermStandardizer,
    DomainSpecificStopwordHandler,
    SpecialCharacterCleaner,
    WhitespaceNormalizer,
    MedicalTokenizer,

    # Main Pipeline Creation Function
    create_ordered_medical_pipeline
)

# Version info
__version__ = '1.0.0'

# Default export for easy importing
__all__ = [
    # Main Classes
    'TextPreprocessor',
    'PreprocessingPipeline',
    'ProcessingResult',
    'PreprocessingStats',
    'PreprocessingContext',

    # Preprocessor Classes
    'LengthNormalizer',
    'MedicalTermPreprocessor',
    'SpecialCharacterHandler',
    'DiseaseCategoryPreprocessor',
    'MedicalScorePreprocessor',
    'MedicalAbbreviationNormalizer',
    'MedicalTermStandardizer',
    'DomainSpecificStopwordHandler',
    'SpecialCharacterCleaner',
    'WhitespaceNormalizer',
    'MedicalTokenizer',

    # Functions
    'create_ordered_medical_pipeline'
]

# Example usage documentation
'''
Example Usage:
from preprocessing import create_ordered_medical_pipeline

# Create pipeline
pipeline = create_ordered_medical_pipeline(
    disease_category='ALS',
    config={
        'max_length': 5000,
        'preserve_case': True,
        'include_scores': True
    }
)

# Process text
processed_text = pipeline.process("Sample medical text...")
'''