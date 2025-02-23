from abc import ABC, abstractmethod
from typing import List, Set, Dict, Optional, Union, Tuple
import re
import string
import logging
from dataclasses import dataclass
from collections import defaultdict
from nltk.corpus import stopwords as nltk_stopwords
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------
# Utility function
# ---------------------------
def _get_plain_text(text: Union[str, "ProcessingResult"]) -> str:
    """Helper to extract plain text from a ProcessingResult or any object with a .text attribute."""
    if hasattr(text, "text"):
        return text.text
    if isinstance(text, str):
        return text
    return str(text)

# ---------------------------
# Data Classes
# ---------------------------
@dataclass
class ProcessingResult:
    """Data class to hold processing results and metrics"""
    text: str
    original_length: int
    processed_length: int
    special_chars_removed: Dict[str, int]
    numbers_detected: List[str]
    medical_terms_found: List[str]
    abbreviations_normalized: Dict[str, str]

    def split(self):
        """Splits the text into tokens"""
        return self.text.split()

    def __len__(self):
        """Returns the length of the text"""
        return len(self.text)

@dataclass
class PreprocessingStats:
    """Statistics about the preprocessing operation"""
    original_length: int
    processed_length: int
    special_chars_removed: int
    numbers_found: int
    medical_terms_preserved: int

class PreprocessingContext:
    """Context for preprocessing operations"""
    def __init__(self):
        self.preserved_terms: Set[str] = set()
        self.stats = {}
        self.original_text: str = ""
        self.metadata: Dict = {}

# ---------------------------
# Abstract Base Class
# ---------------------------
class TextPreprocessor(ABC):
    """Abstract base class for text preprocessors with enhanced medical text capabilities"""
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        # Medical abbreviations based on EDA findings
        self.medical_abbreviations = {
            'ALS': 'amyotrophic lateral sclerosis',
            'OCD': 'obsessive compulsive disorder',
            'PD': "parkinson's disease",
            'AD': "alzheimer's disease",
            'CBT': 'cognitive behavioral therapy'
        }
        # Disease-specific terms to preserve (from n-gram analysis)
        self.preserve_terms = {
            'lateral sclerosis', 'amyotrophic', 'respiratory', 'muscle strength',
            'compulsive', 'obsessive', 'anxiety', 'behavioral therapy',
            'motor', 'levodopa', 'dopaminergic', 'movement disorders',
            'cognitive', 'alzheimer', 'caregivers', 'memory',
            'spine', 'curve', 'surgical', 'idiopathic'
        }
        # Medical measurements (from special character analysis)
        self.measurement_patterns = [
            r'\d+\s*mg', r'\d+\s*kg', r'\d+\s*ml',
            r'\d+\s*cm', r'\d+\s*mm', r'\d+\s*units'
        ]
        # Important special characters to handle (based on EDA)
        self.special_chars_handling = {
            '(': 'preserve',  # Important for medical context
            ')': 'preserve',
            '.': 'preserve_with_space',
            ',': 'preserve_with_space',
            '-': 'preserve_in_numbers',
            '/': 'preserve_in_measurements',
            '%': 'preserve_with_number',
            '±': 'preserve_in_measurements'
        }

    def _preserve_medical_terms(self, text: str) -> Tuple[str, List[str]]:
        """Preserve important medical terms and their context"""
        text = _get_plain_text(text)
        preserved_terms = []
        for term in self.preserve_terms:
            if term.lower() in text.lower():
                preserved_terms.append(term)
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                text = pattern.sub(lambda m: m.group().lower(), text)
        return text, preserved_terms

    def _handle_abbreviations(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Handle medical abbreviations consistently"""
        text = _get_plain_text(text)
        normalized = {}
        for abbr, full_form in self.medical_abbreviations.items():
            if abbr in text:
                normalized[abbr] = full_form
                text = text.replace(abbr.lower(), abbr)
                text = text.replace(abbr.upper(), abbr)
        return text, normalized

    def _process_measurements(self, text: str) -> Tuple[str, List[str]]:
        """Handle medical measurements consistently"""
        text = _get_plain_text(text)
        measurements_found = []
        for pattern in self.measurement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                measurement = match.group()
                measurements_found.append(measurement)
                text = text.replace(measurement, measurement.lower().replace(' ', ''))
        return text, measurements_found

    def _handle_special_characters(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Process special characters based on their context.
        This improved version uses a replacement function to check boundaries.
        """
        text = _get_plain_text(text)
        # Count the original occurrences of each special char.
        char_counts = {char: text.count(char) for char in self.special_chars_handling.keys()}

        # Process each special character according to its handling rule.
        for char, handling in self.special_chars_handling.items():
            # For 'preserve' rule, do nothing.
            if handling == 'preserve':
                continue

            # For the other rules, define a replacement function:
            def replacer(match):
                # Fetch the matched character.
                current = match.group()
                start, end = match.start(), match.end()
                # For hyphen ('-'), check if it lies between two digits or letters.
                if char == '-' and start > 0 and end < len(text):
                    prev_char, next_char = text[start - 1], text[end]
                    # If between digits or alphanumerics, do not replace.
                    if prev_char.isdigit() and next_char.isdigit():
                        return current
                    if prev_char.isalnum() and next_char.isalnum():
                        return current
                # Otherwise, add a space before and after.
                # (You might want to adjust this if you need to preserve some punctuation)
                return f' {current} '

            # Adjust regex patterns based on rules.
            if handling == 'preserve_with_space':
                # Replace every occurrence of the char with space around it, only if not already spaced.
                pattern = re.compile(re.escape(char))
                text = pattern.sub(replacer, text)

            elif handling in ['preserve_in_numbers', 'preserve_with_number']:
                # For these, only replace if the char is *not* between digits.
                # (Note: if it's between digits, leave it unmodified.)
                pattern = re.compile(re.escape(char))
                text = pattern.sub(lambda m: m.group() if (
                        m.start() > 0 and m.string[m.start() - 1].isdigit() and
                        m.end() < len(m.string) and m.string[m.end()].isdigit())
                else replacer(m), text)

            elif handling == 'preserve_in_measurements':
                # For measurements, try not to change the char when it is adjacent to digits or whitespace.
                pattern = re.compile(re.escape(char))
                text = pattern.sub(lambda m: m.group() if (
                        m.start() > 0 and (m.string[m.start() - 1].isdigit() or m.string[m.start() - 1].isspace())
                        and m.end() < len(m.string) and (m.string[m.end()].isdigit() or m.string[m.end()].isspace()))
                else replacer(m), text)

        return text, char_counts

    def _standardize_whitespace(self, text: str) -> str:
        """Standardize whitespace while preserving important formatting"""
        text = _get_plain_text(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.,!?])(?=[^\s.,!?])', r'\1 ', text)
        return text.strip()

    def _extract_numbers(self, text: str) -> List[str]:
        """Extract and standardize numerical values"""
        text = _get_plain_text(text)
        number_patterns = [r'\d+\.\d+', r'\d+', r'\d+\s*-\s*\d+', r'\d+\/\d+']
        numbers_found = []
        for pattern in number_patterns:
            matches = re.finditer(pattern, text)
            numbers_found.extend(match.group() for match in matches)
        return numbers_found

    def validate_text(self, text: str) -> bool:
        """Validate text for non-empty content"""
        text = _get_plain_text(text)
        if not text or len(text.strip()) == 0:
            return False
        return True

# ---------------------------
# Pipeline Classes
# ---------------------------
class PreprocessingPipeline:
    """Enhanced preprocessing pipeline for medical text"""
    def __init__(self):
        self.steps: List[TextPreprocessor] = []
        self.logger = get_logger(__name__)
        self.context = PreprocessingContext()
        self.max_length: Optional[int] = 9000

    def add_step(self, processor: TextPreprocessor) -> None:
        self.steps.append(processor)
        self.logger.info(f"Added preprocessing step: {processor.get_name()}")

    def set_max_length(self, length: int) -> None:
        self.max_length = length
        self.logger.info(f"Set maximum length to {length}")

    def add_preserved_terms(self, terms: Set[str]) -> None:
        self.context.preserved_terms.update(terms)

    def process(self, text: str, collect_stats: bool = True) -> Union[str, Tuple[str, PreprocessingStats]]:
        text = _get_plain_text(text)
        self.context.original_text = text
        processed_text = text
        stats = PreprocessingStats(
            original_length=len(text),
            processed_length=0,
            special_chars_removed=0,
            numbers_found=0,
            medical_terms_preserved=0
        )
        try:
            for step in self.steps:
                self.logger.debug(f"Applying {step.get_name()}")
                result = step.process(processed_text)
                # Always update processed_text with the output from the step
                processed_text = _get_plain_text(result)
            if self.max_length and len(processed_text) > self.max_length:
                processed_text = processed_text[:self.max_length]
                self.logger.info(f"Truncated text to {self.max_length} characters")
            if collect_stats:
                stats.processed_length = len(processed_text)
                return processed_text, stats
            return processed_text
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.process(text, collect_stats=False) for text in texts]

    def get_pipeline_info(self) -> Dict:
        return {
            'steps': [step.get_name() for step in self.steps],
            'max_length': self.max_length,
            'preserved_terms': len(self.context.preserved_terms)
        }

    def reset_context(self) -> None:
        self.context = PreprocessingContext()

# ---------------------------
# Preprocessing Step Implementations
# ---------------------------
class LengthNormalizer(TextPreprocessor):
    """Handles text length normalization based on EDA findings"""
    def __init__(self, max_length: int = 9000):
        super().__init__()
        self.max_length = max_length

    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        original_length = len(text)
        if original_length > self.max_length:
            truncated = text[:self.max_length]
            last_period = truncated.rfind('.')
            if last_period > 0:
                text = text[:last_period + 1]
            else:
                text = truncated
        return ProcessingResult(
            text=text,
            original_length=original_length,
            processed_length=len(text),
            special_chars_removed={},
            numbers_detected=[],
            medical_terms_found=[],
            abbreviations_normalized={}
        )

    def get_name(self) -> str:
        return "length_normalizer"

class MedicalTermPreprocessor(TextPreprocessor):
    """Handles medical terminology and abbreviations"""
    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        text, abbreviations = self._handle_abbreviations(text)
        text, preserved_terms = self._preserve_medical_terms(text)
        text, measurements = self._process_measurements(text)
        return ProcessingResult(
            text=text,
            original_length=len(text),
            processed_length=len(text),
            special_chars_removed={},
            numbers_detected=measurements,
            medical_terms_found=preserved_terms,
            abbreviations_normalized=abbreviations
        )

    def get_name(self) -> str:
        return "medical_term_processor"

class SpecialCharacterHandler(TextPreprocessor):
    """Handles special characters based on EDA findings"""
    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        text, char_counts = self._handle_special_characters(text)
        numbers = self._extract_numbers(text)
        return ProcessingResult(
            text=text,
            original_length=len(text),
            processed_length=len(text),
            special_chars_removed=char_counts,
            numbers_detected=numbers,
            medical_terms_found=[],
            abbreviations_normalized={}
        )

    def get_name(self) -> str:
        return "special_character_handler"

class DiseaseCategoryPreprocessor(TextPreprocessor):
    """Disease-specific preprocessing based on EDA findings"""
    def __init__(self, disease_category: str):
        super().__init__()
        self.disease_category = disease_category
        self.patterns = {
            'ALS': {
                'terms': ['amyotrophic lateral sclerosis', 'motor function', 'respiratory'],
                'measurements': [r'\d+\s*fvc', r'\d+\s*alsfrs'],
            },
            'OCD': {
                'terms': ['obsessive compulsive', 'anxiety', 'behavioral therapy'],
                'measurements': [r'\d+\s*ybocs', r'\d+\s*severity'],
            },
            'Parkinson': {
                'terms': ['motor symptoms', 'levodopa', 'dopaminergic'],
                'measurements': [r'\d+\s*updrs', r'\d+\s*hoehn'],
            },
            'Dementia': {
                'terms': ['cognitive function', 'memory', 'alzheimer'],
                'measurements': [r'\d+\s*mmse', r'\d+\s*cdr'],
            },
            'Scoliosis': {
                'terms': ['spinal curve', 'surgical correction', 'idiopathic'],
                'measurements': [r'\d+\s*degree', r'\d+\s*cobb'],
            }
        }

    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        disease_patterns = self.patterns.get(self.disease_category, {})
        preserved_terms = []
        for term in disease_patterns.get('terms', []):
            if term.lower() in text.lower():
                preserved_terms.append(term)
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                text = pattern.sub(lambda m: m.group().lower(), text)
        measurements_found = []
        for pattern in disease_patterns.get('measurements', []):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            measurements_found.extend(match.group() for match in matches)
        return ProcessingResult(
            text=text,
            original_length=len(text),
            processed_length=len(text),
            special_chars_removed={},
            numbers_detected=measurements_found,
            medical_terms_found=preserved_terms,
            abbreviations_normalized={}
        )

    def get_name(self) -> str:
        return f"{self.disease_category.lower()}_preprocessor"

class MedicalScorePreprocessor(TextPreprocessor):
    """Handles disease-specific medical scores based on EDA findings"""
    def __init__(self):
        super().__init__()
        self.score_patterns = {
            'ALS': {
                'FVC': r'\d+\s*%?\s*FVC',
                'ALSFRS': r'ALSFRS-?R?\s*(?:score)?\s*:?\s*\d+',
            },
            'OCD': {
                'YBOCS': r'Y-?BOCS\s*(?:score)?\s*:?\s*\d+',
                'Severity': r'severity\s*(?:score)?\s*:?\s*\d+',
            },
            'Parkinson': {
                'UPDRS': r'UPDRS\s*(?:score)?\s*:?\s*\d+',
                'Hoehn': r'Hoehn\s*(?:and)?\s*Yahr\s*(?:stage)?\s*:?\s*\d+',
            },
            'Dementia': {
                'MMSE': r'MMSE\s*(?:score)?\s*:?\s*\d+',
                'CDR': r'CDR\s*(?:score)?\s*:?\s*[\d\.]+',
            },
            'Scoliosis': {
                'Cobb': r'Cobb\s*(?:angle)?\s*:?\s*\d+\s*°?',
                'Degree': r'\d+\s*(?:degree[s]?)?\s*(?:curve|angle)',
            }
        }
    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        scores_found = []
        for disease, patterns in self.score_patterns.items():
            for score_type, pattern in patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    score = match.group()
                    scores_found.append(f"{disease}_{score_type}: {score}")
                    standardized = re.sub(r'\s+', ' ', score).strip()
                    text = text.replace(score, standardized)
        return ProcessingResult(
            text=text,
            original_length=len(text),
            processed_length=len(text),
            special_chars_removed={},
            numbers_detected=scores_found,
            medical_terms_found=[],
            abbreviations_normalized={}
        )
    def get_name(self) -> str:
        return "medical_score_processor"

class MedicalAbbreviationNormalizer(TextPreprocessor):
    """Specialized processor for handling medical abbreviations based on EDA findings"""
    def __init__(self):
        super().__init__()
        self.disease_abbreviations = {
            'ALS': 'amyotrophic lateral sclerosis',
            'OCD': 'obsessive compulsive disorder',
            'PD': "parkinson's disease",
            'AD': "alzheimer's disease",
        }
        self.measurement_abbreviations = {
            'mg': 'milligrams',
            'kg': 'kilograms',
            'ml': 'milliliters',
            'mm': 'millimeters',
            'cm': 'centimeters',
        }
        self.score_abbreviations = {
            'ALSFRS-R': 'ALS Functional Rating Scale Revised',
            'FVC': 'Forced Vital Capacity',
            'YBOCS': 'Yale-Brown Obsessive Compulsive Scale',
            'UPDRS': 'Unified Parkinson Disease Rating Scale',
            'MMSE': 'Mini-Mental State Examination',
            'CDR': 'Clinical Dementia Rating',
        }
        self.procedure_abbreviations = {
            'MRI': 'Magnetic Resonance Imaging',
            'CT': 'Computed Tomography',
            'EEG': 'Electroencephalogram',
            'EMG': 'Electromyography',
        }
        self.medical_term_abbreviations = {
            'tx': 'treatment',
            'dx': 'diagnosis',
            'hx': 'history',
            'pts': 'patients',
            'sig': 'significant',
        }
    def _create_abbreviation_pattern(self) -> str:
        all_abbrevs = set()
        all_abbrevs.update(self.disease_abbreviations.keys())
        all_abbrevs.update(self.score_abbreviations.keys())
        all_abbrevs.update(self.procedure_abbreviations.keys())
        all_abbrevs.update(self.medical_term_abbreviations.keys())
        sorted_abbrevs = sorted(all_abbrevs, key=len, reverse=True)
        pattern = '|'.join(re.escape(abbr) for abbr in sorted_abbrevs)
        return f'\\b({pattern})\\b'
    def _detect_context(self, text: str, abbr: str, window: int = 50) -> bool:
        text = _get_plain_text(text)
        abbr_idx = text.find(abbr)
        if abbr_idx == -1:
            return True
        start = max(0, abbr_idx - window)
        end = min(len(text), abbr_idx + len(abbr) + window)
        context = text[start:end].lower()
        full_form = self._get_full_form(abbr).lower()
        if full_form in context:
            return False
        return True
    def _get_full_form(self, abbr: str) -> str:
        for abbrev_dict in [self.disease_abbreviations, self.score_abbreviations,
                            self.procedure_abbreviations, self.medical_term_abbreviations]:
            if abbr in abbrev_dict:
                return abbrev_dict[abbr]
        return abbr
    def _normalize_measurements(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        text = _get_plain_text(text)
        normalized = text
        found_measurements = defaultdict(list)
        for abbr, full_form in self.measurement_abbreviations.items():
            pattern = f'(\\d+(?:\\.\\d+)?)\\s*{abbr}\\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                found_measurements[abbr].append(value)
                normalized = normalized.replace(match.group(), f"{value} {abbr}")
        return normalized, dict(found_measurements)
    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        original_text = text
        abbreviations_found = {}
        text, measurements = self._normalize_measurements(text)
        pattern = self._create_abbreviation_pattern()
        for match in re.finditer(pattern, text, re.IGNORECASE):
            abbr = match.group()
            if not self._detect_context(text, abbr):
                continue
            full_form = self._get_full_form(abbr)
            abbreviations_found[abbr] = full_form
            first_idx = text.find(abbr)
            if first_idx >= 0:
                replacement = f"{full_form} ({abbr})"
                text = text[:first_idx] + replacement + text[first_idx + len(abbr):]
            text = text.replace(abbr, abbr)
        for abbr, values in measurements.items():
            abbreviations_found[abbr] = f"{self.measurement_abbreviations[abbr]} (found with values: {', '.join(values)})"
        return ProcessingResult(
            text=text,
            original_length=len(original_text),
            processed_length=len(text),
            special_chars_removed={},
            numbers_detected=[],
            medical_terms_found=[],
            abbreviations_normalized=abbreviations_found
        )
    def get_name(self) -> str:
        return "medical_abbreviation_normalizer"

class MedicalTermStandardizer(TextPreprocessor):
    """Standardizes medical terminology based on EDA findings and disease categories"""
    def __init__(self):
        super().__init__()
        self.disease_terms = {
            'ALS': {
                'motor neuron disease': 'amyotrophic lateral sclerosis',
                'muscle weakness': 'muscular weakness',
                'breathing difficulty': 'respiratory difficulty',
                'bulbar symptoms': 'bulbar dysfunction',
                'als disease': 'ALS',
                'motor function loss': 'motor deficit',
            },
            'OCD': {
                'obsessive thoughts': 'obsessions',
                'compulsive behaviors': 'compulsions',
                'intrusive thoughts': 'obsessive thoughts',
                'ritualistic behavior': 'compulsive behavior',
                'anxiety symptoms': 'anxiety manifestations',
                'ocd symptoms': 'OCD manifestations',
            },
            'Parkinson': {
                'tremors': 'tremor',
                'shaking': 'tremor',
                'movement problems': 'motor symptoms',
                'balance problems': 'postural instability',
                'stiffness': 'rigidity',
                'slow movement': 'bradykinesia',
            },
            'Dementia': {
                'memory loss': 'cognitive decline',
                'confusion': 'cognitive impairment',
                'behavioral changes': 'behavioral symptoms',
                'memory problems': 'memory impairment',
                'thinking problems': 'cognitive dysfunction',
                'mental decline': 'cognitive deterioration',
            },
            'Scoliosis': {
                'spine curvature': 'spinal curvature',
                'curved spine': 'spinal curvature',
                'back deformity': 'spinal deformity',
                'spine deviation': 'spinal deviation',
                'backbone curve': 'spinal curve',
                'spinal bend': 'spinal curvature',
            }
        }
        self.common_terms = {
            'symptom': 'manifestation',
            'side effect': 'adverse effect',
            'adverse reaction': 'adverse effect',
            'negative effect': 'adverse effect',
            'medicine': 'medication',
            'drug': 'medication',
            'therapy': 'treatment',
            'therapeutic': 'treatment',
            'test': 'assessment',
            'examination': 'assessment',
            'evaluation': 'assessment',
            'checkup': 'examination',
            'condition': 'status',
            'state': 'status',
            'progression': 'disease progression',
            'improvement': 'clinical improvement',
            'deterioration': 'clinical deterioration'
        }
        self.measurement_terms = {
            'monthly': 'per month',
            'weekly': 'per week',
            'daily': 'per day',
            'yearly': 'per year',
            'twice daily': 'BID',
            'three times daily': 'TID',
            'four times daily': 'QID',
            'once daily': 'QD',
            'milliliters': 'ml',
            'milligrams': 'mg',
            'kilograms': 'kg',
            'centimeters': 'cm'
        }
    def _create_term_pattern(self, terms: Dict[str, str]) -> str:
        sorted_terms = sorted(terms.keys(), key=len, reverse=True)
        pattern = '|'.join(re.escape(term) for term in sorted_terms)
        return f'\\b({pattern})\\b'
    def _standardize_disease_terms(self, text: str, disease: str) -> Tuple[str, List[str]]:
        text = _get_plain_text(text)
        standardized_terms = []
        if disease in self.disease_terms:
            pattern = self._create_term_pattern(self.disease_terms[disease])
            def replace_term(match):
                term = match.group(0)
                standardized = self.disease_terms[disease].get(term.lower(), term)
                standardized_terms.append(f"{term} → {standardized}")
                return standardized
            text = re.sub(pattern, replace_term, text, flags=re.IGNORECASE)
        return text, standardized_terms
    def _standardize_common_terms(self, text: str) -> Tuple[str, List[str]]:
        text = _get_plain_text(text)
        standardized_terms = []
        pattern = self._create_term_pattern(self.common_terms)
        def replace_term(match):
            term = match.group(0)
            standardized = self.common_terms.get(term.lower(), term)
            standardized_terms.append(f"{term} → {standardized}")
            return standardized
        text = re.sub(pattern, replace_term, text, flags=re.IGNORECASE)
        return text, standardized_terms
    def _standardize_measurements(self, text: str) -> Tuple[str, List[str]]:
        text = _get_plain_text(text)
        standardized_terms = []
        pattern = self._create_term_pattern(self.measurement_terms)
        def replace_term(match):
            term = match.group(0)
            standardized = self.measurement_terms.get(term.lower(), term)
            standardized_terms.append(f"{term} → {standardized}")
            return standardized
        text = re.sub(pattern, replace_term, text, flags=re.IGNORECASE)
        return text, standardized_terms
    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        original_text = text
        all_standardized_terms = []
        for disease in self.disease_terms.keys():
            text, disease_terms = self._standardize_disease_terms(text, disease)
            all_standardized_terms.extend(disease_terms)
        text, common_terms = self._standardize_common_terms(text)
        all_standardized_terms.extend(common_terms)
        text, measurement_terms = self._standardize_measurements(text)
        all_standardized_terms.extend(measurement_terms)
        return ProcessingResult(
            text=text,
            original_length=len(original_text),
            processed_length=len(text),
            special_chars_removed={},
            numbers_detected=[],
            medical_terms_found=all_standardized_terms,
            abbreviations_normalized={}
        )
    def get_name(self) -> str:
        return "medical_term_standardizer"

class DomainSpecificStopwordHandler(TextPreprocessor):
    """
    Handles stopwords specifically for medical trial descriptions,
    preserving domain-critical terms based on EDA findings
    """
    def __init__(self):
        super().__init__()
        self.standard_stopwords = set(nltk_stopwords.words('english'))
        self.preserve_terms = {
            'disease', 'disorder', 'syndrome', 'condition', 'symptoms',
            'treatment', 'therapy', 'drug', 'dose', 'trial',
            'high', 'low', 'increase', 'decrease', 'level',
            'daily', 'weekly', 'monthly', 'duration',
            'positive', 'negative', 'normal', 'abnormal'
        }
        self.domain_stopwords = {
            'study', 'research', 'clinical', 'medical', 'patient',
            'subject', 'participant', 'investigator', 'physician',
            'protocol', 'procedure', 'visit', 'center', 'facility',
            'including', 'included', 'excluded', 'following',
            'based', 'related', 'associated', 'regarding'
        }
        self.disease_specific_terms = {
            'ALS': {'respiratory', 'bulbar', 'motor', 'muscle', 'weakness',
                    'progression', 'function', 'strength', 'vital', 'capacity'},
            'OCD': {'compulsive', 'obsessive', 'anxiety', 'behavior',
                    'ritual', 'intrusive', 'thoughts', 'severity'},
            'Parkinson': {'motor', 'tremor', 'rigidity', 'movement', 'balance',
                          'gait', 'dopamine', 'dyskinesia'},
            'Dementia': {'cognitive', 'memory', 'mental', 'behavioral',
                         'function', 'decline', 'caregiver', 'activities'},
            'Scoliosis': {'curve', 'spine', 'degree', 'correction', 'fusion',
                          'thoracic', 'lumbar', 'surgical'}
        }
    def _is_measurement_or_value(self, token: str) -> bool:
        measurement_patterns = [
            r'\d+', r'\d+\.\d+', r'\d+%', r'\d+mg', r'\d+kg', r'\d+ml', r'\d+cm'
        ]
        return any(re.match(pattern, token) for pattern in measurement_patterns)
    def _is_medical_abbreviation(self, token: str) -> bool:
        abbreviation_patterns = [
            r'^[A-Z]{2,}$',
            r'^[A-Z]{2,}-[A-Z]$',
            r'^[A-Z][a-z]+[A-Z]+$'
        ]
        return any(re.match(pattern, token) for pattern in abbreviation_patterns)
    def _should_preserve(self, token: str, disease_category: Optional[str] = None) -> bool:
        if self._is_measurement_or_value(token) or self._is_medical_abbreviation(token):
            return True
        token_lower = token.lower()
        if token_lower in self.preserve_terms:
            return True
        if disease_category and token_lower in self.disease_specific_terms.get(disease_category, set()):
            return True
        return False
    def process(self, text: str, disease_category: Optional[str] = None) -> ProcessingResult:
        text = _get_plain_text(text)
        original_text = text
        preserved_terms = []
        removed_terms = []
        tokens = text.split()
        processed_tokens = []
        for token in tokens:
            if self._should_preserve(token, disease_category):
                processed_tokens.append(token)
                preserved_terms.append(token)
            elif token.lower() in self.domain_stopwords or token.lower() in self.standard_stopwords:
                removed_terms.append(token)
            else:
                processed_tokens.append(token)
        processed_text = ' '.join(processed_tokens)
        return ProcessingResult(
            text=processed_text,
            original_length=len(original_text),
            processed_length=len(processed_text),
            special_chars_removed={},
            numbers_detected=[],
            medical_terms_found=preserved_terms,
            abbreviations_normalized={'removed_stopwords': removed_terms}
        )
    def get_name(self) -> str:
        return "domain_specific_stopword_handler"

class SpecialCharacterCleaner(TextPreprocessor):
    """
    Specialized cleaner for handling special characters in medical texts
    based on EDA findings
    """
    def __init__(self):
        super().__init__()
        self.char_categories = {
            'preserve_always': {'.', ',', '(', ')', '%', '±', '-', '/'},
            'preserve_with_numbers': {'.', '-', '/', '%', '±'},
            'preserve_in_compounds': {'-', '/'},
            'replace_with_space': {';', ':', '|', '\\', '[', ']', '{', '}', '_', '=', '+', '*'}
        }
        self.medical_patterns = {
            'measurements': [
                r'\d+(?:\.\d+)?\s*(?:mg|kg|ml|cm|mm)',
                r'\d+(?:\.\d+)?\s*%',
                r'\d+(?:\.\d+)?\s*±\s*\d+(?:\.\d+)?'
            ],
            'ranges': [
                r'\d+\s*-\s*\d+',
                r'\d+\s*/\s*\d+'
            ],
            'compound_terms': [
                r'\w+(?:-\w+)+',
                r'\w+(?:/\w+)+'
            ],
            'statistical_values': [
                r'p\s*[<>]\s*0\.\d+',
                r'[\+-]\s*\d+(?:\.\d+)?'
            ]
        }

    def _identify_protected_spans(self, text: str) -> List[Tuple[int, int, str]]:
        text = _get_plain_text(text)
        protected_spans = []
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    protected_spans.append((match.start(), match.end(), category))
        return sorted(protected_spans, key=lambda x: x[0])

    def _clean_special_chars(self, text: str, protected_spans: List[Tuple[int, int, str]]) -> Tuple[str, Dict[str, int]]:
        text = _get_plain_text(text)
        cleaned_text = ""
        last_end = 0
        char_counts = defaultdict(int)
        for start, end, span_type in protected_spans:
            segment = text[last_end:start]
            cleaned_segment = self._process_segment(segment, char_counts)
            cleaned_text += cleaned_segment
            cleaned_text += text[start:end]
            last_end = end
        if last_end < len(text):
            final_segment = text[last_end:]
            cleaned_text += self._process_segment(final_segment, char_counts)
        return cleaned_text, dict(char_counts)

    def _process_segment(self, segment: str, char_counts: Dict[str, int]) -> str:
        segment = _get_plain_text(segment)
        # Instead of a fixed process, use our unified replacement function for "replace_with_space" characters.
        for ch in self.char_categories['replace_with_space']:
            # Count occurrences first.
            char_counts[ch] += segment.count(ch)
            # Add spaces around the character if not already spaced.
            segment = re.sub(rf'(?<!\s){re.escape(ch)}(?!\s)', f' {ch} ', segment)
        # Collapse multiple spaces.
        segment = re.sub(r'\s+', ' ', segment)
        return segment

    def _standardize_measurements(self, text: str) -> str:
        text = _get_plain_text(text)
        standardized = text
        for pattern in self.medical_patterns['measurements']:
            standardized = re.sub(
                pattern,
                lambda m: m.group().replace(' ', ''),
                standardized
            )
        for pattern in self.medical_patterns['ranges']:
            standardized = re.sub(
                pattern,
                lambda m: m.group().replace(' ', ''),
                standardized
            )
        return standardized

    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        original_text = text
        protected_spans = self._identify_protected_spans(text)
        cleaned_text, char_counts = self._clean_special_chars(text, protected_spans)
        final_text = self._standardize_measurements(cleaned_text)
        preserved_patterns = [f"{span_type}: {text[start:end]}" for start, end, span_type in protected_spans]
        return ProcessingResult(
            text=final_text,
            original_length=len(original_text),
            processed_length=len(final_text),
            special_chars_removed=char_counts,
            numbers_detected=[],
            medical_terms_found=preserved_patterns,
            abbreviations_normalized={}
        )

    def get_name(self) -> str:
        return "special_character_cleaner"

class LowercasePreprocessor(TextPreprocessor):
    """Convert text to lowercase"""
    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        lowercased_text = text.lower()
        return ProcessingResult(
            text=lowercased_text,
            original_length=len(text),
            processed_length=len(lowercased_text),
            special_chars_removed={},
            numbers_detected=[],
            medical_terms_found=[],
            abbreviations_normalized={}
        )
    def get_name(self) -> str:
        return "lowercase"

class WhitespaceNormalizer(TextPreprocessor):
    """Normalize whitespace in text"""
    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        normalized_text = ' '.join(text.split())
        return ProcessingResult(
            text=normalized_text,
            original_length=len(text),
            processed_length=len(normalized_text),
            special_chars_removed={},
            numbers_detected=[],
            medical_terms_found=[],
            abbreviations_normalized={}
        )
    def get_name(self) -> str:
        return "whitespace_normalizer"

class MedicalTokenizer(TextPreprocessor):
    """Specialized tokenizer for medical text that preserves important terms"""
    def __init__(self):
        super().__init__()
        self.term_boundaries = {
            'measurements': r'\d+\s*(?:mg|kg|ml|cm|mm)',
            'scores': r'\d+\s*(?:points?|score)',
            'percentages': r'\d+\s*%',
            'ranges': r'\d+\s*-\s*\d+',
        }
    def process(self, text: str) -> ProcessingResult:
        text = _get_plain_text(text)
        preserved = {}
        for term_type, pattern in self.term_boundaries.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for i, match in enumerate(matches):
                placeholder = f"__{term_type}_{i}__"
                preserved[placeholder] = match.group()
                text = text.replace(match.group(), placeholder)
        tokens = text.split()
        processed_text = ' '.join(tokens)
        for placeholder, original in preserved.items():
            processed_text = processed_text.replace(placeholder, original)
        return ProcessingResult(
            text=processed_text,
            original_length=len(text),
            processed_length=len(processed_text),
            special_chars_removed={},
            numbers_detected=list(preserved.values()),
            medical_terms_found=[],
            abbreviations_normalized={}
        )
    def get_name(self) -> str:
        return "medical_tokenizer"

# ---------------------------
# Pipeline Builder
# ---------------------------
def create_ordered_medical_pipeline(
        disease_category: Optional[str] = None,
        config: Optional[Dict] = None
) -> PreprocessingPipeline:
    """Create a properly ordered preprocessing pipeline with improved text handling."""
    pipeline = PreprocessingPipeline()
    default_config = {
        'max_length': 9000,
        'preserve_case': True,
        'include_scores': True,
        'standardize_terms': True,
        'handle_stopwords': False,
        'preserve_measurements': True
    }
    config = {**default_config, **(config or {})}

    class ImprovedSpecialCharacterCleaner(SpecialCharacterCleaner):
        def _standardize_measurements(self, text: str) -> str:
            standardized = text
            for pattern in self.medical_patterns['measurements']:
                standardized = re.sub(
                    pattern,
                    lambda m: re.sub(r'(\d+)\s*([a-zA-Z%]+)', r'\1 \2', m.group()),
                    standardized
                )
            standardized = re.sub(
                r'(\d+)\s*-\s*(\d+)',
                r'\1 - \2',
                standardized
            )
            standardized = re.sub(r'\s*([±])\s*', r' \1 ', standardized)
            return standardized

    class ImprovedMedicalTokenizer(MedicalTokenizer):
        def process(self, text: str) -> ProcessingResult:
            text = _get_plain_text(text)
            preserved = {}
            for term_type, pattern in self.term_boundaries.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for i, match in enumerate(matches):
                    placeholder = f"__{term_type}_{i}__"
                    preserved[placeholder] = match.group()
                    text = text.replace(match.group(), placeholder)
            text = re.sub(r'([A-Z]-[A-Z])', r' \1 ', text)
            text = re.sub(r'([A-Z]+)([a-z])', r'\1 \2', text)
            text = ' '.join(text.split())
            for placeholder, original in preserved.items():
                text = text.replace(placeholder, original)
            return ProcessingResult(
                text=text,
                original_length=len(text),
                processed_length=len(text),
                special_chars_removed={},
                numbers_detected=list(preserved.values()),
                medical_terms_found=[],
                abbreviations_normalized={}
            )

    # 1. Initial Text Cleanup
    pipeline.add_step(WhitespaceNormalizer())
    # 2. Length Normalization
    pipeline.add_step(LengthNormalizer(max_length=config['max_length']))
    # 3. Improved Special Character Processing
    pipeline.add_step(ImprovedSpecialCharacterCleaner())
    # 4. Medical Content Processing
    if config['include_scores']:
        pipeline.add_step(MedicalScorePreprocessor())
    pipeline.add_step(MedicalAbbreviationNormalizer())
    # 5. Term Standardization
    if config['standardize_terms']:
        pipeline.add_step(MedicalTermStandardizer())
    # 6. Disease-Specific Processing
    if disease_category:
        pipeline.add_step(DiseaseCategoryPreprocessor(disease_category))
    # 7. Stop Words
    if config['handle_stopwords']:
        pipeline.add_step(DomainSpecificStopwordHandler())
    # 8. Lowercase (if not preserving case)
    if not config.get('preserve_case', True):
        pipeline.add_step(LowercasePreprocessor())
    # 9. Final Tokenization with Improved Handling
    pipeline.add_step(ImprovedMedicalTokenizer())
    return pipeline