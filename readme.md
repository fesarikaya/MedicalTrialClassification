# Medical Trial Classification System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask Version](https://img.shields.io/badge/flask-3.0.2-green.svg)](https://flask.palletsprojects.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

The Medical Trial Classification System is an automated machine learning solution that classifies medical trial descriptions into five disease categories. Currently in partial implementation status, the system aims to reduce the manual effort required in categorizing medical trials.

### Current Implementation Status

✅ Core preprocessing pipeline  
✅ Basic model implementation  
✅ Initial API setup  
✅ Basic testing framework  
❌ Complete unit test coverage  
❌ Advanced preprocessing features  
❌ Model optimization  
❌ Full system integration testing  

### Disease Categories
- Amyotrophic Lateral Sclerosis (ALS)
- Obsessive Compulsive Disorder (OCD)
- Parkinson's Disease
- Dementia
- Scoliosis

## Project Structure

```
root/
├── data/                  # Data storage and processing
├── docs/                  # Project documentation
├── logs/                  # Application logs
├── notebooks/            # Analysis notebooks
├── scripts/              # Utility scripts
├── src/                  # Source code
└── tests/                # Test files
```

### Key Components

- `src/preprocessing/`: Text preprocessing pipeline
- `src/models/`: Model implementation and training
- `src/data/`: Data processing and pipeline
- `src/utils/`: Utility functions and logging
- `tests/`: Test implementations

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd medical-trial-classification
```

2. Run the environment setup:
```bash
python environment_setup.py
```

### Requirements

- Python 3.8+
- 8GB+ RAM recommended
- Disk space for model storage
- Internet connection for package installation

### Key Dependencies

- Flask==3.0.2
- pandas==2.2.0
- scikit-learn==1.4.0
- nltk==3.8.1
- spacy==3.7.2
- pytest==8.0.0

Full dependencies are listed in `requirements.txt`.

## Current Performance

### Model Performance
- Best performer: Bagging Classifier
  - Accuracy: 50.0%
  - F1 Score: 0.490

### Known Issues

1. Preprocessing Pipeline
   - Performance issues in current implementation
   - Medical term standardization needs improvement
   - Special character handling requires optimization

2. Model Performance
   - Lower than target accuracy due to preprocessing issues
   - Feature engineering needs enhancement
   - Model tuning incomplete

## Usage

### API Endpoints

1. Prediction Endpoint:
```bash
POST /predict
Content-Type: application/json
{
    "description": "Medical trial description text"
}
```

2. Health Check:
```bash
GET /health
```

### Testing

Basic tests are implemented in the `tests/` directory:
- `API_test.py`: API endpoint testing
- `model_evaluation_test.py`: Basic model evaluation
- Latest test results available in `prediction_test_results.json`

## Future Work

1. Preprocessing Enhancements
   - Optimize medical term handling
   - Improve text normalization
   - Enhance special character processing

2. Model Optimization
   - Implement advanced feature engineering
   - Optimize model parameters
   - Enhance ensemble methods

3. Testing Completion
   - Implement comprehensive unit tests
   - Add integration tests
   - Complete performance testing

## Important Notes

- System is currently in partial implementation status
- Use with caution and verify all predictions
- Current accuracy is limited
- Future updates will address known issues

## Development Status

The project is currently incomplete due to deadline constraints. Key pending items include:
- Complete unit test coverage
- Advanced preprocessing features
- Model optimization
- Full system integration testing

## License

[License information to be added]

## Warning

⚠️ This system is currently in partial implementation status with known preprocessing issues affecting model performance. Use as an assistance tool only and verify all predictions manually.