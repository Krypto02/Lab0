# MLOps Data Preprocessing Toolkit

A comprehensive Python toolkit for data preprocessing operations following ML Lifecycle best practices and principles. This project provides robust data cleaning, numeric transformations, text processing, and data structure operations designed for machine learning pipelines.

##  Overview

This toolkit implements industry-standard data preprocessing functions with built-in validation, logging, and quality checks. It's designed to support the entire ML lifecycle from data ingestion to model preparation, with emphasis on reproducibility, monitoring, and data quality assurance.

##  Features

### Data Cleaning
- **Missing Value Handling**: Remove or fill missing values (None, empty strings, NaN)
- **Duplicate Removal**: Eliminate duplicates while preserving order
- **Data Quality Validation**: Comprehensive quality metrics and validation

### Numeric Transformations
- **Min-Max Normalization**: Scale values to custom ranges
- **Z-Score Standardization**: Standardize to mean=0, std=1
- **Value Clipping**: Constrain values within specified bounds
- **Type Conversion**: Convert strings to integers with error handling
- **Log Transformation**: Apply logarithmic transformations to positive values

### Text Processing
- **Tokenization**: Clean text tokenization with alphanumeric filtering
- **Punctuation Removal**: Strip punctuation while preserving structure
- **Stop Word Removal**: Case-insensitive stop word filtering

### Data Structure Operations
- **List Flattening**: Convert nested lists to flat structures
- **Shuffling**: Reproducible random shuffling with seed support
- **Unique Value Extraction**: Preserve original order during deduplication

##  Requirements

- Python 3.8+
- Dependencies:
  - `click` - CLI framework
  - `pytest` - Testing framework

##  Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Mlops
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install click pytest
```

##  Usage

### As a Python Library

```python
from src import preprocessing

# Data cleaning
data = [1, None, 3, "", 5]
cleaned = preprocessing.remove_missing(data)
print(cleaned)  # [1, 3, 5]

# Normalization
values = [1, 2, 3, 4, 5]
normalized = preprocessing.normalize_minmax(values, 0, 1)
print(normalized)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Data quality validation
quality = preprocessing.validate_data_quality(data, missing_threshold=0.3)
print(f"Quality Score: {quality['quality_score']}")
print(f"Missing Ratio: {quality['missing_ratio']}")

# Text processing
text = "Hello, World! This is a TEST."
tokenized = preprocessing.tokenize_text(text)
print(tokenized)  # "hello world this is a test"
```

### Command Line Interface (CLI)

The toolkit includes a comprehensive CLI with organized command groups:

#### Data Cleaning Operations

```bash
# Remove missing values
python -m src.cli clean remove-missing "1,None,3,," --quality-check

# Fill missing values
python -m src.cli clean fill-missing "1,,3,None" --fill-value 0 --quality-check
```

#### Numeric Operations

```bash
# Normalize data
python -m src.cli numeric normalize "1,2,3,4,5" --min-val 0 --max-val 1 --report

# Standardize using z-score
python -m src.cli numeric standardize "1,2,3,4,5" --report

# Clip values
python -m src.cli numeric clip "1,2,3,4,5" --min-val 2 --max-val 4 --report

# Convert to integers
python -m src.cli numeric to-int "1,2.5,abc,4" --report

# Log transformation
python -m src.cli numeric log-transform "1,2,3,4,5" --report
```

#### Text Operations

```bash
# Tokenize text
python -m src.cli text tokenize "Hello, World! How are you?" --report

# Remove punctuation
python -m src.cli text remove-punctuation "Hello, World!" --report

# Remove stop words
python -m src.cli text remove-stopwords "this is a test" --stopwords "is,a" --report
```

#### Structure Operations

```bash
# Shuffle list
python -m src.cli struct shuffle "1,2,3,4,5" --seed 42 --report

# Flatten nested lists
python -m src.cli struct flatten "[[1,2],[3,4]]" --report

# Get unique values
python -m src.cli struct unique "1,2,2,3,3,4" --report
```

#### Verbose Logging

Add `--verbose` or `-v` flag to any command for detailed logging:

```bash
python -m src.cli -v numeric normalize "1,2,3,4,5"
```

##  Testing

The project includes comprehensive unit tests following ML Lifecycle testing principles:

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Files

```bash
# Test preprocessing functions
pytest tests/test_preprocessing.py

# Test CLI commands
pytest tests/test_cli.py
```

### Run with Verbose Output

```bash
pytest -v tests/
```

### Run with Coverage

```bash
pytest --cov=src tests/
```

### Test Categories

- **Data Quality Tests**: Validation and quality metrics
- **Data Cleaning Tests**: Missing values, duplicates, edge cases
- **Numeric Transformations Tests**: Normalization, standardization, clipping
- **Text Processing Tests**: Tokenization, cleaning, stop words
- **Data Structure Tests**: Flattening, shuffling, uniqueness
- **Edge Cases**: Empty inputs, single elements, large datasets
- **Error Handling**: Type errors, invalid ranges, malformed data

##  Project Structure

```
Mlops/
 src/
    __init__.py
    cli.py              # Command-line interface
    preprocessing.py    # Core preprocessing functions
 tests/
    __init__.py
    test_cli.py         # CLI tests
    test_preprocessing.py  # Unit tests
 .venv/                  # Virtual environment
 .pytest_cache/          # Pytest cache
 README.md               # This file
```

##  ML Lifecycle Features

### Data Quality Monitoring

```python
quality_report = preprocessing.validate_data_quality(
    data, 
    missing_threshold=0.3, 
    check_duplicates=True
)

print(f"Total Count: {quality_report['total_count']}")
print(f"Missing Ratio: {quality_report['missing_ratio']:.2%}")
print(f"Duplicate Count: {quality_report['duplicate_count']}")
print(f"Quality Score: {quality_report['quality_score']:.2f}")
print(f"Validation Errors: {quality_report['validation_errors']}")
```

### Pipeline Reporting

```python
report = preprocessing.create_data_pipeline_report(
    processed_data, 
    operation="remove_missing", 
    original_count=100
)

print(f"Operation: {report['operation']}")
print(f"Data Retention: {report['data_retention']:.2%}")
print(f"Quality Metrics: {report['quality_metrics']}")
```

### Reproducibility

All operations support reproducible results:

```python
# Reproducible shuffling
shuffled1 = preprocessing.shuffle_list(data, seed=42)
shuffled2 = preprocessing.shuffle_list(data, seed=42)
assert shuffled1 == shuffled2  # Guaranteed same result
```

##  Advanced Usage

### Chaining Operations

```python
from src import preprocessing

# Complete preprocessing pipeline
data = [1, None, 2, "", 3, 2, 4, 5]

# Step 1: Remove missing values
cleaned = preprocessing.remove_missing(data)

# Step 2: Remove duplicates
unique = preprocessing.remove_duplicates(cleaned)

# Step 3: Normalize
normalized = preprocessing.normalize_minmax(unique, 0, 1)

# Step 4: Validate quality
quality = preprocessing.validate_data_quality(normalized)

print(f"Pipeline completed with quality score: {quality['quality_score']}")
```

### Custom Validation Thresholds

```python
# Strict quality requirements
strict_quality = preprocessing.validate_data_quality(
    data,
    missing_threshold=0.1,  # Allow only 10% missing
    check_duplicates=True
)

if not strict_quality['is_valid']:
    print("Data does not meet quality standards")
    for error in strict_quality['validation_errors']:
        print(f"  - {error}")
```

##  Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

##  Best Practices

### Error Handling

All functions include proper error handling:
- Type validation for inputs
- Range validation for numeric operations
- Graceful handling of edge cases
- Informative error messages

### Logging

Comprehensive logging throughout:
```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# All operations automatically log
preprocessing.normalize_minmax([1, 2, 3, 4, 5])
# INFO: Normalizing 5 values to range [0.0, 1.0]
```

### Testing Strategy

- Use pytest fixtures for reusable test data
- Parametrized tests for multiple input combinations
- Edge case coverage (empty, single element, large datasets)
- Mock logging for validation
- Test both success and failure paths

##  License

This project is created for educational purposes as part of an MLOps course.

##  Authors

- Wassim

##  Acknowledgments

- Built following ML Lifecycle theory and best practices
- Implements industry-standard preprocessing techniques
- Designed for production ML pipelines

##  Support

For issues, questions, or contributions, please open an issue in the repository.

---

**Happy Preprocessing! **
