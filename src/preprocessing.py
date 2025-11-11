"""
Data preprocessing functions for various data types.
"""

import logging
import math
import random
import re
import statistics
from typing import Any, List, Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def remove_missing(values: List[Any]) -> List[Any]:
    """
    Remove missing values from a list following ML Lifecycle data quality principles.

    This function implements robust missing value detection as part of the data quality
    assurance layer in the ML pipeline, ensuring clean data flows to downstream processes.

    Args:
        values: List containing potential missing values (None, '', nan)

    Returns:
        List with missing values removed
    """
    if not isinstance(values, list):
        raise TypeError("Input must be a list")

    logger.info("Removing missing values from list of length %d", len(values))

    cleaned = [
        v
        for v in values
        if v is not None and v != "" and not (isinstance(v, float) and math.isnan(v))
    ]

    removed_count = len(values) - len(cleaned)
    if removed_count > 0:
        logger.info("Removed %d missing values", removed_count)

    return cleaned


def fill_missing(values: List[Any], fill_value: Any = 0) -> List[Any]:
    """
    Fill missing values with a specified value.

    Args:
        values: List containing potential missing values
        fill_value: Value to replace missing values with (default: 0)

    Returns:
        List with missing values replaced
    """
    result = []
    for v in values:
        if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
            result.append(fill_value)
        else:
            result.append(v)
    return result


def remove_duplicates(values: List[Any]) -> List[Any]:
    """
    Remove duplicate values while preserving order.

    Args:
        values: List of values that may contain duplicates

    Returns:
        List of unique values in original order

    """
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_minmax(
    values: List[Union[int, float]], new_min: float = 0.0, new_max: float = 1.0
) -> List[float]:
    """
    Normalize numerical values using min-max method with ML pipeline validation.

    Implements feature scaling as per ML Lifecycle theory, ensuring consistent
    numerical ranges for downstream ML algorithms. Includes validation and
    edge case handling for robust data pipeline operation.

    Args:
        values: List of numerical values
        new_min: New minimum value (default: 0.0)
        new_max: New maximum value (default: 1.0)

    Returns:
        List of normalized values
    """
    if not isinstance(values, list):
        raise TypeError("Values must be a list")

    if not values:
        logger.warning("Empty list provided for normalization")
        return []

    if new_min >= new_max:
        raise ValueError("new_min must be less than new_max")

    if not all(isinstance(v, (int, float)) and not math.isnan(v) for v in values):
        raise ValueError("All values must be numeric and not NaN")

    logger.info("Normalizing %d values to range [%s, %s]", len(values), new_min, new_max)

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        logger.warning("All values are identical, returning constant array")
        return [new_min] * len(values)

    range_old = max_val - min_val
    range_new = new_max - new_min

    normalized = [new_min + (v - min_val) * range_new / range_old for v in values]

    logger.info(
        "Normalization complete: original range [%s, %s] -> [%s, %s]",
        min_val,
        max_val,
        new_min,
        new_max,
    )
    return normalized


def standardize_zscore(values: List[Union[int, float]]) -> List[float]:
    """
    Standardize numerical values using z-score method.

    Args:
        values: List of numerical values

    Returns:
        List of standardized values (mean=0, std=1)
    """
    if not values:
        return []

    if len(values) == 1:
        return [0.0]

    mean_val = statistics.mean(values)
    stdev_val = statistics.stdev(values)

    if stdev_val == 0:
        return [0.0] * len(values)

    return [(v - mean_val) / stdev_val for v in values]


def clip_values(
    values: List[Union[int, float]], min_val: Union[int, float], max_val: Union[int, float]
) -> List[Union[int, float]]:
    """
    Clip numerical values to a specified range.

    Args:
        values: List of numerical values
        min_val: Minimum value for clipping
        max_val: Maximum value for clipping

    Returns:
        List of clipped values
    """
    return [max(min_val, min(max_val, v)) for v in values]


def convert_to_int(values: List[str]) -> List[int]:
    """
    Convert string values to integers, excluding non-numerical values.

    Args:
        values: List of string values

    Returns:
        List of successfully converted integers
    """
    result = []
    for v in values:
        try:
            result.append(int(float(v)))
        except (ValueError, TypeError):
            continue
    return result


def log_transform(values: List[Union[int, float]]) -> List[float]:
    """
    Apply logarithmic transformation to positive values.

    Args:
        values: List of numerical values

    Returns:
        List of log-transformed values (only positive numbers)
    """
    return [math.log(v) for v in values if v > 0]


def tokenize_text(text: str) -> str:
    """
    Tokenize text into words, keeping only alphanumeric characters and lowercasing.

    Args:
        text: Text to be processed

    Returns:
        Processed text with tokens separated by spaces
    """
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    tokens = cleaned.split()
    return " ".join(tokens)


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation, keeping only alphanumeric characters and spaces.

    Args:
        text: Text to be processed

    Returns:
        Text with punctuation removed
    """
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_stopwords(text: str, stopwords: List[str]) -> str:
    """
    Remove stop words from text (case-insensitive).

    Args:
        text: Text to be processed (will be lowercased)
        stopwords: List of stop words to remove

    Returns:
        Text with stop words removed
    """
    text_lower = text.lower()
    stopwords_lower = [sw.lower() for sw in stopwords]

    words = text_lower.split()
    filtered_words = [word for word in words if word not in stopwords_lower]

    return " ".join(filtered_words)


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a list of lists into a single list.

    Args:
        nested_list: List containing sublists

    Returns:
        Flattened list
    """
    result = []
    for sublist in nested_list:
        result.extend(sublist)
    return result


def shuffle_list(values: List[Any], seed: int = None) -> List[Any]:
    """
    Randomly shuffle a list of values.

    Args:
        values: List of values to shuffle
        seed: Random seed for reproducibility (optional)

    Returns:
        Shuffled list
    """
    result = values.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(result)
    return result


def get_unique(values: List[Any]) -> List[Any]:
    """
    Get unique values from list (alias for remove_duplicates).

    Args:
        values: List of values

    Returns:
        List of unique values
    """
    return remove_duplicates(values)


def validate_data_quality(
    values: List[Any], missing_threshold: float = 0.3, check_duplicates: bool = True
) -> dict:
    """
    Validate data quality metrics following ML Lifecycle monitoring principles.

    Implements comprehensive data quality assessment as required by ML Lifecycle theory
    for continuous monitoring and validation of data pipeline health.

    Args:
        values: List of values to validate
        missing_threshold: Maximum acceptable ratio of missing values (0-1)
        check_duplicates: Whether to check for duplicate values

    Returns:
        Dictionary containing quality metrics and validation status
    """
    if not isinstance(values, list):
        raise TypeError("Values must be a list")

    total_count = len(values)
    if total_count == 0:
        return {
            "is_valid": False,
            "total_count": 0,
            "missing_count": 0,
            "missing_ratio": 0.0,
            "duplicate_count": 0,
            "duplicate_ratio": 0.0,
            "quality_score": 0.0,
            "validation_errors": ["Empty dataset"],
        }

    missing_count = total_count - len(remove_missing(values))
    missing_ratio = missing_count / total_count

    duplicate_count = 0
    duplicate_ratio = 0.0
    if check_duplicates:
        unique_count = len(remove_duplicates(values))
        duplicate_count = total_count - unique_count
        duplicate_ratio = duplicate_count / total_count

    quality_score = 1.0 - missing_ratio - (duplicate_ratio * 0.5)
    quality_score = max(0.0, quality_score)

    validation_errors = []
    is_valid = True

    if missing_ratio > missing_threshold:
        validation_errors.append(
            f"Missing value ratio ({missing_ratio:.2f}) exceeds threshold ({missing_threshold})"
        )
        is_valid = False

    logger.info(
        "Data quality assessment: %d values, "
        "%.2f%% missing, %.2f%% duplicates, quality score: %.2f",
        total_count,
        missing_ratio * 100,
        duplicate_ratio * 100,
        quality_score,
    )

    return {
        "is_valid": is_valid,
        "total_count": total_count,
        "missing_count": missing_count,
        "missing_ratio": missing_ratio,
        "duplicate_count": duplicate_count,
        "duplicate_ratio": duplicate_ratio,
        "quality_score": quality_score,
        "validation_errors": validation_errors,
    }


def create_data_pipeline_report(
    values: List[Any], operation: str, original_count: Optional[int] = None
) -> dict:
    """
    Create a data pipeline processing report for ML Lifecycle monitoring.

    Generates comprehensive reports for data processing operations to support
    ML Lifecycle theory requirements for traceability and monitoring.

    Args:
        values: Processed data values
        operation: Name of the processing operation
        original_count: Original data count before processing
    """
    if original_count is None:
        original_count = len(values)

    current_count = len(values)
    processing_ratio = current_count / original_count if original_count > 0 else 0.0

    report = {
        "operation": operation,
        "timestamp": str(math.floor(random.random() * 1000000)),
        "original_count": original_count,
        "processed_count": current_count,
        "processing_ratio": processing_ratio,
        "data_retention": processing_ratio,
        "quality_metrics": validate_data_quality(values),
    }

    logger.info(
        "Pipeline report for '%s': %d -> %d (%.2f%% retention)",
        operation,
        original_count,
        current_count,
        processing_ratio * 100,
    )

    return report
