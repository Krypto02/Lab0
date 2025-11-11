"""
Unit tests for preprocessing functions following ML Lifecycle testing principles.

"""

import math
import pytest
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import preprocessing


class TestDataQuality:
    """Test data quality validation functions."""

    def test_validate_data_quality_empty_list(self):
        """Test validation with empty dataset."""
        result = preprocessing.validate_data_quality([])
        assert result["is_valid"] == False
        assert result["total_count"] == 0
        assert "Empty dataset" in result["validation_errors"]

    def test_validate_data_quality_perfect_data(self):
        """Test validation with perfect quality data."""
        data = [1, 2, 3, 4, 5]
        result = preprocessing.validate_data_quality(data)
        assert result["is_valid"] == True
        assert result["missing_ratio"] == 0.0
        assert result["quality_score"] == 1.0

    def test_validate_data_quality_high_missing_ratio(self):
        """Test validation fails with high missing ratio."""
        data = [1, None, None, None, 5]
        result = preprocessing.validate_data_quality(data, missing_threshold=0.3)
        assert result["is_valid"] == False
        assert result["missing_ratio"] == 0.6
        assert len(result["validation_errors"]) > 0


class TestDataCleaning:
    """Test data cleaning functions with fixtures and edge cases."""

    @pytest.fixture
    def mixed_data_with_missing(self):
        """Fixture providing mixed data with various missing value types."""
        return [1, None, 3, "", 5, math.nan, 7, None, 9, ""]

    @pytest.fixture
    def numeric_data(self):
        """Fixture providing clean numeric data for transformations."""
        return [1.0, 2.0, 3.0, 4.0, 5.0]

    @pytest.fixture
    def text_data(self):
        """Fixture providing text data for processing tests."""
        return "Hello, World! This is a test with punctuation & numbers 123."

    def test_remove_missing_basic(self, mixed_data_with_missing):
        """Test basic missing value removal."""
        result = preprocessing.remove_missing(mixed_data_with_missing)
        expected = [1, 3, 5, 7, 9]
        assert result == expected

    def test_remove_missing_empty_input(self):
        """Test remove_missing with empty list."""
        result = preprocessing.remove_missing([])
        assert result == []

    def test_remove_missing_no_missing_values(self, numeric_data):
        """Test remove_missing when no missing values present."""
        result = preprocessing.remove_missing(numeric_data)
        assert result == numeric_data

    def test_remove_missing_all_missing(self):
        """Test remove_missing when all values are missing."""
        missing_data = [None, "", math.nan]
        result = preprocessing.remove_missing(missing_data)
        assert result == []

    def test_remove_missing_invalid_input(self):
        """Test remove_missing with invalid input type."""
        with pytest.raises(TypeError):
            preprocessing.remove_missing("not a list")

    @pytest.mark.parametrize(
        "input_data,fill_value,expected",
        [
            ([1, None, 3], 0, [1, 0, 3]),
            ([1, "", 3], -1, [1, -1, 3]),
            ([1, math.nan, 3], 999, [1, 999, 3]),
            ([], 0, []),
            ([1, 2, 3], 0, [1, 2, 3]),
        ],
    )
    def test_fill_missing_parametrized(self, input_data, fill_value, expected):
        """Test fill_missing with various input combinations."""
        result = preprocessing.fill_missing(input_data, fill_value)
        assert result == expected

    def test_remove_duplicates_preserve_order(self):
        """Test that remove_duplicates preserves original order."""
        input_data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        result = preprocessing.remove_duplicates(input_data)
        expected = [3, 1, 4, 5, 9, 2, 6]
        assert result == expected

    def test_remove_duplicates_mixed_types(self):
        """Test remove_duplicates with mixed data types."""
        input_data = [1, "1", 1, 2, "2", 2.0]
        result = preprocessing.remove_duplicates(input_data)
        expected = [1, "1", 2, "2"]
        assert result == expected


class TestNumericTransformations:
    """Test numeric transformation functions."""

    @pytest.fixture
    def sample_numeric_data(self):
        """Fixture for standard numeric data."""
        return [1, 2, 3, 4, 5]

    @pytest.fixture
    def large_range_data(self):
        """Fixture for data with large numeric range."""
        return [100, 200, 300, 400, 500]

    def test_normalize_minmax_default_range(self, sample_numeric_data):
        """Test min-max normalization with default 0-1 range."""
        result = preprocessing.normalize_minmax(sample_numeric_data)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert result == expected

    @pytest.mark.parametrize(
        "new_min,new_max,expected",
        [
            (-1, 1, [-1.0, -0.5, 0.0, 0.5, 1.0]),
            (0, 10, [0.0, 2.5, 5.0, 7.5, 10.0]),
            (100, 200, [100.0, 125.0, 150.0, 175.0, 200.0]),
        ],
    )
    def test_normalize_minmax_custom_ranges(self, sample_numeric_data, new_min, new_max, expected):
        """Test min-max normalization with various custom ranges."""
        result = preprocessing.normalize_minmax(sample_numeric_data, new_min, new_max)
        assert result == expected

    def test_normalize_minmax_identical_values(self):
        """Test normalization with identical input values."""
        identical_data = [5, 5, 5, 5]
        result = preprocessing.normalize_minmax(identical_data, 0, 1)
        expected = [0, 0, 0, 0]
        assert result == expected

    def test_normalize_minmax_invalid_range(self, sample_numeric_data):
        """Test normalization with invalid min/max range."""
        with pytest.raises(ValueError, match="new_min must be less than new_max"):
            preprocessing.normalize_minmax(sample_numeric_data, 1, 1)

    def test_normalize_minmax_invalid_input_type(self):
        """Test normalization with invalid input type."""
        with pytest.raises(TypeError):
            preprocessing.normalize_minmax("not a list")

    def test_normalize_minmax_nan_values(self):
        """Test normalization with NaN values."""
        data_with_nan = [1, 2, math.nan, 4, 5]
        with pytest.raises(ValueError, match="All values must be numeric and not NaN"):
            preprocessing.normalize_minmax(data_with_nan)

    def test_standardize_zscore_basic(self, sample_numeric_data):
        """Test z-score standardization."""
        result = preprocessing.standardize_zscore(sample_numeric_data)

        mean_result = sum(result) / len(result)
        assert abs(mean_result) < 1e-10

        variance = sum((x - mean_result) ** 2 for x in result) / (len(result) - 1)
        std_result = math.sqrt(variance)
        assert abs(std_result - 1.0) < 1e-10

    def test_standardize_zscore_single_value(self):
        """Test standardization with single value."""
        result = preprocessing.standardize_zscore([5])
        assert result == [0.0]

    def test_standardize_zscore_identical_values(self):
        """Test standardization with identical values."""
        result = preprocessing.standardize_zscore([3, 3, 3, 3])
        expected = [0.0, 0.0, 0.0, 0.0]
        assert result == expected

    @pytest.mark.parametrize(
        "values,min_val,max_val,expected",
        [
            ([1, 2, 3, 4, 5], 2, 4, [2, 2, 3, 4, 4]),
            ([-10, -5, 0, 5, 10], -3, 3, [-3, -3, 0, 3, 3]),
            ([1.5, 2.5, 3.5], 2, 3, [2, 2.5, 3]),
        ],
    )
    def test_clip_values_parametrized(self, values, min_val, max_val, expected):
        """Test clipping with various value ranges."""
        result = preprocessing.clip_values(values, min_val, max_val)
        assert result == expected

    def test_convert_to_int_mixed_input(self):
        """Test integer conversion with mixed valid/invalid inputs."""
        input_strings = ["1", "2.7", "abc", "4.0", "def", "-3"]
        result = preprocessing.convert_to_int(input_strings)
        expected = [1, 2, 4, -3]
        assert result == expected

    def test_convert_to_int_empty_input(self):
        """Test integer conversion with empty input."""
        result = preprocessing.convert_to_int([])
        assert result == []

    def test_log_transform_positive_values(self):
        """Test logarithmic transformation with positive values."""
        input_values = [1, 2, 3, 4, 5]
        result = preprocessing.log_transform(input_values)

        assert len(result) == 5

        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[1] - math.log(2)) < 1e-10

    def test_log_transform_mixed_values(self):
        """Test log transformation filtering negative and zero values."""
        input_values = [-2, -1, 0, 1, 2, 3]
        result = preprocessing.log_transform(input_values)

        assert len(result) == 3
        assert abs(result[0] - 0.0) < 1e-10


class TestTextProcessing:
    """Test text processing functions."""

    @pytest.fixture
    def sample_text(self):
        """Fixture providing sample text for processing."""
        return "Hello, World! This is a TEST with 123 numbers."

    def test_tokenize_text_basic(self, sample_text):
        """Test basic text tokenization."""
        result = preprocessing.tokenize_text(sample_text)
        expected = "hello world this is a test with 123 numbers"
        assert result == expected

    def test_tokenize_text_empty_string(self):
        """Test tokenization with empty string."""
        result = preprocessing.tokenize_text("")
        assert result == ""

    def test_tokenize_text_only_punctuation(self):
        """Test tokenization with only punctuation."""
        result = preprocessing.tokenize_text("!@#$%^&*()")
        assert result == ""

    def test_remove_punctuation_basic(self, sample_text):
        """Test punctuation removal."""
        result = preprocessing.remove_punctuation(sample_text)
        expected = "Hello World This is a TEST with 123 numbers"
        assert result == expected

    @pytest.mark.parametrize(
        "text,stopwords,expected",
        [
            ("this is a test", ["is", "a"], "this test"),
            ("The quick brown fox", ["the", "quick"], "brown fox"),
            ("hello world", [], "hello world"),
            ("", ["any"], ""),
        ],
    )
    def test_remove_stopwords_parametrized(self, text, stopwords, expected):
        """Test stop word removal with various inputs."""
        result = preprocessing.remove_stopwords(text, stopwords)
        assert result == expected

    def test_remove_stopwords_case_insensitive(self):
        """Test that stop word removal is case insensitive."""
        text = "The Quick BROWN Fox"
        stopwords = ["the", "quick"]
        result = preprocessing.remove_stopwords(text, stopwords)
        expected = "brown fox"
        assert result == expected


class TestDataStructures:
    """Test data structure manipulation functions."""

    @pytest.fixture
    def sample_list(self):
        """Fixture providing sample list for structure operations."""
        return [1, 2, 3, 4, 5]

    def test_flatten_list_basic(self):
        """Test basic list flattening."""
        nested = [[1, 2], [3, 4], [5]]
        result = preprocessing.flatten_list(nested)
        expected = [1, 2, 3, 4, 5]
        assert result == expected

    def test_flatten_list_empty_sublists(self):
        """Test flattening with empty sublists."""
        nested = [[1, 2], [], [3, 4]]
        result = preprocessing.flatten_list(nested)
        expected = [1, 2, 3, 4]
        assert result == expected

    def test_shuffle_list_with_seed(self, sample_list):
        """Test shuffling with reproducible seed."""
        result1 = preprocessing.shuffle_list(sample_list, seed=42)
        result2 = preprocessing.shuffle_list(sample_list, seed=42)

        assert result1 == result2

        assert sorted(result1) == sorted(sample_list)

    def test_shuffle_list_different_seeds(self, sample_list):
        """Test that different seeds produce different results."""
        result1 = preprocessing.shuffle_list(sample_list, seed=42)
        result2 = preprocessing.shuffle_list(sample_list, seed=123)

        assert len(result1) == len(result2)
        assert sorted(result1) == sorted(result2)

    def test_shuffle_list_no_seed(self, sample_list):
        """Test shuffling without seed (random)."""
        result = preprocessing.shuffle_list(sample_list, seed=None)

        assert sorted(result) == sorted(sample_list)
        assert len(result) == len(sample_list)

    def test_get_unique_alias(self):
        """Test that get_unique is correctly aliased to remove_duplicates."""
        test_data = [1, 2, 2, 3, 1, 4]
        result1 = preprocessing.get_unique(test_data)
        result2 = preprocessing.remove_duplicates(test_data)
        assert result1 == result2


class TestReproducibilityAndLogging:
    """Test reproducibility and logging functionality."""

    def test_shuffle_reproducibility(self):
        """Test that shuffle operations are reproducible with same seed."""
        data = list(range(100))
        results = []
        for _ in range(3):
            result = preprocessing.shuffle_list(data, seed=12345)
            results.append(result)

        for result in results[1:]:
            assert result == results[0]

    @patch("src.preprocessing.logger")
    def test_logging_calls(self, mock_logger):
        """Test that appropriate logging calls are made."""

        preprocessing.remove_missing([1, None, 3])
        mock_logger.info.assert_called()

        preprocessing.normalize_minmax([1, 2, 3, 4, 5])
        assert mock_logger.info.call_count >= 2


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_inputs_handling(self):
        """Test that functions handle empty inputs gracefully."""
        empty_list = []

        assert preprocessing.remove_missing(empty_list) == []
        assert preprocessing.fill_missing(empty_list) == []
        assert preprocessing.remove_duplicates(empty_list) == []
        assert preprocessing.normalize_minmax(empty_list) == []
        assert preprocessing.standardize_zscore(empty_list) == []
        assert preprocessing.flatten_list(empty_list) == []

    def test_single_element_inputs(self):
        """Test functions with single element inputs."""
        single_element = [42]

        assert preprocessing.remove_missing(single_element) == [42]
        assert preprocessing.fill_missing(single_element) == [42]
        assert preprocessing.remove_duplicates(single_element) == [42]
        assert preprocessing.normalize_minmax(single_element) == [0.0]
        assert preprocessing.standardize_zscore(single_element) == [0.0]

    def test_very_large_inputs(self):
        """Test functions with large datasets."""
        large_data = list(range(10000))

        result = preprocessing.remove_duplicates(large_data)
        assert len(result) == 10000

        normalized = preprocessing.normalize_minmax(large_data)
        assert len(normalized) == 10000
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
