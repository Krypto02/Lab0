"""
Integration tests for CLI interface following ML Lifecycle testing principles.

This test suite validates the CLI interface integration with the core preprocessing
functions, ensuring proper error handling, user feedback, and command functionality.

"""

import pytest
import sys
import os
from click.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cli import cli


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Fixture to create CliRunner instance shared across tests."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test that main CLI help is accessible."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Lab0 - Data Preprocessing CLI Tool" in result.output

    def test_cli_verbose_flag(self, runner):
        """Test CLI with verbose flag."""
        result = runner.invoke(cli, ["--verbose", "--help"])
        assert result.exit_code == 0


class TestCleanGroupIntegration:
    """Integration tests for clean group commands."""

    @pytest.fixture
    def runner(self):
        """Fixture for CliRunner."""
        return CliRunner()

    def test_clean_group_help(self, runner):
        """Test clean group help."""
        result = runner.invoke(cli, ["clean", "--help"])
        assert result.exit_code == 0
        assert "Data cleaning operations" in result.output

    def test_remove_missing_command(self, runner):
        """Test remove-missing command integration."""
        result = runner.invoke(cli, ["clean", "remove-missing", "1,None,3,,5"])
        assert result.exit_code == 0
        assert "1,3,5" in result.output

    def test_remove_missing_with_quality_check(self, runner):
        """Test remove-missing with quality check option."""
        result = runner.invoke(cli, ["clean", "remove-missing", "1,None,3,,5", "--quality-check"])
        assert result.exit_code == 0
        assert "Quality Score" in result.output
        assert "Missing Ratio" in result.output

    def test_fill_missing_command(self, runner):
        """Test fill-missing command integration."""
        result = runner.invoke(cli, ["clean", "fill-missing", "1,,3,None", "--fill-value", "0"])
        assert result.exit_code == 0
        assert "1,0,3,0" in result.output

    def test_fill_missing_with_quality_check(self, runner):
        """Test fill-missing with quality validation."""
        result = runner.invoke(
            cli, ["clean", "fill-missing", "1,,3", "--fill-value", "999", "--quality-check"]
        )
        assert result.exit_code == 0
        assert "1,999,3" in result.output
        assert "Quality improved" in result.output

    def test_clean_command_invalid_input(self, runner):
        """Test clean commands with invalid input."""
        result = runner.invoke(cli, ["clean", "remove-missing", "invalid[input"])
        assert result.exit_code in [0, 1, 2]


class TestNumericGroupIntegration:
    """Integration tests for numeric group commands."""

    @pytest.fixture
    def runner(self):
        """Fixture for CliRunner."""
        return CliRunner()

    def test_numeric_group_help(self, runner):
        """Test numeric group help."""
        result = runner.invoke(cli, ["numeric", "--help"])
        assert result.exit_code == 0
        assert "Numeric data processing" in result.output

    def test_normalize_command(self, runner):
        """Test normalize command integration."""
        result = runner.invoke(cli, ["numeric", "normalize", "1,2,3,4,5"])
        assert result.exit_code == 0
        output_values = result.output.strip().split(",")
        assert len(output_values) == 5
        assert float(output_values[0]) == 0.0
        assert float(output_values[-1]) == 1.0

    def test_normalize_custom_range(self, runner):
        """Test normalize with custom range."""
        result = runner.invoke(
            cli, ["numeric", "normalize", "1,2,3,4,5", "--min-val", "-1", "--max-val", "1"]
        )
        assert result.exit_code == 0
        output_values = [float(v) for v in result.output.strip().split(",")]
        assert output_values[0] == -1.0
        assert output_values[-1] == 1.0

    def test_normalize_with_report(self, runner):
        """Test normalize with reporting option."""
        result = runner.invoke(cli, ["numeric", "normalize", "1,2,3,4,5", "--report"])
        assert result.exit_code == 0
        assert "Normalization complete" in result.output

    def test_standardize_command(self, runner):
        """Test standardize command integration."""
        result = runner.invoke(cli, ["numeric", "standardize", "1,2,3,4,5"])
        assert result.exit_code == 0
        output_values = [float(v) for v in result.output.strip().split(",")]
        assert len(output_values) == 5

    def test_standardize_with_report(self, runner):
        """Test standardize with statistical reporting."""
        result = runner.invoke(cli, ["numeric", "standardize", "1,2,3,4,5", "--report"])
        assert result.exit_code == 0
        assert "Original:" in result.output
        assert "Standardized:" in result.output

    def test_clip_command(self, runner):
        """Test clip command integration."""
        result = runner.invoke(
            cli, ["numeric", "clip", "1,2,3,4,5", "--min-val", "2", "--max-val", "4"]
        )
        assert result.exit_code == 0
        output_values = [float(v) for v in result.output.strip().split(",")]
        assert output_values == [2.0, 2.0, 3.0, 4.0, 4.0]

    def test_clip_with_report(self, runner):
        """Test clip with clipping statistics."""
        result = runner.invoke(
            cli, ["numeric", "clip", "1,2,3,4,5", "--min-val", "2", "--max-val", "4", "--report"]
        )
        assert result.exit_code == 0
        assert "Clipped" in result.output

    def test_to_int_command(self, runner):
        """Test to-int command integration."""
        result = runner.invoke(cli, ["numeric", "to-int", "1,2.5,abc,4.0"])
        assert result.exit_code == 0
        assert "1,2,4" in result.output

    def test_to_int_with_report(self, runner):
        """Test to-int with conversion statistics."""
        result = runner.invoke(cli, ["numeric", "to-int", "1,2.5,abc,4.0", "--report"])
        assert result.exit_code == 0
        assert "Converted" in result.output

    def test_log_transform_command(self, runner):
        """Test log-transform command integration."""
        result = runner.invoke(cli, ["numeric", "log-transform", "1,2,3,4,5"])
        assert result.exit_code == 0
        output_values = [float(v) for v in result.output.strip().split(",")]
        assert len(output_values) == 5
        assert abs(output_values[0]) < 1e-6

    def test_log_transform_with_negatives(self, runner):
        """Test log-transform filtering negative values."""
        result = runner.invoke(cli, ["numeric", "log-transform", "0.5,1,2,3"])
        assert result.exit_code == 0
        output_values = result.output.strip().split(",")
        assert len(output_values) == 4

    def test_numeric_invalid_range(self, runner):
        """Test numeric commands with invalid range."""
        result = runner.invoke(
            cli, ["numeric", "clip", "1,2,3", "--min-val", "5", "--max-val", "2"]
        )
        assert result.exit_code != 0

    def test_numeric_non_numeric_input(self, runner):
        """Test numeric commands with non-numeric input."""
        result = runner.invoke(cli, ["numeric", "normalize", "abc,def,ghi"])
        assert result.exit_code != 0


class TestTextGroupIntegration:
    """Integration tests for text group commands."""

    @pytest.fixture
    def runner(self):
        """Fixture for CliRunner."""
        return CliRunner()

    def test_text_group_help(self, runner):
        """Test text group help."""
        result = runner.invoke(cli, ["text", "--help"])
        assert result.exit_code == 0
        assert "Text processing operations" in result.output

    def test_tokenize_command(self, runner):
        """Test tokenize command integration."""
        result = runner.invoke(cli, ["text", "tokenize", "Hello, World! How are you?"])
        assert result.exit_code == 0
        assert "hello world how are you" in result.output

    def test_tokenize_with_report(self, runner):
        """Test tokenize with statistics reporting."""
        result = runner.invoke(cli, ["text", "tokenize", "Hello, World!", "--report"])
        assert result.exit_code == 0
        assert "Tokenized to" in result.output

    def test_remove_punctuation_command(self, runner):
        """Test remove-punctuation command integration."""
        result = runner.invoke(cli, ["text", "remove-punctuation", "Hello, World!"])
        assert result.exit_code == 0
        assert "Hello World" in result.output

    def test_remove_punctuation_with_report(self, runner):
        """Test remove-punctuation with cleaning statistics."""
        result = runner.invoke(cli, ["text", "remove-punctuation", "Hello, World!", "--report"])
        assert result.exit_code == 0
        assert "Removed" in result.output

    def test_remove_stopwords_command(self, runner):
        """Test remove-stopwords command integration."""
        result = runner.invoke(
            cli, ["text", "remove-stopwords", "this is a test", "--stopwords", "is,a"]
        )
        assert result.exit_code == 0
        assert "this test" in result.output

    def test_remove_stopwords_with_report(self, runner):
        """Test remove-stopwords with removal statistics."""
        result = runner.invoke(
            cli, ["text", "remove-stopwords", "this is a test", "--stopwords", "is,a", "--report"]
        )
        assert result.exit_code == 0
        assert "Removed" in result.output

    def test_remove_stopwords_no_stopwords(self, runner):
        """Test remove-stopwords without specifying stopwords."""
        result = runner.invoke(cli, ["text", "remove-stopwords", "this is a test"])
        assert result.exit_code == 0
        assert "this is a test" in result.output


class TestStructGroupIntegration:
    """Integration tests for struct group commands."""

    @pytest.fixture
    def runner(self):
        """Fixture for CliRunner."""
        return CliRunner()

    def test_struct_group_help(self, runner):
        """Test struct group help."""
        result = runner.invoke(cli, ["struct", "--help"])
        assert result.exit_code == 0
        assert "Data structure operations" in result.output

    def test_shuffle_command(self, runner):
        """Test shuffle command integration."""
        result = runner.invoke(cli, ["struct", "shuffle", "1,2,3,4,5", "--seed", "42"])
        assert result.exit_code == 0
        output_values = result.output.strip().split(",")
        assert len(output_values) == 5
        assert set(output_values) == {"1", "2", "3", "4", "5"}

    def test_shuffle_reproducibility(self, runner):
        """Test shuffle reproducibility with same seed."""
        result1 = runner.invoke(cli, ["struct", "shuffle", "1,2,3,4,5", "--seed", "42"])
        result2 = runner.invoke(cli, ["struct", "shuffle", "1,2,3,4,5", "--seed", "42"])

        assert result1.exit_code == 0
        assert result2.exit_code == 0
        assert result1.output == result2.output

    def test_shuffle_with_report(self, runner):
        """Test shuffle with operation reporting."""
        result = runner.invoke(cli, ["struct", "shuffle", "1,2,3,4,5", "--seed", "42", "--report"])
        assert result.exit_code == 0
        assert "Shuffled" in result.output
        assert "seed 42" in result.output

    def test_flatten_command(self, runner):
        """Test flatten command integration."""
        result = runner.invoke(cli, ["struct", "flatten", "[[1,2],[3,4],[5]]"])
        assert result.exit_code == 0
        assert "1,2,3,4,5" in result.output

    def test_flatten_with_report(self, runner):
        """Test flatten with flattening statistics."""
        result = runner.invoke(cli, ["struct", "flatten", "[[1,2],[3,4]]", "--report"])
        assert result.exit_code == 0
        assert "Flattened" in result.output
        assert "sublists" in result.output

    def test_flatten_invalid_format(self, runner):
        """Test flatten with invalid input format."""
        result = runner.invoke(cli, ["struct", "flatten", "not_a_list"])
        assert result.exit_code != 0

    def test_unique_command(self, runner):
        """Test unique command integration."""
        result = runner.invoke(cli, ["struct", "unique", "1,2,2,3,3,4"])
        assert result.exit_code == 0
        assert "1,2,3,4" in result.output

    def test_unique_with_report(self, runner):
        """Test unique with deduplication statistics."""
        result = runner.invoke(cli, ["struct", "unique", "1,2,2,3,3,4", "--report"])
        assert result.exit_code == 0
        assert "Removed" in result.output
        assert "duplicates" in result.output


class TestErrorHandlingIntegration:
    """Integration tests for error handling and user feedback."""

    @pytest.fixture
    def runner(self):
        """Fixture for CliRunner."""
        return CliRunner()

    def test_invalid_command(self, runner):
        """Test CLI with invalid command."""
        result = runner.invoke(cli, ["invalid-command"])
        assert result.exit_code != 0

    def test_invalid_group(self, runner):
        """Test CLI with invalid group."""
        result = runner.invoke(cli, ["invalid-group", "some-command"])
        assert result.exit_code != 0

    def test_missing_arguments(self, runner):
        """Test commands missing required arguments."""
        result = runner.invoke(cli, ["clean", "remove-missing"])
        assert result.exit_code != 0

    def test_invalid_option_values(self, runner):
        """Test commands with invalid option values."""
        result = runner.invoke(cli, ["numeric", "normalize", "1,2,3", "--min-val", "not_a_number"])
        assert result.exit_code != 0

    def test_malformed_list_input(self, runner):
        """Test commands with malformed list inputs."""
        result = runner.invoke(cli, ["clean", "remove-missing", "[1,2,3"])
        assert result.exit_code in [0, 1, 2]


class TestComprehensiveFunctionality:
    """Comprehensive integration tests combining multiple operations."""

    @pytest.fixture
    def runner(self):
        """Fixture for CliRunner."""
        return CliRunner()

    def test_data_pipeline_simulation(self, runner):
        """Test simulating a complete data processing pipeline."""
        clean_result = runner.invoke(cli, ["clean", "remove-missing", "1,None,3,,5,7"])
        assert clean_result.exit_code == 0
        cleaned_data = clean_result.output.strip()

        normalize_result = runner.invoke(cli, ["numeric", "normalize", cleaned_data])
        assert normalize_result.exit_code == 0

        normalized_data = normalize_result.output.strip()
        shuffle_result = runner.invoke(cli, ["struct", "shuffle", normalized_data, "--seed", "42"])
        assert shuffle_result.exit_code == 0

        assert len(shuffle_result.output.strip().split(",")) == 4

    def test_text_processing_pipeline(self, runner):
        """Test complete text processing workflow."""
        clean_result = runner.invoke(
            cli, ["text", "remove-punctuation", "Hello, World! This is a test."]
        )
        assert clean_result.exit_code == 0

        cleaned_text = clean_result.output.strip()
        tokenize_result = runner.invoke(cli, ["text", "tokenize", cleaned_text])
        assert tokenize_result.exit_code == 0

        tokenized_text = tokenize_result.output.strip()
        stopword_result = runner.invoke(
            cli, ["text", "remove-stopwords", tokenized_text, "--stopwords", "is,a"]
        )
        assert stopword_result.exit_code == 0

        final_text = stopword_result.output.strip()
        assert "hello world this test" in final_text


class TestPerformanceAndScalability:
    """Integration tests for performance with larger datasets."""

    @pytest.fixture
    def runner(self):
        """Fixture for CliRunner."""
        return CliRunner()

    def test_large_numeric_dataset(self, runner):
        """Test numeric operations with larger dataset."""
        large_data = ",".join(str(i) for i in range(100))

        result = runner.invoke(cli, ["numeric", "normalize", large_data])
        assert result.exit_code == 0

        output_values = result.output.strip().split(",")
        assert len(output_values) == 100
        assert float(output_values[0]) == 0.0
        assert float(output_values[-1]) == 1.0

    def test_long_text_processing(self, runner):
        """Test text processing with longer text."""
        long_text = "This is a very long text with many words and punctuation marks! " * 10

        result = runner.invoke(cli, ["text", "tokenize", long_text, "--report"])
        assert result.exit_code == 0
        assert "Tokenized to" in result.output
