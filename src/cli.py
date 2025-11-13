"""
Command Line Interface for Lab data preprocessing functions.
"""

import ast
import logging
import sys

import click

from . import preprocessing

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_list(ctx, param, value):
    """
    Helper function to parse comma-separated values into a list with validation.

    Implements robust input parsing following ML Lifecycle data validation principles.
    """
    if value is None:
        return None

    try:
        if value.startswith("[") and value.endswith("]"):
            parsed = ast.literal_eval(value)
            logger.info("Parsed list literal with %d elements", len(parsed))
            return parsed
        else:
            items = [item.strip() for item in value.split(",")]
            result = []
            for item in items:
                if item == "" or item.lower() == "none":
                    result.append(None)
                else:
                    try:
                        if "." in item:
                            result.append(float(item))
                        else:
                            result.append(int(item))
                    except ValueError:
                        result.append(item)

            logger.info("Parsed comma-separated list with %d elements", len(result))
            return result

    except Exception as exc:
        logger.error("List parsing failed: %s", exc)
        raise click.BadParameter(
            f"Invalid list format: {value}. Use 'item1,item2,item3' or '[item1,item2,item3]'"
        ) from exc


def parse_numeric_list(ctx, param, value):
    """
    Helper function to parse comma-separated numeric values with validation.

    Ensures all values are numeric as required by ML data processing pipelines.
    """
    if value is None:
        return None

    try:
        items = [item.strip() for item in value.split(",") if item.strip()]
        numeric_values = []

        for item in items:
            try:
                numeric_values.append(float(item))
            except ValueError as exc:
                raise click.BadParameter(
                    f"Non-numeric value found: '{item}'. All values must be numbers."
                ) from exc

        logger.info("Parsed %d numeric values", len(numeric_values))
        return numeric_values

    except click.BadParameter:
        raise
    except Exception as exc:
        logger.error("Numeric list parsing failed: %s", exc)
        raise click.BadParameter(f"Invalid numeric list format: {value}") from exc


def handle_processing_error(operation: str, error: Exception):
    """
    Centralized error handling for data processing operations.

    Implements ML Lifecycle error management with proper logging and user feedback.
    """
    error_msg = f"Error in {operation}: {str(error)}"
    logger.error("%s", error_msg)
    click.echo(f"{error_msg}", err=True)

    if isinstance(error, (TypeError, ValueError)):
        click.echo("Check your input format and data types", err=True)
    elif isinstance(error, ZeroDivisionError):
        click.echo("Check for division by zero in your data", err=True)

    sys.exit(1)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """
    Lab0 - Data Preprocessing CLI Tool

    A comprehensive tool for data cleaning, numeric processing,
    text processing, and data structure operations following
    ML Lifecycle theory and best practices.

    Use --verbose for detailed operation logging.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")


@cli.group()
def clean():
    """
    Data cleaning operations

    Commands for data quality assurance including missing value handling,
    deduplication, and data validation following ML pipeline standards.
    """


@cli.group()
def numeric():
    """
    Numeric data processing operations

    Commands for numerical transformations including normalization,
    standardization, clipping, and scaling for ML model preparation.
    """


@cli.group()
def text():
    """
    Text processing operations

    Commands for natural language processing including tokenization,
    cleaning, and preprocessing for text-based ML workflows.
    """


@cli.group()
def struct():
    """
    Data structure operations

    Commands for data organization including shuffling, flattening,
    and structural transformations for efficient data pipeline processing.
    """


@clean.command(name="remove-missing")
@click.argument("values", callback=parse_list)
@click.option("--quality-check", is_flag=True, help="Perform data quality validation")
def remove_missing_cmd(values, quality_check):
    """
    Remove missing values from a list with optional quality assessment.

    VALUES: Comma-separated list of values (use 'None' or empty for missing)

    Example: lab0 clean remove-missing "1,None,3,," --quality-check
    """
    try:
        if quality_check:
            quality_report = preprocessing.validate_data_quality(values)
            click.echo(f"Quality Score: {quality_report['quality_score']:.2f}")
            click.echo(f"Missing Ratio: {quality_report['missing_ratio']:.2%}")

        result = preprocessing.remove_missing(values)
        click.echo(",".join(str(v) for v in result))

        if quality_check:
            pipeline_report = preprocessing.create_data_pipeline_report(
                result, "remove_missing", len(values)
            )
            click.echo(f"Data retention: {pipeline_report['data_retention']:.2%}")

    except Exception as exc:
        handle_processing_error("remove_missing", exc)


@clean.command(name="fill-missing")
@click.argument("values", callback=parse_list)
@click.option("--fill-value", default=0, help="Value to replace missing values with")
@click.option("--quality-check", is_flag=True, help="Perform data quality validation")
def fill_missing_cmd(values, fill_value, quality_check):
    """
    Fill missing values with a specified value and validate data quality.

    VALUES: Comma-separated list of values (use 'None' or empty for missing)

    Example: lab0 clean fill-missing "1,,3,None" --fill-value 0 --quality-check
    """
    try:
        if quality_check:
            original_quality = preprocessing.validate_data_quality(values)
            click.echo(f"Original missing ratio: {original_quality['missing_ratio']:.2%}")

        result = preprocessing.fill_missing(values, fill_value)
        click.echo(",".join(str(v) for v in result))

        if quality_check:
            new_quality = preprocessing.validate_data_quality(result)
            click.echo(f"Quality improved: {new_quality['quality_score']:.2f}")

    except Exception as exc:
        handle_processing_error("fill_missing", exc)


@numeric.command(name="normalize")
@click.argument("values", callback=parse_numeric_list)
@click.option("--min-val", default=0.0, help="New minimum value")
@click.option("--max-val", default=1.0, help="New maximum value")
@click.option("--report", is_flag=True, help="Generate processing report")
def normalize_cmd(values, min_val, max_val, report):
    """
    Normalize numerical values using min-max scaling with validation.

    VALUES: Comma-separated list of numbers

    Example: lab0 numeric normalize "1,2,3,4,5" --min-val 0 --max-val 1 --report
    """
    try:
        result = preprocessing.normalize_minmax(values, min_val, max_val)
        click.echo(",".join(f"{v:.6f}" for v in result))

        if report:
            preprocessing.create_data_pipeline_report(result, "normalize_minmax", len(values))
            click.echo(
                f"Normalization complete: "
                f"[{min(values):.3f}, {max(values):.3f}] → [{min_val}, {max_val}]"
            )

    except Exception as exc:
        handle_processing_error("normalize", exc)


@numeric.command(name="standardize")
@click.argument("values", callback=parse_numeric_list)
@click.option("--report", is_flag=True, help="Generate processing report")
def standardize_cmd(values, report):
    """
    Standardize numerical values using z-score method with statistics reporting.

    VALUES: Comma-separated list of numbers

    Example: lab0 numeric standardize "1,2,3,4,5" --report
    """
    try:
        result = preprocessing.standardize_zscore(values)
        click.echo(",".join(f"{v:.6f}" for v in result))

        if report:
            import statistics

            original_mean = statistics.mean(values)
            original_std = statistics.stdev(values) if len(values) > 1 else 0
            result_mean = statistics.mean(result) if result else 0
            result_std = statistics.stdev(result) if len(result) > 1 else 0

            click.echo(f"Original: μ={original_mean:.3f}, σ={original_std:.3f}")
            click.echo(f"Standardized: μ={result_mean:.3f}, σ={result_std:.3f}")

    except Exception as exc:
        handle_processing_error("standardize", exc)


@numeric.command(name="clip")
@click.argument("values", callback=parse_numeric_list)
@click.option("--min-val", default=0.0, help="Minimum value for clipping")
@click.option("--max-val", default=1.0, help="Maximum value for clipping")
@click.option("--report", is_flag=True, help="Show clipping statistics")
def clip_cmd(values, min_val, max_val, report):
    """
    Clip numerical values to a specified range with outlier reporting.

    VALUES: Comma-separated list of numbers

    Example: lab0 numeric clip "1,2,3,4,5" --min-val 2 --max-val 4 --report
    """
    try:
        if min_val >= max_val:
            raise ValueError("min-val must be less than max-val")

        result = preprocessing.clip_values(values, min_val, max_val)
        click.echo(",".join(str(v) for v in result))

        if report:
            clipped_low = sum(1 for v in values if v < min_val)
            clipped_high = sum(1 for v in values if v > max_val)
            click.echo(f"Clipped {clipped_low} low values, {clipped_high} high values")

    except Exception as exc:
        handle_processing_error("clip", exc)


@numeric.command(name="to-int")
@click.argument("values")
@click.option("--report", is_flag=True, help="Show conversion statistics")
def to_int_cmd(values, report):
    """
    Convert string values to integers with conversion reporting.

    VALUES: Comma-separated list of values

    Example: lab0 numeric to-int "1,2.5,abc,4" --report
    """
    try:
        string_list = [item.strip() for item in values.split(",")]
        original_count = len(string_list)

        result = preprocessing.convert_to_int(string_list)
        click.echo(",".join(str(v) for v in result))

        if report:
            converted_count = len(result)
            success_rate = converted_count / original_count if original_count > 0 else 0
            click.echo(f"Converted {converted_count}/{original_count} values ({success_rate:.1%})")

    except Exception as exc:
        handle_processing_error("to_int", exc)


@numeric.command(name="log-transform")
@click.argument("values", callback=parse_numeric_list)
@click.option("--report", is_flag=True, help="Show transformation statistics")
def log_transform_cmd(values, report):
    """
    Apply logarithmic transformation to positive values with validation.

    VALUES: Comma-separated list of numbers

    Example: lab0 numeric log-transform "1,2,3,4,5" --report
    """
    try:
        positive_count = sum(1 for v in values if v > 0)
        if positive_count == 0:
            click.echo("No positive values found for log transformation", err=True)
            return

        result = preprocessing.log_transform(values)
        click.echo(",".join(f"{v:.6f}" for v in result))

        if report:
            excluded_count = len(values) - len(result)
            click.echo(f"Transformed {len(result)} values, excluded {excluded_count} non-positive")

    except Exception as exc:
        handle_processing_error("log_transform", exc)


@text.command(name="tokenize")
@click.argument("text")
@click.option("--report", is_flag=True, help="Show tokenization statistics")
def tokenize_cmd(text, report):
    """
    Tokenize text into words with alphanumeric characters only.

    TEXT: Text string to tokenize

    Example: lab0 text tokenize "Hello, World! How are you?" --report
    """
    try:
        original_length = len(text)
        result = preprocessing.tokenize_text(text)
        click.echo(result)

        if report:
            token_count = len(result.split())
            click.echo(f"Tokenized to {token_count} words from {original_length} characters")

    except Exception as exc:
        handle_processing_error("tokenize", exc)


@text.command(name="remove-punctuation")
@click.argument("text")
@click.option("--report", is_flag=True, help="Show cleaning statistics")
def remove_punctuation_cmd(text, report):
    """
    Remove punctuation from text, keeping alphanumeric and spaces.

    TEXT: Text string to process

    Example: lab0 text remove-punctuation "Hello, World!" --report
    """
    try:
        original_length = len(text)
        result = preprocessing.remove_punctuation(text)
        click.echo(result)

        if report:
            removed_chars = original_length - len(result)
            click.echo(f"Removed {removed_chars} punctuation characters")

    except Exception as exc:
        handle_processing_error("remove_punctuation", exc)


@text.command(name="remove-stopwords")
@click.argument("text")
@click.option("--stopwords", help="Comma-separated list of stop words to remove")
@click.option("--report", is_flag=True, help="Show removal statistics")
def remove_stopwords_cmd(text, stopwords, report):
    """
    Remove stop words from text with detailed reporting.

    TEXT: Text string to process

    Example: lab0 text remove-stopwords "this is a test" --stopwords "is,a" --report
    """
    try:
        original_words = text.split()

        if stopwords:
            stop_list = [word.strip() for word in stopwords.split(",")]
        else:
            stop_list = []

        result = preprocessing.remove_stopwords(text, stop_list)
        click.echo(result)

        if report:
            result_words = result.split()
            removed_count = len(original_words) - len(result_words)
            click.echo(f"Removed {removed_count} stop words")

    except Exception as exc:
        handle_processing_error("remove_stopwords", exc)


@struct.command(name="shuffle")
@click.argument("values", callback=parse_list)
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option("--report", is_flag=True, help="Show shuffle information")
def shuffle_cmd(values, seed, report):
    """
    Randomly shuffle a list of values with reproducible seeding.

    VALUES: Comma-separated list of values

    Example: lab0 struct shuffle "1,2,3,4,5" --seed 42 --report
    """
    try:
        result = preprocessing.shuffle_list(values, seed)
        click.echo(",".join(str(v) for v in result))

        if report:
            seed_info = f"seed {seed}" if seed is not None else "random seed"
            click.echo(f"Shuffled {len(result)} values using {seed_info}")

    except Exception as exc:
        handle_processing_error("shuffle", exc)


@struct.command(name="flatten")
@click.argument("nested_list")
@click.option("--report", is_flag=True, help="Show flattening statistics")
def flatten_cmd(nested_list, report):
    """
    Flatten a list of lists into a single list with validation.

    NESTED_LIST: String representation of nested list (e.g., "[[1,2],[3,4]]")

    Example: lab0 struct flatten "[[1,2],[3,4]]" --report
    """
    try:
        parsed_list = ast.literal_eval(nested_list)
        if not isinstance(parsed_list, list) or not all(
            isinstance(sublist, list) for sublist in parsed_list
        ):
            raise ValueError("Input must be a list of lists")

        result = preprocessing.flatten_list(parsed_list)
        click.echo(",".join(str(v) for v in result))

        if report:
            sublists_count = len(parsed_list)
            total_elements = len(result)
            click.echo(f"Flattened {sublists_count} sublists into {total_elements} elements")

    except Exception as exc:
        handle_processing_error("flatten", exc)


@struct.command(name="unique")
@click.argument("values", callback=parse_list)
@click.option("--report", is_flag=True, help="Show deduplication statistics")
def unique_cmd(values, report):
    """
    Get unique values from a list while preserving order.

    VALUES: Comma-separated list of values

    Example: lab0 struct unique "1,2,2,3,3,4" --report
    """
    try:
        original_count = len(values)
        result = preprocessing.get_unique(values)
        click.echo(",".join(str(v) for v in result))

        if report:
            unique_count = len(result)
            duplicates_removed = original_count - unique_count
            dedup_rate = duplicates_removed / original_count if original_count > 0 else 0
            click.echo(f"Removed {duplicates_removed} duplicates ({dedup_rate:.1%} reduction)")

    except Exception as exc:
        handle_processing_error("unique", exc)


if __name__ == "__main__":
    cli()
