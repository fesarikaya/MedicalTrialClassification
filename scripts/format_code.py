import subprocess
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def format_python_files():
    """Format Python files using Black."""
    try:
        # First check what would be reformatted
        logger.info("Checking files that need formatting...")
        check_result = subprocess.run(
            ["black", "--check", "."],
            capture_output=True,
            text=True
        )

        if check_result.returncode == 0:
            logger.info("All Python files are properly formatted!")
            return

        # Apply formatting
        logger.info("Applying Black formatting...")
        format_result = subprocess.run(
            ["black", "."],
            capture_output=True,
            text=True
        )

        if format_result.returncode == 0:
            logger.info("Formatting completed successfully!")
        else:
            logger.error(f"Formatting failed: {format_result.stderr}")

    except Exception as e:
        logger.exception("Error during formatting", extra={"error": str(e)})


if __name__ == "__main__":
    format_python_files()