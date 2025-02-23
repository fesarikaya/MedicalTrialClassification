import logging
import sys
from datetime import datetime
from pathlib import Path
from pythonjsonlogger import jsonlogger


def setup_logger(name: str = __name__) -> logging.Logger:
    """Configure and return a JSON logger with custom formatting"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Custom JSON formatter
    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)

            # Add timestamp
            if not log_record.get('timestamp'):
                log_record['timestamp'] = datetime.utcnow().isoformat()

            # Add log level
            if log_record.get('level'):
                log_record['level'] = log_record['level'].upper()
            else:
                log_record['level'] = record.levelname

            # Add module info
            log_record['module'] = record.module
            log_record['function'] = record.funcName
            log_record['line'] = record.lineno

    # Create formatters
    json_formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(module)s %(function)s %(line)s %(message)s'
    )

    # Console handler (JSON format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)

    # File handler (JSON format)
    file_handler = logging.FileHandler(
        log_dir / f"{datetime.now().strftime('%Y-%m-%d')}_application.log"
    )
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)

    return logger


# Example usage
if __name__ == "__main__":
    # Test the logger
    logger = setup_logger("test_logger")

    # Log some test messages
    logger.info("Info message", extra={"user": "test_user", "action": "login"})
    logger.warning("Warning message", extra={"user": "test_user", "status": "retry"})
    logger.error("Error message", extra={"user": "test_user", "error_code": 500})

    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.exception("Exception occurred", extra={"error_type": type(e).__name__})
