import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Global state to track logging initialization
_logging_initialized = False
_main_logger = None
_log_file_path = None


def setup_logging(console_level: str = None, file_level: str = None, log_file: str = None) -> logging.Logger:
    """Setup comprehensive logging system with singleton pattern to prevent multiple log files"""
    
    global _logging_initialized, _main_logger, _log_file_path
    
    # If logging is already initialized, return the existing logger
    if _logging_initialized and _main_logger is not None:
        return _main_logger
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get log levels from environment variables or use defaults
    if console_level is None:
        console_level = os.getenv('CONSOLE_LOG_LEVEL', 'INFO')
    
    if file_level is None:
        file_level = os.getenv('FILE_LOG_LEVEL', 'DEBUG')
    
    # Default log file name with timestamp (only if not already set)
    if log_file is None and _log_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_file_path = log_dir / f"trading_system_{timestamp}.log"
    elif log_file is not None:
        _log_file_path = Path(log_file)
    
    # Convert string log levels to logging constants
    console_numeric_level = getattr(logging, console_level.upper(), logging.INFO)
    file_numeric_level = getattr(logging, file_level.upper(), logging.DEBUG)
    
    # Set root logger to the lowest level (most verbose) to capture everything
    root_logger = logging.getLogger()
    root_logger.setLevel(min(console_numeric_level, file_numeric_level))
    
    # Clear existing handlers to prevent duplicates
    root_logger.handlers.clear()
    
    # Console formatter (clean and simple for INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File formatter (detailed for DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_numeric_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (DEBUG level)
    file_handler = logging.FileHandler(_log_file_path, encoding='utf-8')
    file_handler.setLevel(file_numeric_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Create main logger
    _main_logger = logging.getLogger("TradingSystem")
    
    # Mark logging as initialized
    _logging_initialized = True
    
    # Log the configuration
    _main_logger.info(f"Logging initialized - Console: {console_level.upper()}, File: {file_level.upper()}")
    _main_logger.debug(f"🐛 DEBUG logging active in file: {_log_file_path}")
    _main_logger.debug(f"📊 Console shows {console_level.upper()} and above, File captures {file_level.upper()} and above")
    _main_logger.debug(f"🔒 Logging system locked to prevent multiple log files")
    
    return _main_logger


def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance - ensures logging is initialized"""
    global _logging_initialized, _main_logger
    
    # If logging hasn't been initialized yet, initialize it
    if not _logging_initialized:
        setup_logging()
    
    if name is None:
        name = "TradingSystem"
    
    return logging.getLogger(name)


def reset_logging():
    """Reset logging system (for testing purposes only)"""
    global _logging_initialized, _main_logger, _log_file_path
    
    # Clear all handlers from root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Reset global state
    _logging_initialized = False
    _main_logger = None
    _log_file_path = None


def get_current_log_file() -> Optional[str]:
    """Get the current log file path"""
    global _log_file_path
    return str(_log_file_path) if _log_file_path else None


def is_logging_initialized() -> bool:
    """Check if logging has been initialized"""
    global _logging_initialized
    return _logging_initialized