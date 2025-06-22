"""
Logger Configuration for Smart Form Fill API
Customizes loguru format to be more readable with clean time format
"""

import sys
from loguru import logger


def configure_logger():
    """Configure loguru logger with clean, short time format"""
    # Remove default handler
    logger.remove()
    
    # Add custom handler with clean HH:MM format
    logger.add(
        sys.stderr,
        format="<green>{time:HH:MM}</green> | <level>{level: <4}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    return logger


# Configure logger when module is imported
configure_logger()
