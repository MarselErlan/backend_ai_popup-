"""
Logger Configuration for Smart Form Fill API
Customizes loguru format to be more readable
"""

import sys
from loguru import logger


def configure_logger():
    """Configure loguru logger with custom format"""
    # Remove default handler
    logger.remove()
    
    # Add custom handler with shorter format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    return logger


# Configure logger when module is imported
configure_logger()
