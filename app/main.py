"""
AI Research Assistant - Phase 1
Main entry point for the semantic search application.
"""

import sys
import logging
from app.gui import main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)
