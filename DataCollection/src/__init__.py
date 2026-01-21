"""
Synthetic Data Generation Pipeline
AI/ML Pipeline Workshop - CyVerse

This package provides modules for generating synthetic social movement images
using Google Gemini API, integrated with Atropia, World Bank, and social media data.
"""

__version__ = "1.0.0"
__author__ = "Laura Dozal"
__license__ = "GNU GPL v3"

# Core modules
from . import config
from . import gemini_client
from . import data_loader
from . import prompt_builder
from . import output_handler
from . import validation

__all__ = [
    "config",
    "gemini_client",
    "data_loader",
    "prompt_builder",
    "output_handler",
    "validation",
]
