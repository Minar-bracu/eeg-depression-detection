"""
EEG Depression Detection Pipeline
Signal processing and machine learning for depression biomarker identification from EEG signals.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from . import preprocessing
from . import feature_extraction
from . import models

__all__ = ['preprocessing', 'feature_extraction', 'models']
