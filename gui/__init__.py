"""
NMR Reader GUI Module
"""

from .video import VideoTab
from .main_window import EnhancedBrukerReader
from .selection_tab import SelectionTab
from .analysis_tab import AnalysisTab
from .batch_tab import BatchTab
from .cluster_tab import ClusterTab

__version__ = "1.0.0"
__author__ = "JH"

__all__ = [
    'VideoTab',
    'EnhancedBrukerReader', 
    'SelectionTab',
    'AnalysisTab',
    'BatchTab',
    'ClusterTab',  # Add this
]