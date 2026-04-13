"""
ragtriage: From RAG metrics to action items

Turn your RAG system's failures into a prioritized content backlog.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .eval import evaluate_rag
from .analyze import analyze_gaps

__all__ = ["evaluate_rag", "analyze_gaps"]
