"""Transformer-based text summarization models.

This package provides implementations of various transformer-based models
for text summarization. It includes both abstractive and extractive approaches
using state-of-the-art transformer architectures.

Available Models:
    - BART (Bidirectional and Auto-Regressive Transformers)
    - T5 (Text-to-Text Transfer Transformer) - Coming soon
    - Pegasus - Coming soon

Example:
    >>> from auto_summarizer.models.transformers import BARTSummarizer
    >>> 
    >>> # Initialize the BART summarizer
    >>> summarizer = BARTSummarizer("facebook/bart-large-cnn")
    >>> 
    >>> # Generate a summary
    >>> text = ("The tower is 324 metres (1,063 ft) tall, about the same height "
    ...         "as an 81-storey building, and the tallest structure in Paris. "
    ...         "Its base is square, measuring 125 metres (410 ft) on each side. "
    ...         "During its construction, the Eiffel Tower surpassed the Washington "
    ...         "Monument to become the tallest man-made structure in the world, "
    ...         "a title it held for 41 years until the Chrysler Building in New "
    ...         "York City was finished in 1930.")
    >>> summary = summarizer(text)
    >>> print(summary)
"""

from .base_summarizer import BaseTransformerSummarizer
from .bart_summarizer import BARTSummarizer

__all__ = [
    'BaseTransformerSummarizer',
    'BARTSummarizer',
]
