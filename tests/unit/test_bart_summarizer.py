"""Unit tests for BARTSummarizer.

These tests verify that the summarizer can load the model and generate summaries.
We use a lightweight distilled BART model to keep the test fast and reduce memory
requirements.
"""
import pytest

pytest.importorskip("transformers")

from auto_summarizer.models.transformers import BARTSummarizer


@pytest.fixture(scope="module")
def summarizer():
    """Instantiate the BART summarizer with a small model for testing."""
    model_name = "sshleifer/distilbart-cnn-12-6"
    return BARTSummarizer(model_name, device="cpu")


def test_generate_summary(summarizer):
    """Test that the summarizer generates a non-empty summary."""
    text = (
        "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey "
        "building, and the tallest structure in Paris. Its base is square, measuring 125 "
        "metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed "
        "the Washington Monument to become the tallest man-made structure in the world, a "
        "title it held for 41 years until the Chrysler Building in New York City was "
        "finished in 1930."
    )
    summary = summarizer(text, max_length=60, min_length=10, num_beams=2)
    assert isinstance(summary, str) and len(summary) > 0


def test_batch_summary(summarizer):
    """Test batch summarization returns correct number of summaries."""
    texts = [
        "Natural language processing (NLP) is a subfield of linguistics, computer science, "
        "and artificial intelligence concerned with the interactions between computers and "
        "human language.",
        "Machine learning algorithms are used to automate analytical model building and rely "
        "on patterns and inference.",
    ]
    summaries = summarizer.generate_summary_batch(texts, batch_size=2, max_length=40, min_length=5, num_beams=2)
    assert isinstance(summaries, list) and len(summaries) == len(texts)
    for s in summaries:
        assert isinstance(s, str) and len(s) > 0
