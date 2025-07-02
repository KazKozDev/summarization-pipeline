"""
Integration tests for the main Summarizer class.
"""
import pytest
from auto_summarizer.core.summarizer import Summarizer

def test_summarizer_basic():
    """Test basic summarization functionality."""
    # Initialize summarizer
    summarizer = Summarizer()
    
    # Test text
    text = """
    Artificial intelligence is transforming industries across the globe. 
    From healthcare to finance, AI applications are becoming increasingly common. 
    Many companies are investing heavily in AI research and development. 
    The potential benefits of AI are enormous, but there are also significant challenges. 
    Ethical considerations must be taken into account when developing AI systems.
    """
    
    # Test different summarization methods
    for method in ['textrank', 'features', 'combined']:
        result = summarizer.summarize(
            text=text,
            method=method,
            top_n=2
        )
        
        # Check basic structure of result
        assert 'summary' in result
        assert 'scores' in result
        assert 'method' in result
        assert result['method'] == method
        
        # Check summary content
        assert isinstance(result['summary'], list)
        assert len(result['summary']) > 0
        assert all(isinstance(s, str) for s in result['summary'])
        
        # Check scores
        assert len(result['scores']) > 0
        assert all(isinstance(s, (int, float)) for s in result['scores'])

def test_summarizer_edge_cases():
    """Test summarizer with edge cases."""
    summarizer = Summarizer()
    
    # Test with empty text
    empty_result = summarizer.summarize("", method='textrank')
    assert empty_result['summary'] == []
    assert empty_result['scores'] == []
    
    # Test with very short text
    short_result = summarizer.summarize("Short text.", method='textrank')
    assert len(short_result['summary']) == 1
    assert short_result['summary'][0] == "Short text."
    
    # Test with invalid method (should default to combined)
    invalid_result = summarizer.summarize("Test text.", method='invalid')
    assert 'error' in invalid_result

def test_summarizer_different_lengths():
    """Test summarizer with different text lengths."""
    summarizer = Summarizer()
    
    # Test with a single sentence
    single_sentence = "This is a single sentence."
    result = summarizer.summarize(single_sentence, method='combined')
    assert len(result['summary']) == 1
    assert result['summary'][0] == single_sentence
    
    # Test with a very long text (more than 1000 words)
    long_text = "This is a test. " * 100
    result = summarizer.summarize(long_text, method='combined', top_n=5)
    assert 1 <= len(result['summary']) <= 5

def test_summarizer_evaluation():
    """Test summary evaluation functionality."""
    summarizer = Summarizer()
    
    # Test text and reference summary
    text = """
    Artificial intelligence is transforming industries across the globe. 
    From healthcare to finance, AI applications are becoming increasingly common. 
    Many companies are investing heavily in AI research and development. 
    The potential benefits of AI are enormous, but there are also significant challenges. 
    Ethical considerations must be taken into account when developing AI systems.
    """
    reference = [
        "Artificial intelligence is transforming industries across the globe.",
        "Many companies are investing heavily in AI research and development."
    ]
    
    # Generate summary
    result = summarizer.summarize(text, method='combined', top_n=2)
    
    # Evaluate
    metrics = summarizer.evaluate_summary(result['summary'], reference)
    
    # Check metrics
    assert 'rouge1' in metrics
    assert 'rouge2' in metrics
    assert 'rougeL' in metrics
    assert 'bleu' in metrics
    
    # Scores should be between 0 and 1
    for score in metrics.values():
        assert 0 <= score <= 1

def test_summarizer_with_preprocessing():
    """Test summarizer with preprocessed input."""
    summarizer = Summarizer()
    
    # Preprocess the text first
    text = """
    Artificial intelligence is transforming industries across the globe. 
    From healthcare to finance, AI applications are becoming increasingly common.
    """
    processed_data = summarizer.preprocess(text)
    
    # Test with preprocessed data
    result = summarizer.summarize_with_textrank(processed_data, top_n=1)
    
    # Check results
    assert len(result[0]) == 1  # One sentence summary
    assert len(result[1]) == len(processed_data['sentences'])  # Scores for all sentences

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
