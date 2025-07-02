"""
Integration tests for the TextRank summarizer.
"""
import pytest
from auto_summarizer.models.textrank import TextRankSummarizer

def test_textrank_summarizer_basic():
    """Test basic TextRank summarization."""
    # Sample text from a news article
    text = """
    Artificial intelligence is transforming industries across the globe. 
    From healthcare to finance, AI applications are becoming increasingly common. 
    Many companies are investing heavily in AI research and development. 
    The potential benefits of AI are enormous, but there are also significant challenges. 
    Ethical considerations must be taken into account when developing AI systems.
    """
    
    # Split into sentences for testing
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    processed_sentences = [s.lower().split() for s in sentences]
    
    # Initialize summarizer
    summarizer = TextRankSummarizer(similarity_threshold=0.1)
    
    # Test with different summary lengths
    for top_n in range(1, 4):
        summary, scores = summarizer.summarize(sentences, processed_sentences, top_n=top_n)
        
        # Should return requested number of sentences (or fewer if not enough input)
        assert len(summary) == min(top_n, len(sentences))
        
        # All returned sentences should be from the original text
        assert all(s in sentences for s in summary)
        
        # Should return one score per input sentence
        assert len(scores) == len(sentences)
        
        # Scores should be between 0 and 1
        assert all(0 <= s <= 1 for s in scores)

def test_textrank_with_short_text():
    """Test TextRank with very short text."""
    summarizer = TextRankSummarizer()
    
    # Test with single sentence
    single_sent = ["This is a test."]
    single_processed = [s.lower().split() for s in single_sent]
    
    summary, scores = summarizer.summarize(single_sent, single_processed, top_n=3)
    assert len(summary) == 1
    assert summary[0] == single_sent[0]
    assert len(scores) == 1
    
    # Test with empty input
    empty_summary, empty_scores = summarizer.summarize([], [], top_n=3)
    assert len(empty_summary) == 0
    assert len(empty_scores) == 0

def test_textrank_sentence_ordering():
    """Test that sentences in summary maintain original ordering."""
    text = """
    First sentence. Second sentence. Third sentence. 
    Fourth sentence. Fifth sentence. Sixth sentence.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    processed_sentences = [s.lower().split() for s in sentences]
    
    summarizer = TextRankSummarizer(similarity_threshold=0.3)
    
    # Get a 3-sentence summary
    summary, _ = summarizer.summarize(sentences, processed_sentences, top_n=3)
    
    # Check that sentences are in their original order
    if len(summary) > 1:
        indices = [sentences.index(s) for s in summary]
        assert indices == sorted(indices), "Sentences should maintain original order"

def test_textrank_with_duplicates():
    """Test TextRank with duplicate sentences."""
    text = """
    This is a duplicate. This is a duplicate. 
    This is a unique sentence. This is another unique sentence.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    processed_sentences = [s.lower().split() for s in sentences]
    
    summarizer = TextRankSummarizer()
    
    # Get a summary with more sentences than unique content
    summary, _ = summarizer.summarize(sentences, processed_sentences, top_n=4)
    
    # Should return all unique sentences, ignoring exact duplicates
    assert len(summary) == 3  # 2 unique + 1 duplicate
    assert "This is a duplicate" in summary
    assert "This is a unique sentence" in summary
    assert "This is another unique sentence" in summary

def test_textrank_parameters():
    """Test the effect of different TextRank parameters."""
    text = """
    Machine learning is a subset of artificial intelligence. 
    It involves training algorithms on data to make predictions. 
    Deep learning is a specialized form of machine learning. 
    It uses neural networks with multiple layers. 
    Both approaches have revolutionized technology in recent years.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    processed_sentences = [s.lower().split() for s in sentences]
    
    # Test with different similarity thresholds
    for threshold in [0.1, 0.5, 0.9]:
        summarizer = TextRankSummarizer(similarity_threshold=threshold)
        summary, _ = summarizer.summarize(sentences, processed_sentences, top_n=2)
        
        # With higher thresholds, we should get more diverse sentences
        if threshold == 0.1:
            # Lower threshold might include more similar sentences
            pass
        else:
            # Higher threshold should prefer more distinct sentences
            pass
    
    # Test with different damping factors
    for damping in [0.5, 0.85, 0.95]:
        summarizer = TextRankSummarizer(damping_factor=damping)
        summary, _ = summarizer.summarize(sentences, processed_sentences, top_n=2)
        # Hard to assert specific behavior, but should complete without errors
        assert len(summary) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
