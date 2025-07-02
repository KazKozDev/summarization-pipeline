"""
Unit tests for the FeatureExtractor class.
"""
import pytest
import numpy as np
from auto_summarizer.core.feature_extractor import FeatureExtractor

def test_tfidf_scores():
    """Test TF-IDF score calculation."""
    # Using more distinct sentences to ensure TF-IDF works as expected
    sentences = [
        "a simple test sentence",
        "a simple test sentence with some additional words",
        "a completely different sentence with unique terms"
    ]
    processed_sentences = [s.split() for s in sentences]
    
    extractor = FeatureExtractor(sentences, processed_sentences)
    scores = extractor.calculate_tfidf_scores()
    
    # Should return one score per sentence
    assert len(scores) == len(sentences)
    # All scores should be between 0 and 1
    assert all(0 <= score <= 1 for score in scores)
    # The most specific sentence should have the highest score
    # The third sentence has the most unique terms
    assert scores[2] == max(scores)

def test_position_scores():
    """Test position-based scoring."""
    sentences = ["First", "Second", "Third", "Fourth", "Fifth"]
    processed_sentences = [[s.lower()] for s in sentences]
    
    extractor = FeatureExtractor(sentences, processed_sentences)
    scores = extractor.calculate_position_scores()
    
    # Should return one score per sentence
    assert len(scores) == len(sentences)
    # First and last sentences should have highest scores
    assert scores[0] > scores[2]  # First > Middle
    assert scores[-1] > scores[2]  # Last > Middle
    # Scores should be symmetric
    assert abs(scores[0] - scores[-1]) < 0.1  # First ≈ Last

def test_length_scores():
    """Test length-based scoring."""
    sentences = ["short", "a bit longer sentence", "this is a very very long sentence that should be penalized"]
    processed_sentences = [s.split() for s in sentences]
    
    extractor = FeatureExtractor(sentences, processed_sentences)
    scores = extractor.calculate_length_scores(min_length=3, max_length=5)
    
    # Should return one score per sentence
    assert len(scores) == len(sentences)
    # Middle sentence should have highest score (ideal length)
    assert scores[1] == max(scores)
    # Very short and very long sentences should be penalized
    assert scores[1] > scores[0]  # Ideal > Too short
    assert scores[1] > scores[2]  # Ideal > Too long

def test_title_similarity():
    """Test title similarity scoring."""
    sentences = [
        "this is about dogs",
        "this is about cats",
        "completely different topic"
    ]
    processed_sentences = [s.split() for s in sentences]
    title_tokens = "dogs and cats".split()
    
    extractor = FeatureExtractor(sentences, processed_sentences)
    scores = extractor.calculate_title_similarity(title_tokens)
    
    # Should return one score per sentence
    assert len(scores) == len(sentences)
    # First two sentences should have higher scores than the third
    assert scores[0] > 0
    assert scores[1] > 0
    assert scores[2] == 0  # No overlap with title
    # First sentence should have higher score than second (more matching terms)
    assert scores[0] >= scores[1]

def test_keyword_scores():
    """Test keyword-based scoring."""
    sentences = [
        "this contains important and urgent information",  # Both keywords
        "this is just regular information with important details",  # One keyword
        "nothing special here"  # No keywords
    ]
    processed_sentences = [s.split() for s in sentences]
    keywords = ["important", "urgent"]
    
    extractor = FeatureExtractor(sentences, processed_sentences)
    scores = extractor.calculate_keyword_scores(keywords)
    
    # Should return one score per sentence
    assert len(scores) == len(sentences)
    # First sentence should have highest score (contains both keywords)
    assert scores[0] == max(scores)
    # Second sentence should have some score (contains one keyword)
    assert scores[1] > 0, f"Expected score > 0 for sentence with one keyword, got {scores[1]}"
    # Third sentence should have zero score (no keywords)
    assert scores[2] == 0, f"Expected score 0 for sentence with no keywords, got {scores[2]}"

def test_named_entity_scores():
    """Test named entity scoring."""
    sentences = [
        "John Smith works at Google in California.",
        "This is a test sentence.",
        "Microsoft is a technology company."
    ]
    named_entities = [
        ("John Smith", "PERSON"),
        ("Google", "ORG"),
        ("California", "GPE"),
        ("Microsoft", "ORG")
    ]
    
    extractor = FeatureExtractor(sentences, [s.split() for s in sentences])
    scores = extractor.calculate_named_entity_scores(named_entities)
    
    # Should return one score per sentence
    assert len(scores) == len(sentences)
    # First sentence should have highest score (3 entities)
    assert scores[0] == max(scores)
    # Second sentence should have zero score (no entities)
    assert scores[1] == 0
    # Third sentence should have some score (1 entity)
    assert scores[2] > 0

def test_combine_features():
    """Test feature combination with weights."""
    features = {
        'feature1': np.array([0.5, 0.8, 0.3]),
        'feature2': np.array([0.2, 0.5, 0.9]),
        'feature3': np.array([0.1, 0.3, 0.7])
    }
    weights = {
        'feature1': 0.5,
        'feature2': 0.3,
        'feature3': 0.2
    }
    
    combined = FeatureExtractor.combine_features(features, weights)
    
    # Should return one score per item
    assert len(combined) == 3
    # Should be a weighted combination
    assert combined[0] == pytest.approx(0.5*0.5 + 0.2*0.3 + 0.1*0.2)
    assert combined[1] == pytest.approx(0.8*0.5 + 0.5*0.3 + 0.3*0.2)
    assert combined[2] == pytest.approx(0.3*0.5 + 0.9*0.3 + 0.7*0.2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
