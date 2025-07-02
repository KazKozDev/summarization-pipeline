"""
Unit tests for the DocumentPreprocessor class.
"""
import pytest
from auto_summarizer.core.preprocessor import DocumentPreprocessor

def test_clean_text():
    """Test text cleaning functionality."""
    preprocessor = DocumentPreprocessor()
    
    # Test HTML tag removal
    assert preprocessor.clean_text("<p>Test</p>") == "test"
    
    # Test URL removal
    assert "http" not in preprocessor.clean_text("Visit https://example.com")
    
    # Test special character removal
    assert "@#$%" not in preprocessor.clean_text("Test @#$%^&*()")
    
    # Test multiple spaces and newlines
    assert "test text" == preprocessor.clean_text("test    text\n")
    
    # Test empty input
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""

def test_tokenize_sentences():
    """Test sentence tokenization."""
    preprocessor = DocumentPreprocessor()
    
    # Test basic sentence splitting
    text = "This is a test. This is another test."
    sentences = preprocessor.tokenize_sentences(text)
    assert len(sentences) == 2
    assert sentences[0].startswith("This is a test")
    
    # Test empty input
    assert preprocessor.tokenize_sentences("") == []
    assert preprocessor.tokenize_sentences(None) == []

def test_preprocess_sentence():
    """Test sentence preprocessing."""
    preprocessor = DocumentPreprocessor()
    
    # Test stopword removal and lemmatization
    processed = preprocessor.preprocess_sentence("The quick brown fox jumps over the lazy dog")
    assert "the" not in processed  # Stopword removed
    assert "jump" in processed     # Lemmatized
    
    # Test punctuation removal
    processed = preprocessor.preprocess_sentence("Hello, world!")
    assert "," not in processed
    assert "!" not in processed
    
    # Test empty input
    assert preprocessor.preprocess_sentence("") == []
    assert preprocessor.preprocess_sentence(None) == []

def test_process_document():
    """Test complete document processing."""
    preprocessor = DocumentPreprocessor()
    text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language. It is used to apply machine learning 
    algorithms to text and speech.
    """
    
    result = preprocessor.process_document(text)
    
    # Check basic structure
    assert 'original_text' in result
    assert 'cleaned_text' in result
    assert 'sentences' in result
    assert 'tokens' in result
    assert 'processed_sentences' in result
    assert 'named_entities' in result
    
    # Verify content
    assert len(result['sentences']) > 0
    assert len(result['tokens']) > 0
    assert len(result['processed_sentences']) == len(result['sentences'])
    
    # Test empty input
    empty_result = preprocessor.process_document("")
    assert empty_result['original_text'] == ""
    assert empty_result['sentences'] == []
    assert empty_result['tokens'] == []

def test_get_named_entities():
    """Test named entity recognition."""
    preprocessor = DocumentPreprocessor()
    text = "Apple is looking at buying U.K. startup for $1 billion"
    
    entities = preprocessor.get_named_entities(text)
    
    # Should at least detect "Apple" and "U.K." as entities
    assert len(entities) >= 2
    
    # Check entity types
    entity_texts = [e[0] for e in entities]
    assert "Apple" in entity_texts or "U.K." in entity_texts
    
    # Test empty input
    assert preprocessor.get_named_entities("") == []
    assert preprocessor.get_named_entities(None) == []

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
