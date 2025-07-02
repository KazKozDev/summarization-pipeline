"""
Feature extraction module for text summarization.

This module provides the `FeatureExtractor` class which calculates various features
from text that are useful for extractive text summarization. These features include:
- TF-IDF scores for term importance
- Position-based scores to prioritize sentences at the beginning/end of documents
- Length-based scores to filter very short or very long sentences
- Title similarity scores to find sentences similar to the document title
- Keyword-based scores to highlight sentences containing important terms
- Named entity recognition to identify important entities in the text

Example usage:
    from auto_summarizer.core.feature_extractor import FeatureExtractor
    from auto_summarizer.core.preprocessor import DocumentPreprocessor
    
    # Example document
    document = ("Natural language processing (NLP) is a subfield of linguistics, "
                "computer science, and artificial intelligence concerned with the "
                "interactions between computers and human language. "
                "It is used to apply machine learning algorithms to text and speech.")
    
    # Preprocess the document
    preprocessor = DocumentPreprocessor()
    processed = preprocessor.process_document(document)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        sentences=processed['sentences'],
        processed_sentences=processed['processed_sentences']
    )
    
    # Extract all features
    features = extractor.extract_all_features(
        title_tokens=processed['processed_sentences'][0],  # Using first sentence as title
        keywords=['NLP', 'machine learning', 'artificial intelligence'],
        named_entities=processed['named_entities']
    )
    
    # Combine features with custom weights
    combined_scores = FeatureExtractor.combine_features(
        features,
        weights={
            'tfidf': 0.4,
            'position': 0.2,
            'length': 0.1,
            'title_similarity': 0.2,
            'keyword': 0.2,
            'named_entity': 0.1
        }
    )
    
    # Get top 2 sentences based on combined scores
    top_indices = np.argsort(combined_scores)[-2:][::-1]
    summary = ' '.join([processed['sentences'][i] for i in sorted(top_indices)])
    print(summary)
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import math

class FeatureExtractor:
    """Extracts and calculates features from text for extractive summarization.
    
    This class provides methods to compute various features for sentences in a document,
    which can be combined to score and rank sentences for extractive summarization.
    
    Features include:
        - TF-IDF scores for term importance
        - Position-based scores (higher for beginning/end of document)
        - Length-based scores (penalizing very short or very long sentences)
        - Title similarity scores (comparing to document title)
        - Keyword-based scores (for important terms)
        - Named entity recognition scores
    
    The class is designed to work with preprocessed text where sentences have been
    tokenized, cleaned, and normalized.
    
    Example:
        >>> from auto_summarizer.core.preprocessor import DocumentPreprocessor
        >>> preprocessor = DocumentPreprocessor()
        >>> doc = ("Natural language processing (NLP) is a subfield of linguistics, "
        ...        "computer science, and artificial intelligence concerned with the "
        ...        "interactions between computers and human language. It is used to apply "
        ...        "machine learning algorithms to text and speech.")
        >>> processed = preprocessor.process_document(doc)
        >>> extractor = FeatureExtractor(processed['sentences'], processed['processed_sentences'])
        >>> features = extractor.extract_all_features(
        ...     title_tokens=processed['processed_sentences'][0],
        ...     keywords=['NLP', 'machine learning', 'artificial intelligence'],
        ...     named_entities=processed['named_entities']
        ... )
        >>> scores = FeatureExtractor.combine_features(features)
    """
    
    def __init__(self, sentences: List[str], processed_sentences: List[List[str]]):
        """Initialize the feature extractor with sentences and their processed versions.
        
        Args:
            sentences: List of original sentence strings from the document
            processed_sentences: List of preprocessed token lists for each sentence,
                              where each token has been lowercased, lemmatized,
                              and had stopwords/punctuation removed
        
        Raises:
            ValueError: If the number of sentences doesn't match the number of processed sentences
        """
        if len(sentences) != len(processed_sentences):
            raise ValueError("Number of sentences must match number of processed sentences")
            
        self.sentences = sentences
        self.processed_sentences = processed_sentences
        self.num_sentences = len(sentences)
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False,
                                              token_pattern=None)
        self._extract_vocabulary()
        
    def _extract_vocabulary(self):
        """Extract vocabulary and calculate document frequency.
        
        This builds a set of unique terms from all processed sentences to be used
        for feature extraction and analysis.
        """
        self.vocab = set()
        self.doc_freq = defaultdict(int)
        
        for tokens in self.processed_sentences:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freq[token] += 1
            self.vocab.update(unique_tokens)
            
    def calculate_tfidf_scores(self) -> np.ndarray:
        """Calculate TF-IDF scores for each sentence in the document.
        
        This method computes the mean TF-IDF score for each sentence, which represents
        the average importance of terms in the sentence relative to the entire document.
        Scores are normalized to the range [0, 1].
        
        Returns:
            np.ndarray: Array of TF-IDF scores, one per sentence, with values in [0, 1]
            
        Example:
            >>> extractor = FeatureExtractor(sentences, processed_sentences)
            >>> tfidf_scores = extractor.calculate_tfidf_scores()
            >>> print(f"TF-IDF scores: {tfidf_scores}")
        """
        if not any(len(tokens) > 0 for tokens in self.processed_sentences):
            return np.zeros(self.num_sentences)
            
        try:
            # Fit TF-IDF on all sentences
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                [' '.join(tokens) for tokens in self.processed_sentences]
            ).toarray()
            
            # Use mean TF-IDF score per sentence for better normalization
            sentence_scores = tfidf_matrix.mean(axis=1)
            
            # Normalize to [0, 1]
            max_score = np.max(sentence_scores)
            if max_score > 0:
                sentence_scores = sentence_scores / max_score
                
            return sentence_scores
            
        except ValueError:
            return np.zeros(self.num_sentences)
    
    def calculate_position_scores(self, position_weight: float = 0.2) -> np.ndarray:
        """Calculate position-based scores for sentences.
        
        Args:
            position_weight: Weight for position scoring (0-1)
            
        Returns:
            numpy.ndarray: Position scores for each sentence
        """
        if not self.sentences:
            return np.array([])
            
        scores = np.zeros(self.num_sentences)
        
        if self.num_sentences == 1:
            scores[0] = position_weight
            return scores
            
        # Higher scores for sentences at the beginning and end of the document
        for i in range(self.num_sentences):
            # Normalized position (0 to 1)
            pos = i / (self.num_sentences - 1)
            # Create a U-shaped score distribution
            # Higher scores at the edges (beginning and end)
            if pos <= 0.5:
                scores[i] = position_weight * (1 - 2 * pos)
            else:
                scores[i] = position_weight * (2 * pos - 1)
            
        return scores
    
    def calculate_length_scores(self, min_length: int = 5, max_length: int = 30) -> np.ndarray:
        """Calculate scores based on sentence length.
        
        Args:
            min_length: Minimum desirable sentence length
            max_length: Maximum desirable sentence length
            
        Returns:
            numpy.ndarray: Length-based scores for each sentence
        """
        if not self.processed_sentences:
            return np.array([])
            
        scores = np.zeros(self.num_sentences)
        
        for i, tokens in enumerate(self.processed_sentences):
            length = len(tokens)
            
            # Ideal length range
            if min_length <= length <= max_length:
                scores[i] = 1.0
            # Too short
            elif length < min_length:
                scores[i] = 0.1 + 0.9 * (length / min_length)
            # Too long
            else:
                scores[i] = max(0.1, 1.0 - 0.9 * ((length - max_length) / (2 * max_length)))
                
        return scores
    
    def calculate_title_similarity(self, title_tokens: List[str] = None) -> np.ndarray:
        """Calculate similarity scores between sentences and title.
        
        Args:
            title_tokens: List of tokens in the title
            
        Returns:
            numpy.ndarray: Title similarity scores for each sentence
        """
        if not title_tokens or not self.processed_sentences:
            return np.zeros(self.num_sentences)
            
        title_set = set(title_tokens)
        scores = np.zeros(self.num_sentences)
        
        for i, sent_tokens in enumerate(self.processed_sentences):
            if not sent_tokens:
                continue
                
            # Calculate Jaccard similarity
            sent_set = set(sent_tokens)
            intersection = len(title_set.intersection(sent_set))
            union = len(title_set.union(sent_set))
            
            scores[i] = intersection / union if union > 0 else 0
            
        return scores
    
    def calculate_keyword_scores(self, keywords: List[str] = None) -> np.ndarray:
        """Calculate scores based on presence of keywords.
        
        Args:
            keywords: List of keywords to look for
            
        Returns:
            numpy.ndarray: Keyword-based scores for each sentence
        """
        if not keywords or not self.processed_sentences:
            return np.zeros(self.num_sentences)
            
        keyword_set = set(keywords)
        scores = np.zeros(self.num_sentences)
        
        for i, sent_tokens in enumerate(self.processed_sentences):
            if not sent_tokens:
                continue
                
            # Count keyword matches
            matches = sum(1 for token in sent_tokens if token in keyword_set)
            # Score is proportional to the number of matches, but also give some score for any match
            if matches > 0:
                # At least 0.3 for any match, up to 1.0 for all keywords matched
                scores[i] = 0.3 + 0.7 * (matches / len(keywords))
            
        return scores
    
    def calculate_named_entity_scores(self, named_entities: List[Tuple[str, str]]) -> np.ndarray:
        """Calculate scores based on named entities.
        
        Args:
            named_entities: List of (entity, entity_type) tuples
            
        Returns:
            numpy.ndarray: Named entity scores for each sentence
        """
        if not named_entities or not self.sentences:
            return np.zeros(self.num_sentences)
            
        # Create a set of named entities for faster lookup
        entity_set = {ent.lower() for ent, _ in named_entities}
        
        scores = np.zeros(self.num_sentences)
        
        for i, sentence in enumerate(self.sentences):
            # Simple count of named entities in the sentence
            count = sum(1 for ent in entity_set if ent in sentence.lower())
            scores[i] = min(1.0, count / 3)  # Cap the score
            
        return scores
    
    def extract_all_features(self, 
                           title_tokens: List[str] = None, 
                           keywords: List[str] = None,
                           named_entities: List[Tuple[str, str]] = None) -> Dict[str, np.ndarray]:
        """Extract all features and return them in a dictionary.
        
        Args:
            title_tokens: List of tokens in the title
            keywords: List of important keywords
            named_entities: List of (entity, entity_type) tuples
            
        Returns:
            Dictionary mapping feature names to score arrays
        """
        features = {
            'tfidf': self.calculate_tfidf_scores(),
            'position': self.calculate_position_scores(),
            'length': self.calculate_length_scores(),
            'title_similarity': self.calculate_title_similarity(title_tokens),
            'keyword': self.calculate_keyword_scores(keywords),
            'named_entity': self.calculate_named_entity_scores(named_entities)
        }
        
        return features
    
    @staticmethod
    def combine_features(features: Dict[str, np.ndarray], 
                        weights: Dict[str, float] = None) -> np.ndarray:
        """
        Combine multiple feature scores using weighted average.
        
        Args:
            features: Dictionary of feature name to score array
            weights: Dictionary of feature weights (must sum to 1.0)
            
        Returns:
            Combined scores for each sentence
        """
        if not features:
            return np.array([])
            
        # Default weights if not provided
        if weights is None:
            weights = {
                'tfidf': 0.4,
                'position': 0.2,
                'length': 0.1,
                'title_similarity': 0.1,
                'keyword': 0.1,
                'named_entity': 0.1
            }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = 1.0
        
        # Initialize combined scores
        num_sentences = len(next(iter(features.values())))
        combined_scores = np.zeros(num_sentences)
        
        # Calculate weighted sum
        for feature, score_array in features.items():
            if feature in weights and len(score_array) == num_sentences:
                weight = weights[feature] / total_weight
                combined_scores += weight * score_array
                
        return combined_scores
