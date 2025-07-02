"""
TextRank implementation for extractive text summarization.
Based on the PageRank algorithm for ranking sentences.
"""
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

class TextRankSummarizer:
    """
    TextRank algorithm for extractive summarization.
    Implements a graph-based ranking model for text.
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.3,
                 damping_factor: float = 0.85,
                 max_iter: int = 100,
                 tolerance: float = 1e-6):
        """
        Initialize the TextRank summarizer.
        
        Args:
            similarity_threshold: Minimum similarity between sentences to create an edge
            damping_factor: Damping factor for PageRank (0.85 is standard)
            max_iter: Maximum number of iterations for convergence
            tolerance: Convergence threshold
        """
        self.similarity_threshold = similarity_threshold
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.tolerance = tolerance
    
    def _sentence_similarity(self, sent1: List[str], sent2: List[str], 
                           word_embeddings: Dict[str, np.ndarray] = None) -> float:
        """
        Calculate similarity between two sentences.
        
        Args:
            sent1: First sentence as list of tokens
            sent2: Second sentence as list of tokens
            word_embeddings: Optional word embeddings for better similarity
            
        Returns:
            Similarity score between 0 and 1
        """
        if not sent1 or not sent2:
            return 0.0
            
        # Convert sentences to sets for Jaccard similarity
        set1 = set(sent1)
        set2 = set(sent2)
        
        # Jaccard similarity as fallback
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def _build_similarity_matrix(self, sentences: List[List[str]]) -> np.ndarray:
        """
        Build a similarity matrix where each cell represents similarity between sentences.
        
        Args:
            sentences: List of tokenized sentences
            
        Returns:
            Similarity matrix (n x n) where n is number of sentences
        """
        num_sentences = len(sentences)
        similarity_matrix = np.zeros((num_sentences, num_sentences))
        
        for i in range(num_sentences):
            for j in range(i, num_sentences):
                if i == j:
                    continue
                    
                similarity = self._sentence_similarity(sentences[i], sentences[j])
                
                # Only keep similarities above threshold
                if similarity > self.similarity_threshold:
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
        
        # Normalize the matrix
        for i in range(num_sentences):
            row_sum = similarity_matrix[i].sum()
            if row_sum > 0:
                similarity_matrix[i] /= row_sum
        
        return similarity_matrix
    
    def _apply_pagerank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the PageRank algorithm to the similarity matrix.
        
        Args:
            similarity_matrix: Normalized similarity matrix
            
        Returns:
            PageRank scores for each sentence
        """
        num_sentences = similarity_matrix.shape[0]
        
        # Initialize all scores to 1/n
        scores = np.ones(num_sentences) / num_sentences
        
        # Power iteration
        for _ in range(self.max_iter):
            prev_scores = np.copy(scores)
            
            # Update scores
            scores = (1 - self.damping_factor) / num_sentences + \
                    self.damping_factor * np.dot(similarity_matrix.T, scores)
            
            # Check for convergence
            if np.sum(np.abs(scores - prev_scores)) < self.tolerance:
                break
        
        return scores
    
    def rank_sentences(self, sentences: List[str], 
                      processed_sentences: List[List[str]]) -> List[Tuple[int, float]]:
        """
        Rank sentences using TextRank algorithm.
        
        Args:
            sentences: List of original sentences
            processed_sentences: List of preprocessed token lists
            
        Returns:
            List of (sentence_index, score) tuples, sorted by score descending
        """
        if not sentences or not processed_sentences:
            return []
            
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(processed_sentences)
        
        # Apply PageRank
        scores = self._apply_pagerank(similarity_matrix)
        
        # Rank sentences by score
        ranked_sentences = [(i, score) for i, score in enumerate(scores)]
        ranked_sentences.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_sentences
    
    def summarize(self, sentences: List[str], 
                 processed_sentences: List[List[str]], 
                 top_n: int = 5,
                 summary_ratio: float = None) -> Tuple[List[str], List[float]]:
        """
        Generate a summary using TextRank.
        
        Args:
            sentences: List of original sentences
            processed_sentences: List of preprocessed token lists
            top_n: Number of top sentences to include in summary
            summary_ratio: Ratio of sentences to include (alternative to top_n)
            
        Returns:
            Tuple of (summary_sentences, sentence_scores)
        """
        if not sentences or not processed_sentences:
            return [], []
            
        # Determine number of sentences to include
        if summary_ratio is not None:
            top_n = max(1, int(len(sentences) * summary_ratio))
        else:
            top_n = min(top_n, len(sentences))
        
        # Rank sentences
        ranked_sentences = self.rank_sentences(sentences, processed_sentences)
        
        # Sort by original position for coherence
        top_indices = [idx for idx, _ in ranked_sentences[:top_n]]
        top_indices.sort()
        
        # Get scores for all sentences
        all_scores = [0.0] * len(sentences)
        for idx, score in ranked_sentences:
            all_scores[idx] = score
        
        # Extract summary sentences
        summary = [sentences[idx] for idx in top_indices]
        
        return summary, all_scores
