"""
Main summarization module that integrates all components.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .preprocessor import DocumentPreprocessor
from .feature_extractor import FeatureExtractor
from ..models.textrank import TextRankSummarizer
from loguru import logger

class Summarizer:
    """
    Main summarization class that integrates preprocessing, feature extraction,
    and multiple summarization algorithms.
    """
    
    def __init__(self, language: str = "english"):
        """
        Initialize the summarizer.
        
        Args:
            language: Language of the text (default: "english")
        """
        self.language = language
        self.preprocessor = DocumentPreprocessor(language)
        self.textrank = TextRankSummarizer()
        
    def preprocess(self, text: str) -> Dict[str, Any]:
        """
        Preprocess the input text.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Dictionary containing preprocessed data
        """
        return self.preprocessor.process_document(text)
    
    def extract_features(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from preprocessed data.
        
        Args:
            processed_data: Output from preprocess()
            
        Returns:
            Dictionary containing extracted features
        """
        if not processed_data or 'processed_sentences' not in processed_data:
            return {}
            
        # Initialize feature extractor
        extractor = FeatureExtractor(
            sentences=processed_data['sentences'],
            processed_sentences=processed_data['processed_sentences']
        )
        
        # Extract all features
        features = extractor.extract_all_features(
            title_tokens=processed_data.get('title_tokens', []),
            keywords=processed_data.get('keywords', []),
            named_entities=processed_data.get('named_entities', [])
        )
        
        # Combine features with weights
        combined_scores = extractor.combine_features(features)
        
        return {
            'features': features,
            'combined_scores': combined_scores,
            'feature_extractor': extractor
        }
    
    def summarize_with_textrank(self, 
                              processed_data: Dict[str, Any],
                              top_n: int = 5,
                              summary_ratio: float = None) -> Tuple[List[str], List[float]]:
        """
        Generate a summary using TextRank algorithm.
        
        Args:
            processed_data: Output from preprocess()
            top_n: Number of top sentences to include
            summary_ratio: Ratio of sentences to include (alternative to top_n)
            
        Returns:
            Tuple of (summary_sentences, sentence_scores)
        """
        if not processed_data or 'sentences' not in processed_data:
            return [], []
            
        return self.textrank.summarize(
            sentences=processed_data['sentences'],
            processed_sentences=processed_data['processed_sentences'],
            top_n=top_n,
            summary_ratio=summary_ratio
        )
    
    def summarize_with_features(self,
                              processed_data: Dict[str, Any],
                              feature_data: Dict[str, Any],
                              top_n: int = 5,
                              summary_ratio: float = None) -> Tuple[List[str], List[float]]:
        """
        Generate a summary using feature-based scoring.
        
        Args:
            processed_data: Output from preprocess()
            feature_data: Output from extract_features()
            top_n: Number of top sentences to include
            summary_ratio: Ratio of sentences to include (alternative to top_n)
            
        Returns:
            Tuple of (summary_sentences, sentence_scores)
        """
        if not processed_data or 'sentences' not in processed_data:
            return [], []
            
        if not feature_data or 'combined_scores' not in feature_data:
            return [], []
            
        sentences = processed_data['sentences']
        scores = feature_data['combined_scores']
        
        # Determine number of sentences to include
        if summary_ratio is not None:
            top_n = max(1, int(len(sentences) * summary_ratio))
        else:
            top_n = min(top_n, len(sentences))
        
        # Get top N sentence indices by score
        top_indices = np.argsort(scores)[-top_n:][::-1]
        top_indices_sorted = sorted(top_indices)  # Sort by original position
        
        # Extract summary sentences
        summary = [sentences[i] for i in top_indices_sorted]
        
        return summary, scores.tolist()
    
    def summarize(self, 
                 text: str, 
                 method: str = 'combined',
                 top_n: int = 5,
                 summary_ratio: float = None) -> Dict[str, Any]:
        """
        Main summarization method.
        
        Args:
            text: Input text to summarize
            method: Summarization method ('textrank', 'features', or 'combined')
            top_n: Number of top sentences to include
            summary_ratio: Ratio of sentences to include (alternative to top_n)
            
        Returns:
            Dictionary containing summary and metadata
        """
        if not text or not isinstance(text, str):
            return {
                'summary': [],
                'scores': [],
                'method': method,
                'error': 'Invalid input text'
            }
            
        try:
            # Preprocess the text
            processed_data = self.preprocess(text)
            
            # Initialize result dictionary
            result = {
                'summary': [],
                'scores': [],
                'method': method,
                'processed_data': processed_data
            }
            
            # Extract features if needed
            if method in ['features', 'combined']:
                feature_data = self.extract_features(processed_data)
                result['feature_data'] = feature_data
            
            # Generate summaries based on method
            if method == 'textrank':
                summary, scores = self.summarize_with_textrank(
                    processed_data, top_n, summary_ratio)
                
            elif method == 'features':
                summary, scores = self.summarize_with_features(
                    processed_data, feature_data, top_n, summary_ratio)
                
            elif method == 'combined':
                # Get both summaries and combine them
                tr_summary, tr_scores = self.summarize_with_textrank(
                    processed_data, top_n * 2, summary_ratio * 2 if summary_ratio else None)
                
                if 'feature_data' not in locals():
                    feature_data = self.extract_features(processed_data)
                    result['feature_data'] = feature_data
                    
                feat_summary, feat_scores = self.summarize_with_features(
                    processed_data, feature_data, top_n * 2, 
                    summary_ratio * 2 if summary_ratio else None)
                
                # Combine the two methods
                combined_summary = list(dict.fromkeys(tr_summary + feat_summary))
                
                # Sort by original position
                summary_indices = [processed_data['sentences'].index(s) 
                                 for s in combined_summary 
                                 if s in processed_data['sentences']]
                sorted_indices = sorted(range(len(summary_indices)), 
                                      key=lambda k: summary_indices[k])
                summary = [combined_summary[i] for i in sorted_indices][:top_n]
                
                # Average the scores
                scores = [(tr_scores[i] + feat_scores[i]) / 2 
                         for i in range(len(tr_scores))]
                
            else:
                raise ValueError(f"Unknown summarization method: {method}")
            
            result['summary'] = summary
            result['scores'] = scores
            return result
            
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}", exc_info=True)
            return {
                'summary': [],
                'scores': [],
                'method': method,
                'error': str(e)
            }
    
    def evaluate_summary(self, 
                        summary: List[str], 
                        reference: List[str],
                        metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate the quality of a summary against a reference.
        
        Args:
            summary: Generated summary sentences
            reference: Reference summary sentences
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu']
            
        scores = {}
        
        # Convert to strings for evaluation
        summary_text = ' '.join(summary) if summary else ""
        reference_text = ' '.join(reference) if reference else ""
        
        # Calculate ROUGE scores if requested
        if any(m.startswith('rouge') for m in metrics):
            try:
                from rouge import Rouge
                rouge = Rouge()
                if summary_text and reference_text:
                    rouge_scores = rouge.get_scores(summary_text, reference_text)[0]
                    
                    if 'rouge1' in metrics:
                        scores['rouge1'] = rouge_scores['rouge-1']['f']
                    if 'rouge2' in metrics:
                        scores['rouge2'] = rouge_scores['rouge-2']['f']
                    if 'rougeL' in metrics:
                        scores['rougeL'] = rouge_scores['rouge-l']['f']
            except ImportError:
                logger.warning("ROUGE not installed. Install with: pip install rouge")
        
        # Calculate BLEU score if requested
        if 'bleu' in metrics and summary and reference:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothie = SmoothingFunction().method4
            
            # Tokenize sentences
            summary_tokens = [s.split() for s in summary]
            reference_tokens = [ref.split() for ref in reference]
            
            # Calculate BLEU score (averaged over sentences)
            bleu_scores = []
            for s_tokens in summary_tokens:
                bleu_scores.append(
                    sentence_bleu(
                        reference_tokens,
                        s_tokens,
                        smoothing_function=smoothie
                    )
                )
            
            scores['bleu'] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        
        return scores
