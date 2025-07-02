"""Base class for transformer-based text summarization models.

This module provides an abstract base class for implementing transformer-based
text summarization models. It defines the common interface and functionality
that all transformer summarizers should implement.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig
)
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseTransformerSummarizer(ABC):
    """Abstract base class for transformer-based text summarization.
    
    This class provides a common interface for different transformer models
    used for text summarization. It handles model loading, text preprocessing,
    and provides a consistent API for generating summaries.
    
    Subclasses should implement the specific model loading and text processing
    logic for different transformer architectures.
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        **model_kwargs
    ):
        """Initialize the summarizer with a pre-trained model.
        
        Args:
            model_name: Name or path of the pre-trained model.
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect).
            use_auth_token: Hugging Face authentication token for private models.
            **model_kwargs: Additional keyword arguments passed to the model.
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.model_kwargs = model_kwargs
        self.use_auth_token = use_auth_token
        
        # Default generation parameters
        self.default_generation_config = {
            'max_length': 150,
            'min_length': 30,
            'length_penalty': 2.0,
            'num_beams': 4,
            'early_stopping': True,
            'no_repeat_ngram_size': 3,
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.95,
            'do_sample': True,
        }
        
        # Load the model and tokenizer
        self._load_model()
    
    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """Get the appropriate device for model inference.
        
        Args:
            device: Preferred device ('cuda', 'cpu', or None for auto-detect).
            
        Returns:
            torch.device: The selected device.
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                # Apple Silicon GPU via Metal Performance Shaders
                device = 'mps'
            else:
                device = 'cpu'
        return torch.device(device)
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the model and tokenizer.
        
        This method should be implemented by subclasses to load the specific
        model architecture and tokenizer.
        """
        pass
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess the input text before tokenization.
        
        Args:
            text: Input text to preprocess.
            
        Returns:
            str: Preprocessed text.
        """
        # Basic text cleaning can be added here
        return text.strip()
    
    def _postprocess_summary(self, summary: str) -> str:
        """Postprocess the generated summary.
        
        Args:
            summary: Generated summary to postprocess.
            
        Returns:
            str: Postprocessed summary.
        """
        # Basic postprocessing can be added here
        return summary.strip()
    
    def _chunk_text(
        self,
        text: str,
        max_chunk_size: int = 1000,
        overlap: int = 100
    ) -> List[str]:
        """Split long text into overlapping chunks.
        
        Args:
            text: Input text to chunk.
            max_chunk_size: Maximum size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.
            
        Returns:
            List[str]: List of text chunks.
        """
        if len(text) <= max_chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            # Try to find a sentence boundary near the chunk end
            boundary = text.rfind('. ', start, end)
            if boundary > start and (boundary - start) > (max_chunk_size // 2):
                end = boundary + 1  # Include the period
                
            chunks.append(text[start:end].strip())
            start = end - overlap  # Overlap with next chunk
            
        return chunks
    
    def generate_summary(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        length_penalty: Optional[float] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        no_repeat_ngram_size: Optional[int] = None,
        **generation_kwargs
    ) -> str:
        """Generate a summary for the given text.
        
        Args:
            text: Input text to summarize.
            max_length: Maximum length of the summary in tokens.
            min_length: Minimum length of the summary in tokens.
            length_penalty: Exponential penalty to the length.
            num_beams: Number of beams for beam search.
            temperature: Value for controlling randomness in generation.
            top_k: Number of highest probability tokens to keep for top-k sampling.
            top_p: Nucleus sampling parameter.
            do_sample: Whether to use sampling instead of greedy decoding.
            early_stopping: Whether to stop generation when all beam hypotheses are finished.
            no_repeat_ngram_size: If set, n-grams of this size can only occur once.
            **generation_kwargs: Additional generation parameters.
            
        Returns:
            str: Generated summary.
        """
        # Update default generation parameters with any provided overrides
        generation_config = self.get_generation_config(
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **generation_kwargs
        )
        
        # Preprocess the input text
        text = self._preprocess_text(text)
        
        # If the text is too long, split it into chunks and process each chunk
        if len(text) > 10000:  # Arbitrary threshold
            logger.info("Input text is long, processing in chunks...")
            chunks = self._chunk_text(text)
            summaries = []
            
            for chunk in tqdm(chunks, desc="Processing chunks"):
                chunk_summary = self._generate_single_summary(chunk, generation_config)
                summaries.append(chunk_summary)
                
            # Combine and summarize the chunk summaries
            combined_summary = "\n\n".join(summaries)
            if len(combined_summary) > 1000:  # Arbitrary threshold
                logger.info("Combining chunk summaries...")
                return self._generate_single_summary(combined_summary, generation_config)
            return combined_summary
            
        return self._generate_single_summary(text, generation_config)
    
    def _generate_single_summary(
        self,
        text: str,
        generation_config: Dict[str, Any]
    ) -> str:
        """Generate a summary for a single chunk of text.
        
        Args:
            text: Input text to summarize.
            generation_config: Configuration for text generation.
            
        Returns:
            str: Generated summary.
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _generate_single_summary")
    
    def get_generation_config(
        self,
        **overrides
    ) -> Dict[str, Any]:
        """Get the generation configuration with overrides.
        
        Args:
            **overrides: Parameter overrides.
            
        Returns:
            Dict[str, Any]: Generation configuration.
        """
        config = self.default_generation_config.copy()
        config.update({k: v for k, v in overrides.items() if v is not None})
        return config
    
    def __call__(
        self,
        text: str,
        **generation_kwargs
    ) -> str:
        """Generate a summary for the given text (convenience method).
        
        Args:
            text: Input text to summarize.
            **generation_kwargs: Generation parameters.
            
        Returns:
            str: Generated summary.
        """
        return self.generate_summary(text, **generation_kwargs)
    
    def __repr__(self) -> str:
        """Return a string representation of the summarizer."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
