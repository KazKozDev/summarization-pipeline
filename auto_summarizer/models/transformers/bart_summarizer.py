"""BART model for text summarization.

This module provides an implementation of a BART-based text summarizer.
BART (Bidirectional and Auto-Regressive Transformers) is particularly
effective for abstractive summarization tasks.
"""
from typing import Dict, Any, List, Optional, Union
import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    GenerationConfig
)
from .base_summarizer import BaseTransformerSummarizer
import logging

logger = logging.getLogger(__name__)

class BARTSummarizer(BaseTransformerSummarizer):
    """BART-based text summarizer.
    
    This class implements a text summarizer using the BART (Bidirectional and
    Auto-Regressive Transformers) model. It's particularly well-suited for
    abstractive summarization tasks.
    
    Example:
        >>> from auto_summarizer.models.transformers import BARTSummarizer
        >>> summarizer = BARTSummarizer("facebook/bart-large-cnn")
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
    
    def _load_model(self) -> None:
        """Load the BART model and tokenizer."""
        logger.info(f"Loading BART model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = \
            BartTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=self.use_auth_token,
                **self.model_kwargs
            )
        
        # Load model
        self.model: PreTrainedModel = BartForConditionalGeneration.from_pretrained(
            self.model_name,
            use_auth_token=self.use_auth_token,
            **self.model_kwargs
        ).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Update model config with any additional parameters
        if hasattr(self.model.config, 'update'):
            self.model.config.update(self.model_kwargs)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess the input text before tokenization.
        
        Args:
            text: Input text to preprocess.
            
        Returns:
            str: Preprocessed text.
        """
        # BART works well with the raw text, but we'll do some basic cleaning
        text = text.strip()
        
        # Ensure the text ends with a period
        if text and text[-1] not in {'.', '!', '?'}:
            text += '.'
            
        return text
    
    def _postprocess_summary(self, summary: str) -> str:
        """Postprocess the generated summary.
        
        Args:
            summary: Generated summary to postprocess.
            
        Returns:
            str: Postprocessed summary.
        """
        # Basic postprocessing
        summary = summary.strip()
        
        # Remove any leading/trailing whitespace and newlines
        summary = ' '.join(summary.split())
        
        # Capitalize first letter
        if summary:
            summary = summary[0].upper() + summary[1:]
            
            # Ensure the summary ends with a period
            if summary[-1] not in {'.', '!', '?'}:
                summary += '.'
                
        return summary
    
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
        try:
            # Tokenize the input text
            inputs = self.tokenizer(
                text,
                max_length=1024,  # BART's maximum input length
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config
                )
            
            # Decode the generated summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Postprocess the summary
            return self._postprocess_summary(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Return an empty string on error
            return ""
    
    def generate_summary_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        **generation_kwargs
    ) -> List[str]:
        """Generate summaries for a batch of texts.
        
        Args:
            texts: List of input texts to summarize.
            batch_size: Number of texts to process in parallel.
            **generation_kwargs: Additional generation parameters.
            
        Returns:
            List[str]: List of generated summaries.
        """
        if not texts:
            return []
            
        # Get generation config
        generation_config = self.get_generation_config(**generation_kwargs)
        
        # Process in batches
        summaries = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Preprocess batch
            batch = [self._preprocess_text(text) for text in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summaries for the batch
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config
                )
            
            # Decode the generated summaries
            batch_summaries = self.tokenizer.batch_decode(
                summary_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Postprocess summaries
            batch_summaries = [
                self._postprocess_summary(s) for s in batch_summaries
            ]
            
            summaries.extend(batch_summaries)
            
        return summaries
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information.
        """
        return {
            "model_type": "BART",
            "model_name": self.model_name,
            "device": str(self.device),
            "max_length": self.model.config.max_position_embeddings,
            "vocab_size": self.model.config.vocab_size,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }
