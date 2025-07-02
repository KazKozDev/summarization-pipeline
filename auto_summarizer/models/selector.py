"""Adaptive model selection and scaling for summarization."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque, defaultdict
import psutil
import torch
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

from auto_summarizer.models import get_summarizer

@dataclass
class ModelConfig:
    """Configuration for a summarization model."""
    name: str
    min_length: int = 50
    max_length: int = 4000
    priority: int = 1
    max_batch_size: int = 8
    required_memory_mb: int = 2000

class ModelSelector:
    """Dynamically selects the best model based on text and system metrics.
    
    Features:
    - Adaptive model selection based on text length and available resources
    - Memory-aware batching
    - Performance metrics tracking
    - Fallback mechanisms
    """
    
    def __init__(self, max_workers: int = 3, history_size: int = 100):
        """Initialize the model selector.
        
        Args:
            max_workers: Maximum number of parallel model instances
            history_size: Number of inference metrics to keep in history
        """
        self.models = {
            "extractive": ModelConfig(
                "extractive", 
                max_length=10000,
                priority=1,
                required_memory_mb=500
            ),
            "hybrid": ModelConfig(
                "hybrid-default",
                priority=2,
                required_memory_mb=3000
            ),
            "bart": ModelConfig(
                "bart-large",
                priority=3,
                required_memory_mb=4000
            )
        }
        self.metrics: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=history_size))
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._load_times: Dict[str, float] = {}
        
    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        return psutil.virtual_memory().available / (1024 * 1024)
        
    def get_gpu_memory_mb(self) -> Optional[float]:
        """Get available GPU memory in MB if available."""
        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info()[0] / (1024 * 1024)
    
    def get_model_load_time(self, model_name: str) -> float:
        """Get average load time for a model in seconds."""
        if model_name not in self._load_times:
            start = time.time()
            get_summarizer(model_name)  # Warm up
            self._load_times[model_name] = time.time() - start
        return self._load_times[model_name]
    
    def select_model(self, text: str) -> Tuple[str, Dict]:
        """Select the best model for the given text.
        
        Args:
            text: Input text to summarize
            
        Returns:
            Tuple of (model_name, metadata)
        """
        text_len = len(text)
        available_mem = min(
            self.get_available_memory_mb(),
            self.get_gpu_memory_mb() or float('inf')
        )
        
        # Filter models by requirements
        viable_models = []
        for name, cfg in self.models.items():
            if (cfg.min_length <= text_len <= cfg.max_length and 
                cfg.required_memory_mb <= available_mem):
                # Calculate score based on priority and memory efficiency
                score = cfg.priority * 10 + (available_mem / cfg.required_memory_mb)
                viable_models.append((score, name, cfg))
        
        if not viable_models:
            return "extractive", {"reason": "fallback", "available_mem": available_mem}
            
        # Sort by score (descending)
        viable_models.sort(reverse=True, key=lambda x: x[0])
        selected = viable_models[0][1]
        return selected, {"reason": "optimal", "available_mem": available_mem}
    
    async def batch_summarize(self, texts: List[str]) -> List[str]:
        """Process multiple texts using optimal models in parallel batches.
        
        Args:
            texts: List of input texts to summarize
            
        Returns:
            List of summaries in the same order as input
        """
        # Group texts by selected model
        model_groups = defaultdict(list)
        model_indices = defaultdict(list)
        
        for idx, text in enumerate(texts):
            model_name, _ = self.select_model(text)
            model_groups[model_name].append(text)
            model_indices[model_name].append(idx)
            
        # Process each model group in parallel
        futures: List[Future] = []
        results = {}
        
        for model_name, batch_texts in model_groups.items():
            future = self.executor.submit(
                self._process_batch,
                model_name,
                batch_texts
            )
            future.model_name = model_name  # type: ignore
            futures.append(future)
            
        # Collect and reorder results
        output = [""] * len(texts)
        for future in futures:
            model_name = future.model_name  # type: ignore
            batch_results = future.result()
            for idx, result in zip(model_indices[model_name], batch_results):
                output[idx] = result
                
        return output
        
    def _process_batch(self, model_name: str, texts: List[str]) -> List[str]:
        """Process a batch of texts with the same model.
        
        Args:
            model_name: Name of the model to use
            texts: List of input texts
            
        Returns:
            List of summaries
        """
        start_time = time.time()
        summarizer = get_summarizer(model_name)
        
        try:
            results = []
            batch_size = self.models[model_name].max_batch_size
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = [summarizer(text) for text in batch]
                results.extend(batch_results)
                
            # Record metrics
            duration = time.time() - start_time
            self.metrics[model_name].append({
                "batch_size": len(texts),
                "duration": duration,
                "avg_time_per_text": duration / len(texts),
                "timestamp": time.time()
            })
            
            return results
            
        except Exception as e:
            # Fallback to extractive summarization on error
            if model_name != "extractive":
                return self._process_batch("extractive", texts)
            raise
    
    def get_metrics(self) -> Dict[str, dict]:
        """Get performance metrics for all models."""
        metrics = {}
        for model_name, model_metrics in self.metrics.items():
            if not model_metrics:
                continue
            metrics[model_name] = {
                "total_batches": len(model_metrics),
                "avg_batch_size": np.mean([m["batch_size"] for m in model_metrics]),
                "avg_duration": np.mean([m["duration"] for m in model_metrics]),
                "avg_time_per_text": np.mean([m["avg_time_per_text"] for m in model_metrics]),
            }
        return metrics
