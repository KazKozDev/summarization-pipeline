"""
Unit tests for ModelSelector.

Tests the adaptive model selection and batching functionality.
"""
import time
from unittest.mock import patch, MagicMock

import pytest
import torch

from auto_summarizer.models.selector import ModelSelector, ModelConfig

class TestModelSelector:
    @pytest.fixture
    def mock_models(self):
        return {
            "small": ModelConfig("small", max_length=1000, priority=1, required_memory_mb=1000),
            "medium": ModelConfig("medium", max_length=5000, priority=2, required_memory_mb=2000),
            "large": ModelConfig("large", max_length=10000, priority=3, required_memory_mb=4000),
        }

    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    def test_select_model_memory_aware(self, mock_cuda, mock_vm, mock_models):
        """Test model selection based on available memory."""
        # Setup
        mock_vm.return_value.available = 3 * 1024 * 1024 * 1024  # 3GB
        mock_cuda.return_value = False
        
        selector = ModelSelector()
        selector.models = mock_models
        
        # Test with enough memory for all models
        model, meta = selector.select_model("test" * 100)
        assert model == "large"
        assert meta["reason"] == "optimal"
        
        # Test with limited memory
        mock_vm.return_value.available = 1.5 * 1024 * 1024 * 1024  # 1.5GB
        model, meta = selector.select_model("test" * 100)
        assert model == "medium"

    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    def test_select_model_text_length(self, mock_cuda, mock_vm, mock_models):
        """Test model selection based on input text length."""
        mock_vm.return_value.available = 10 * 1024 * 1024 * 1024  # 10GB
        mock_cuda.return_value = False
        
        selector = ModelSelector()
        selector.models = mock_models
        
        # Test with very long text
        model, _ = selector.select_model("test" * 3000)  # 12k chars
        assert model == "large"
        
        # Test with medium text
        model, _ = selector.select_model("test" * 1000)  # 4k chars
        assert model == "medium"

    @patch('concurrent.futures.ThreadPoolExecutor')
    @patch('auto_summarizer.models.get_summarizer')
    def test_batch_processing(self, mock_get_summarizer, mock_executor, mock_models):
        """Test batch processing with mock executor and summarizer."""
        # Setup mocks
        mock_summarizer = MagicMock()
        mock_summarizer.side_effect = lambda x: f"summary_{x}"
        mock_get_summarizer.return_value = mock_summarizer
        
        future = MagicMock()
        future.result.return_value = ["summary_test1", "summary_test2"]
        setattr(future, 'model_name', "test_model")  # Set attribute for mock
        mock_executor.return_value.submit.return_value = future
        
        selector = ModelSelector()
        selector.models = {"test_model": ModelConfig("test_model")}
        
        # Test with asyncio
        import asyncio
        results = asyncio.run(selector.batch_summarize(["test1", "test2"]))
        
        # Verify results
        assert results == ["summary_test1", "summary_test2"]
        assert mock_summarizer.called

    @patch('psutil.virtual_memory')
    @patch('torch.cuda.is_available')
    def test_fallback_mechanism(self, mock_cuda, mock_vm):
        """Test fallback to extractive model when resources are limited."""
        # Setup low memory condition
        mock_vm.return_value.available = 0.1 * 1024 * 1024 * 1024  # 100MB
        mock_cuda.return_value = False
        
        selector = ModelSelector()
        
        # Should fall back to extractive model
        model, meta = selector.select_model("test")
        assert model == "extractive"
        assert meta["reason"] == "fallback"
        
    def test_get_metrics(self, mock_models):
        """Test metrics collection functionality."""
        selector = ModelSelector()
        selector.models = mock_models
        
        # Add some mock metrics
        selector.metrics["test_model"] = [
            {"batch_size": 2, "duration": 1.0, "avg_time_per_text": 0.5, "timestamp": 0},
            {"batch_size": 4, "duration": 1.8, "avg_time_per_text": 0.45, "timestamp": 1}
        ]
        
        metrics = selector.get_metrics()
        assert "test_model" in metrics
        assert metrics["test_model"]["total_batches"] == 2
        assert metrics["test_model"]["avg_batch_size"] == 3.0
