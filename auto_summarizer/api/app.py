"""FastAPI application for the summarization service."""
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging

from ..models.selector import ModelSelector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model selector
model_selector = ModelSelector()

app = FastAPI(
    title="Auto Summarizer API",
    description="REST API for text summarization using adaptive model selection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50000)
    model: Optional[str] = None
    min_length: int = 30
    max_length: int = 200

class BatchSummarizeRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model: Optional[str] = None
    min_length: int = 30
    max_length: int = 200

class SummarizeResponse(BaseModel):
    summary: str
    model_used: str
    processing_time: float

class BatchSummarizeResponse(BaseModel):
    summaries: List[str]
    model_used: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    system_memory_mb: float
    gpu_available: bool

# Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# API Endpoints
@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Generate a summary for a single text."""
    start_time = time.time()
    
    try:
        # Get the appropriate model
        model_name = request.model
        if not model_name:
            model_name, _ = model_selector.select_model(request.text)
        
        # Generate summary
        summarizer = model_selector.get_summarizer(model_name)
        summary = summarizer(
            request.text,
            min_length=request.min_length,
            max_length=request.max_length
        )
        
        return {
            "summary": summary,
            "model_used": model_name,
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Error in summarize: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/batch-summarize", response_model=BatchSummarizeResponse)
async def batch_summarize(request: BatchSummarizeRequest):
    """Generate summaries for multiple texts in batch."""
    start_time = time.time()
    
    try:
        if request.model:
            # Use specified model for all texts
            summarizer = model_selector.get_summarizer(request.model)
            summaries = [
                summarizer(text, min_length=request.min_length, max_length=request.max_length)
                for text in request.texts
            ]
            model_used = request.model
        else:
            # Use model selector for each text
            summaries = await model_selector.batch_summarize(request.texts)
            model_used = "adaptive"
            
        return {
            "summaries": summaries,
            "model_used": model_used,
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Error in batch_summarize: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and model status."""
    try:
        # Get system info
        import psutil
        import torch
        
        system_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        gpu_available = torch.cuda.is_available()
        
        # Get model status
        models_loaded = {
            name: {
                "status": "loaded" if hasattr(model_selector, f"_{name}_model") else "lazy",
                **model_selector.get_metrics().get(name, {})
            }
            for name in model_selector.models
        }
        
        return {
            "status": "healthy",
            "models_loaded": models_loaded,
            "system_memory_mb": round(system_memory, 2),
            "gpu_available": gpu_available
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Add startup event to preload models
@app.on_event("startup")
async def startup_event():
    """Preload models on startup."""
    try:
        # Preload extractive model (fast and lightweight)
        logger.info("Preloading extractive model...")
        model_selector.get_summarizer("extractive")
        logger.info("Startup complete")
    except Exception as e:
        logger.error(f"Failed to preload models: {e}")
        # Don't crash if preloading fails - models will load on demand
