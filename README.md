# Auto Summarizer

A production-ready text summarization pipeline combining extractive and abstractive approaches.

## Clone & Setup

```bash
git clone https://github.com/KazKozDev/summarization-pipeline.git
cd summarization-pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Tech Stack

**Backend & Core:**
- Python 3.9+, FastAPI, Pydantic, Uvicorn

**ML/NLP:**
- PyTorch, Transformers (Hugging Face), spaCy, NLTK
- scikit-learn, NetworkX, NumPy, Pandas

**Web & UI:**
- Streamlit, Gradio

**DevOps:**
- Docker, GitHub Actions, Prometheus/Grafana

**Testing & Quality:**
- pytest, black, flake8, mypy

## Architecture

**Core Components:**
- **Extractive**: TextRank algorithm with TF-IDF, position, and length features
- **Abstractive**: BART transformer models via Hugging Face
- **Hybrid**: Extractive preprocessing + BART summarization
- **Adaptive**: Automatic model selection based on text length and system resources

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm

# Run web interface
streamlit run streamlit_app.py

# Or run API server
uvicorn auto_summarizer.api.app:app --reload
```

## Usage

**Python API:**
```python
from auto_summarizer.core.summarizer import Summarizer

summarizer = Summarizer()
result = summarizer.summarize(text, method="combined", top_n=3)
print("\n".join(result['summary']))
```

**REST API:**
```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "max_length": 150}'
```

**Docker:**
```bash
docker-compose up --build
```

## Features

- Multiple summarization algorithms (TextRank, feature-based, BART)
- Automatic model selection based on system resources
- Batch processing with parallel execution
- ROUGE/BLEU evaluation metrics
- Production deployment with monitoring
- Comprehensive test suite (unit + integration)

## Project Structure

```
auto_summarizer/
├── core/           # Main logic (preprocessor, feature extraction, summarizer)
├── models/         # Model implementations (TextRank, BART, hybrid)
├── api/           # FastAPI REST endpoints
tests/             # Unit and integration tests
scripts/           # Evaluation and data generation utilities
```