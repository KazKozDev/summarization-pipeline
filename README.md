# Auto Summarizer

*Educational text summarization project demonstrating multiple NLP approaches and modern Python development practices*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://hub.docker.com/)
[![Streamlit](https://img.shields.io/badge/streamlit-app-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/fastapi-framework-009688?logo=fastapi)](https://fastapi.tiangolo.com/)

Auto Summarizer is an educational project that demonstrates different approaches to automatic text summarization. It implements three main summarization methods: extractive (TextRank algorithm), feature-based scoring, and abstractive (using BART transformer), along with a combined approach that merges multiple techniques. The project showcases modern Python development practices including clean architecture, comprehensive testing, Docker containerization, and CI/CD pipelines with multiple interfaces (web UI, Python API, REST endpoints) for easy experimentation.

This project serves as a comprehensive learning resource for computer science students studying NLP and machine learning, Python developers looking for well-structured project examples, and ML engineers wanting to understand different summarization approaches. It was created to demonstrate how different NLP algorithms work in practice, modern Python application architecture, and real-world software development workflow, providing a solid foundation that others can learn from, extend, or use as inspiration for their own projects.

## Tech Stack

### **Core Technologies**
- **Python 3.9+** - Main programming language
- **NLTK & spaCy** - Natural language processing
- **scikit-learn** - Feature extraction and machine learning utilities
- **NetworkX** - Graph algorithms for TextRank implementation

### **Machine Learning**
- **Transformers (Hugging Face)** - BART model for abstractive summarization
- **PyTorch** - Deep learning framework (via transformers)
- **NumPy & Pandas** - Data manipulation and numerical computing

### **Web & API**
- **Streamlit** - Interactive web interface
- **FastAPI** - REST API framework (basic implementation)
- **Uvicorn** - ASGI server

### **Development & DevOps**
- **pytest** - Testing framework
- **Docker** - Containerization
- **GitHub Actions** - CI/CD pipeline
- **black, flake8, mypy** - Code quality tools

## Installation & Getting Started

### Prerequisites

- Python 3.9 or higher
- 4GB+ RAM (for BART model)
- Internet connection (for downloading models)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/KazKozDev/auto-summarizer.git
cd auto-summarizer

# Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLP models
python -m spacy download en_core_web_sm

# Run the web interface
streamlit run streamlit_app.py
```

### Docker Installation

```bash
# Build and run with Docker
docker build -t auto-summarizer .
docker run -p 8000:8000 auto-summarizer

# Or use docker-compose
docker-compose up --build
```

### Basic Usage

**Python API:**
```python
from auto_summarizer.core.summarizer import Summarizer

# Initialize summarizer
summarizer = Summarizer()

# Generate summary
result = summarizer.summarize(
    text="Your long text here...",
    method="combined",  # Options: 'textrank', 'features', 'combined'
    top_n=3
)

# Access results
print("Summary:", result['summary'])
print("Method used:", result['method'])
print("Sentence scores:", result['scores'])
```

**Web Interface:**
1. Run `streamlit run app.py`
2. Open http://localhost:8501 in your browser
3. Paste your text and experiment with different settings

## Architecture & Implementation

### Project Structure

```
auto_summarizer/
├── core/
│   ├── preprocessor.py      # Text cleaning and tokenization
│   ├── feature_extractor.py # TF-IDF, position, length features  
│   └── summarizer.py        # Main orchestration logic
├── models/
│   ├── textrank.py          # TextRank algorithm implementation
│   ├── transformers/        # BART and transformer models
│   └── selector.py          # Model selection logic
└── api/
    └── app.py              # FastAPI endpoints (basic)

tests/
├── unit/                   # Unit tests
└── integration/           # Integration tests

scripts/
├── generate_sample_data.py # Test data generation
└── evaluate_hybrid.py     # Evaluation utilities
```

### Core Components

**1. Text Preprocessing (`preprocessor.py`)**
- Text cleaning and normalization
- Sentence tokenization using NLTK
- Named entity recognition with spaCy
- Stopword removal and lemmatization

**2. Feature Extraction (`feature_extractor.py`)**
- TF-IDF scoring for term importance
- Position-based scoring (beginning/end preference)
- Length normalization
- Title similarity calculation

**3. Summarization Methods**
- **TextRank**: Graph-based sentence ranking using PageRank algorithm
- **Feature-based**: Weighted combination of multiple features
- **BART Transformer**: Abstractive summarization using pre-trained models
- **Combined**: Hybrid approach merging extractive and abstractive methods

### Key Learning Points

**Algorithm Implementation:**
- PageRank adaptation for sentence ranking
- Feature engineering for extractive summarization
- Integration of transformer models
- Evaluation metrics (ROUGE, BLEU)

**Software Engineering:**
- Clean architecture with separation of concerns
- Dependency injection and modular design
- Error handling and fallback mechanisms
- Configuration management

## Testing & Quality

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=auto_summarizer --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
```

### Code Quality

```bash
# Format code
black auto_summarizer tests

# Check linting
flake8 auto_summarizer

# Type checking
mypy auto_summarizer
```

### What's Tested

- Text preprocessing functionality
- Feature extraction algorithms
- TextRank implementation
- BART model integration
- End-to-end summarization pipeline
- API endpoints (basic)

*Note: This is an educational project - test coverage is good but not exhaustive for production use.*

## Example Results

### Sample Input
```
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language. It is used to apply machine learning algorithms to text and speech. 
The ultimate objective of NLP is to read, decipher, understand, and make sense of 
the human language in a manner that is valuable.
```

### TextRank Output
```
• Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.
• The ultimate objective of NLP is to read, decipher, understand, and make sense of the human language.
```

### BART Output
```
Natural language processing is a subfield of computer science and artificial intelligence. 
It applies machine learning algorithms to text and speech to understand human language.
```

## Configuration

### Model Selection

Models are configured in `auto_summarizer/models/models.yml`:

```yaml
bart-large:
  class_path: auto_summarizer.models.transformers.BARTSummarizer
  model_name: facebook/bart-large-cnn
  description: High-quality abstractive summarization

bart-distil:
  class_path: auto_summarizer.models.transformers.BARTSummarizer
  model_name: sshleifer/distilbart-cnn-12-6
  description: Lightweight version for faster inference
```

### Customization

```python
# Adjust TextRank parameters
summarizer.textrank.similarity_threshold = 0.3
summarizer.textrank.damping_factor = 0.85

# Modify feature weights
weights = {
    'tfidf': 0.4,
    'position': 0.2,
    'length': 0.1,
    'title_similarity': 0.2,
    'keyword': 0.1
}
```

## Contributing

This is an educational project - contributions that improve the learning experience are welcome!

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/auto-summarizer.git
cd auto-summarizer

# Install development dependencies
pip install -r requirements.txt -r requirements-test.txt

# Run tests to ensure everything works
pytest tests/
```

### What We'd Welcome

- Additional summarization algorithms
- Better documentation and examples
- Performance improvements
- Bug fixes and code quality improvements
- Additional evaluation metrics
- More comprehensive tests

### Guidelines

- Follow existing code style (black, flake8, mypy)
- Add tests for new functionality
- Update documentation
- Keep the educational focus

## Potential Improvements

This project could be extended with:

**Algorithm Enhancements:**
- T5 and Pegasus transformer models
- Hierarchical summarization for long documents
- Domain-specific fine-tuning
- Multi-document summarization

**Engineering Improvements:**
- Comprehensive performance benchmarks
- Production-ready API with authentication
- Caching and performance optimization
- More sophisticated model selection
- Web interface improvements

**Evaluation & Metrics:**
- Human evaluation framework
- More evaluation datasets
- Comparative analysis tools
- Quality metrics dashboard

---

## Contact

**Questions about the code or want to discuss NLP?**

**Connect with me:**
- **LinkedIn**: [Artem Kazakov Kozlov](https://www.linkedin.com/in/kazkozdev/)
- **GitHub**: [@KazKozDev](https://github.com/KazKozDev)

**⭐If this project helped you learn something new, please give it a star!**

---

**License:** [MIT License](LICENSE) - Feel free to use this code for learning and experimentation

*This is an educational project created for learning purposes. While functional, it's not intended for production use without additional development and testing.*
