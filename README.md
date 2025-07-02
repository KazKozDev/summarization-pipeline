# Auto Summarizer

An adaptive text summarization service that combines extractive and abstractive approaches for optimal results. The system automatically selects the best model based on input text and available system resources.

## Features

- **Adaptive Model Selection**: Automatically chooses between extractive and abstractive models based on text length and system resources
- **Multiple Model Support**: Includes BART, Pegasus, and custom hybrid models
- **REST API**: FastAPI-based endpoints for single and batch summarization
- **Docker Support**: Easy deployment with Docker and docker-compose
- **Monitoring**: Built-in health checks and metrics

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- CUDA-enabled GPU (recommended for production)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/auto-summarizer.git
   cd auto-summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

Start the FastAPI server:
```bash
uvicorn auto_summarizer.api.app:app --reload
```

### Using Docker

Build and run with docker-compose:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Single Text Summarization
```http
POST /summarize
```

**Request Body:**
```json
{
  "text": "Your long text to summarize...",
  "model": "hybrid-default",
  "min_length": 30,
  "max_length": 200
}
```

### Batch Summarization
```http
POST /batch-summarize
```

**Request Body:**
```json
{
  "texts": ["First text...", "Second text..."],
  "model": "hybrid-default",
  "min_length": 30,
  "max_length": 200
}
```

### Health Check
```http
GET /health
```

## Available Models

- `extractive`: Fast, extractive summarization (best for very long texts)
- `hybrid-default`: Hybrid extractive-abstractive approach (good balance of speed and quality)
- `bart-large`: High-quality abstractive summarization (slower but higher quality)
- `pegasus-xsum`: Abstractive model optimized for very short summaries

## Configuration

Environment variables:
- `MODEL_CACHE_DIR`: Directory to cache downloaded models (default: `./models`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `DEVICE`: Force CPU/GPU (`cpu`, `cuda`, or `auto`)

## Monitoring

Prometheus metrics are available at `/metrics` when running with the `--metrics` flag.

## License

MIT

An automatic text summarization application that generates concise summaries from large text inputs using multiple algorithms.

## Features

- **Multiple Summarization Methods**
  - TextRank algorithm
  - Feature-based summarization
  - Combined approach for improved results

- **Advanced Features**
  - Customizable summary length
  - Adjustable similarity thresholds
  - Multiple language support
  - Evaluation metrics (ROUGE, BLEU)

- **User-Friendly Interface**
  - Web-based interface using Streamlit
  - Real-time preview
  - Interactive controls

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/auto-summarizer.git
   cd auto-summarizer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required NLTK data and spaCy model:
   ```bash
   python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
   python -m spacy download en_core_web_sm
   ```

## Usage

### Web Interface

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Paste your text in the input area and click "Generate Summary"

### Python API

You can also use the summarizer in your Python code:

```python
from auto_summarizer.core.summarizer import Summarizer

# Initialize the summarizer
summarizer = Summarizer()

# Input text
text = """
Your long text document goes here. It can be multiple paragraphs long.
The summarizer will extract the most important sentences to create a concise summary.
"""

# Generate a summary
result = summarizer.summarize(
    text=text,
    method="combined",  # 'textrank', 'features', or 'combined'
    top_n=5,           # Number of sentences in the summary
    summary_ratio=0.3  # Alternative to top_n: ratio of sentences to include
)

# Get the summary sentences
summary = result['summary']
print("\n".join(summary))
```

## Architecture

The application is structured as follows:

```
auto-summarizer/
├── auto_summarizer/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── preprocessor.py    # Text cleaning and tokenization
│   │   ├── feature_extractor.py # Feature extraction
│   │   └── summarizer.py      # Main summarization logic
│   ├── models/
│   │   ├── __init__.py
│   │   └── textrank.py        # TextRank implementation
│   └── utils/
│       └── __init__.py
├── app.py                    # Streamlit web interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Configuration

You can customize the summarization behavior by modifying the following parameters:

- **Similarity Threshold**: Controls how similar sentences need to be to be connected in the TextRank graph
- **Damping Factor**: Affects the random walk probability in the PageRank algorithm
- **Summary Length**: Number of sentences in the summary
- **Language**: Currently supports English, with more languages coming soon

## Evaluation

The system provides several metrics to evaluate summary quality:

- **ROUGE-1, ROUGE-2, ROUGE-L**: Measures n-gram overlap with reference summaries
- **BLEU**: Measures precision of n-grams compared to reference

To evaluate a summary, provide a reference summary in the evaluation section of the web interface.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The TextRank algorithm is based on the paper by Rada Mihalcea and Paul Tarau
- Uses spaCy and NLTK for NLP processing
- Streamlit for the web interface
