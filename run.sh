#!/usr/bin/env bash
set -e

# Activate virtual environment if exists
if [ -f "myenv/bin/activate" ]; then
    source myenv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Start FastAPI server in background
echo "Starting FastAPI server..."
uvicorn auto_summarizer.api.app:app --reload &

# Start Streamlit app
echo "Starting Streamlit interface..."
streamlit run streamlit_app.py
