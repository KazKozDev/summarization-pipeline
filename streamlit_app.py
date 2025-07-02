"""Interactive Streamlit interface for Auto Summarizer.

Run with:
    streamlit run streamlit_app.py
"""
import streamlit as st
from auto_summarizer.models import get_summarizer

st.set_page_config(page_title="Auto Summarizer", page_icon="📝", layout="centered")

st.title("📝 Auto Summarizer")
st.markdown("Generate high-quality summaries with state-of-the-art transformer models.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    model_key = st.selectbox(
        "Model",
        options=["bart-large", "bart-distil"],
        index=0,
        help="Choose the summarization model"
    )
    device_option = st.selectbox(
        "Device",
        options=["auto", "cpu", "mps"],
        index=0,
        help="auto → CUDA if available, then MPS, else CPU"
    )
    max_length = st.slider("Max summary length", 30, 200, 150)
    min_length = st.slider("Min summary length", 5, 100, 30)
    num_beams = st.slider("Beam search width", 1, 8, 4)

# Text input
text = st.text_area("Input text", height=300)

if st.button("Summarize"):
    if not text.strip():
        st.warning("Please paste some text to summarize.")
        st.stop()

    with st.spinner("Loading model… this may take a while the first time."):
        summarizer = get_summarizer(model_key, device=device_option if device_option != "auto" else None)

    with st.spinner("Generating summary…"):
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True
        )
    st.subheader("Summary")
    st.success(summary)
