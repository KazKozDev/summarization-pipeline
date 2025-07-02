"""
Streamlit web interface for the automatic text summarization application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from loguru import logger
from auto_summarizer.core.summarizer import Summarizer

# Configure logger
logger.add("summarizer.log", rotation="10 MB")

# Set page config
st.set_page_config(
    page_title="Auto Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 32px; font-weight: 700; color: #1f77b4;}
    .sub-header {font-size: 20px; font-weight: 600; color: #2c3e50;}
    .summary-box {background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin: 10px 0;}
    .score-badge {background-color: #e9ecef; padding: 2px 8px; border-radius: 12px; font-size: 12px;}
    .stProgress > div > div > div > div {background-color: #1f77b4;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = Summarizer()
    st.session_state.summary_result = None
    st.session_state.show_advanced = False
    st.session_state.show_evaluation = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Summarization method
    method = st.selectbox(
        "Summarization Method",
        ["combined", "textrank", "features"],
        help="Choose the summarization algorithm to use"
    )
    
    # Summary length control
    summary_length = st.slider(
        "Summary Length",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of sentences in the summary"
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Minimum similarity between sentences for TextRank"
        )
        
        damping_factor = st.slider(
            "Damping Factor",
            min_value=0.1,
            max_value=0.99,
            value=0.85,
            step=0.01,
            help="Damping factor for PageRank algorithm"
        )
        
        language = st.selectbox(
            "Language",
            ["english", "spanish", "french", "german"],
            index=0,
            help="Language of the input text"
        )
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application provides automatic text summarization using multiple algorithms:
    - **TextRank**: Graph-based ranking algorithm
    - **Feature-based**: Uses TF-IDF, position, and other features
    - **Combined**: Uses both methods for improved results
    """)

# Main content
st.markdown("<h1 class='main-header'>📝 Auto Summarizer</h1>", unsafe_allow_html=True)
st.markdown("Automatically generate concise summaries from your text documents.")

# Input area
with st.container():
    st.markdown("### Input Text")
    input_text = st.text_area(
        "Paste your text here (minimum 3 sentences for best results)",
        height=250,
        placeholder="Enter or paste your text here..."
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        summarize_btn = st.button("✨ Generate Summary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    # Handle clear button
    if clear_btn:
        input_text = ""
        st.session_state.summary_result = None
        st.experimental_rerun()
    
    # Handle summarize button
    if summarize_btn and input_text:
        with st.spinner("Generating summary..."):
            try:
                # Update summarizer parameters
                st.session_state.summarizer.textrank.similarity_threshold = similarity_threshold
                st.session_state.summarizer.textrank.damping_factor = damping_factor
                st.session_state.summarizer.language = language
                
                # Generate summary
                st.session_state.summary_result = st.session_state.summarizer.summarize(
                    text=input_text,
                    method=method,
                    top_n=summary_length
                )
                
                # Log the action
                logger.info(f"Generated {method} summary with {summary_length} sentences")
                
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                logger.error(f"Summarization error: {str(e)}")

# Display results
if st.session_state.summary_result and 'summary' in st.session_state.summary_result:
    result = st.session_state.summary_result
    
    # Summary section
    st.markdown("---")
    st.markdown("### Generated Summary")
    
    # Display summary with scores
    if 'scores' in result and len(result['scores']) > 0:
        # Get top sentences with their scores
        sentences = result['processed_data']['sentences']
        scores = result['scores']
        
        # Normalize scores for display
        if scores:
            max_score = max(scores) if max(scores) > 0 else 1
            norm_scores = [s/max_score for s in scores]
        else:
            norm_scores = [0] * len(sentences)
        
        # Display summary
        summary_container = st.container()
        with summary_container:
            for i, sent in enumerate(result['summary']):
                if sent in sentences:
                    idx = sentences.index(sent)
                    score = norm_scores[idx] if idx < len(norm_scores) else 0
                    st.markdown(
                        f"<div class='summary-box'>"
                        f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>"
                        f"<span style='font-weight: 500;'>{sent}</span>"
                        f"<span class='score-badge'>Score: {score:.2f}</span>"
                        f"</div>"
                        f"<div style='width: 100%; height: 4px; background: #e9ecef; border-radius: 2px;'>"
                        f"<div style='width: {score*100}%; height: 100%; background: #1f77b4; border-radius: 2px;'></div>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
    else:
        # Fallback display without scores
        for sent in result['summary']:
            st.markdown(f"- {sent}")
    
    # Evaluation section
    with st.expander("📊 Evaluation Metrics", expanded=False):
        if 'reference' in result and result['reference']:
            eval_metrics = st.session_state.summarizer.evaluate_summary(
                summary=result['summary'],
                reference=result['reference']
            )
            
            if eval_metrics:
                # Display metrics in columns
                cols = st.columns(len(eval_metrics))
                for i, (metric, score) in enumerate(eval_metrics.items()):
                    with cols[i]:
                        st.metric(
                            label=metric.upper(),
                            value=f"{score:.3f}",
                            help=f"{metric.upper()} score (0-1)"
                        )
        else:
            st.info("No reference summary provided for evaluation.")
            
            # Allow manual reference input
            st.markdown("**Add a reference summary for evaluation:**")
            reference_text = st.text_area(
                "Paste the reference summary here",
                height=150,
                key="reference_input"
            )
            
            if st.button("Evaluate"):
                if reference_text:
                    result['reference'] = reference_text.split(". ")
                    st.experimental_rerun()
    
    # Debug information (collapsed by default)
    with st.expander("🔍 Debug Information", expanded=False):
        st.json({
            "method": result.get('method', 'unknown'),
            "num_sentences": len(result.get('summary', [])),
            "scores_available": 'scores' in result and len(result['scores']) > 0,
            "features_extracted": 'feature_data' in result,
            "processing_time": result.get('processing_time', 'N/A')
        })
        
        # Show processed data if available
        if 'processed_data' in result and result['processed_data']:
            with st.expander("Processed Data"):
                st.json({
                    "num_sentences": len(result['processed_data'].get('sentences', [])),
                    "num_tokens": len(result['processed_data'].get('tokens', [])),
                    "num_named_entities": len(result['processed_data'].get('named_entities', [])),
                    "sample_tokens": result['processed_data'].get('tokens', [])[:10] + ['...'] if 'tokens' in result['processed_data'] else []
                })

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 14px;'>"
    "Auto Summarizer v1.0.0 | Built with ❤️ using Streamlit"
    "</div>",
    unsafe_allow_html=True
)

# Add some JavaScript for better UX
st.markdown(
    """
    <script>
    // Auto-scroll to results
    const scrollToResults = () => {
        const results = document.querySelector("[data-testid='stMarkdownContainer'] h3");
        if (results) {
            results.scrollIntoView({ behavior: 'smooth' });
        }
    }
    
    // Run on page load
    window.addEventListener('load', () => {
        // Add event listener to summary button
        const button = document.querySelector("[data-testid='stButton'] button");
        if (button) {
            button.addEventListener('click', scrollToResults);
        }
    });
    </script>
    """,
    unsafe_allow_html=True
)
