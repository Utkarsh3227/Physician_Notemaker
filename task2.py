import streamlit as st
from transformers import pipeline
import os
import json

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Cache resource-intensive models
@st.cache_resource
def load_models():
    # Load sentiment classifier
    sentiment_pipeline = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )
    
    # Load intent classifier
    intent_pipeline = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
    
    return sentiment_pipeline, intent_pipeline

def analyze_text(text):
    sentiment_map = {
        "NEGATIVE": "Anxious",
        "POSITIVE": "Reassured",
        "NEUTRAL": "Neutral"
    }
    
    intent_labels = ["Seeking reassurance", "Reporting symptoms", "Expressing concern"]
    
    # Get models
    sentiment_pipe, intent_pipe = load_models()
    
    # Sentiment analysis
    sentiment_result = sentiment_pipe(text)
    raw_sentiment = sentiment_result[0]['label']
    sentiment = sentiment_map.get(raw_sentiment.upper(), "Neutral")
    sentiment_score = sentiment_result[0]['score']
    
    # Intent analysis
    intent_result = intent_pipe(text, intent_labels)
    intent = intent_result['labels'][0]
    intent_score = intent_result['scores'][0]
    
    return {
        # "text": text,
        "sentiment": sentiment,
        "intent": intent,
        # "sentiment_score": float(sentiment_score),
        # "intent_score": float(intent_score)
    }

# Streamlit UI
st.title("Patient Response Analyzer")

# Input section
patient_input = st.text_area("Enter patient response:", height=150)

if st.button("Analyze") and patient_input.strip():
    with st.spinner("Analyzing..."):
        results = analyze_text(patient_input.strip())
    
    # Display results as JSON
    st.json(results)