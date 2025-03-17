import streamlit as st
import tensorflow as tf
import numpy as np
from utils.model_loader import load_model, load_tokenizer
from utils.preprocessing import clean_text
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Load model and tokenizer
model = load_model('model/model.h5')
tokenizer = load_tokenizer('model/tokenizer.pkl')

# Streamlit UI
st.title("Sentiment Analysis Chatbot 🚀")
user_input = st.text_input("Enter your message:")

if user_input:
    try:
        # Clean and tokenize input
        cleaned_text = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=200)
        
        # Predict
        prediction = model.predict(padded_sequence)[0][0]
        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😠"
        st.success(f"**Sentiment**: {sentiment} (Confidence: {prediction:.2f})")
    except Exception as e:
        st.error(f"Error: {str(e)}")