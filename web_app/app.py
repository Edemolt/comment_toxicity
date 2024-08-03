import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization

# Load the saved model
model = tf.keras.models.load_model('new_model.h5')

# Define constants
MAX_FEATURES = 100000
OUTPUT_SEQUENCE_LENGTH = 1800

# Create TextVectorization layer
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=OUTPUT_SEQUENCE_LENGTH, output_mode='int')

# You need to adapt the vectorizer with your actual training data
# Example data for adaptation
df = pd.read_csv('train.csv')
x = df['comment_text']  # Replace with actual training data
vectorizer.adapt(x.values)

# Define the labels
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def predict_toxicity(text):
    # Vectorize the input text
    vectorized_text = vectorizer(tf.constant([text]))
    
    # Make prediction
    prediction = model.predict(vectorized_text)
    
    # Create a dictionary of label-prediction pairs
    results = {label: float(pred) for label, pred in zip(labels, prediction[0])}
    
    return results

# Streamlit app
st.title('Toxic Comment Classification')

# Text input
user_input = st.text_area("Enter a comment to analyze:")

if st.button('Analyze'):
    if user_input:
        # Get prediction
        results = predict_toxicity(user_input)
        
        # Display results
        st.subheader('Results:')
        for label, score in results.items():
            yes_no = "Yes" if score > 0.80 else "No"
            st.write(f"{yes_no} - {label.capitalize()}: {score:.2%}")
    else:
        st.write("Please enter a comment to analyze.")
