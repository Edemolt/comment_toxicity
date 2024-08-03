import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the model
model = load_model('my_model.h5')

# Define the labels
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Vectorizer function (replace with your actual vectorizer)
def vectorizer(text):
    # This is a placeholder. Replace with your actual text preprocessing logic.
    # For example, if you're using TF-IDF, load the vectorizer and transform the input text.
    # Example: return tfidf_vectorizer.transform([text]).toarray()
    return np.random.rand(100)  # Assuming the input vector has 100 features

st.title('Toxic Comment Classifier')

# Input text
input_text = st.text_area('Enter your comment:', '')

if st.button('Predict'):
    if input_text:
        # Preprocess the input text
        input_vector = vectorizer(input_text)

        # Ensure the input is the correct shape for the model
        input_vector = np.expand_dims(input_vector, axis=0)  # Add a new axis at position 0 (batch dimension)
        
        # Predict using the model
        prediction = model.predict(input_vector)
        prediction = prediction[0]

        # Display the results
        results = {label: 'Yes' if score > 0.4 else 'No' for label, score in zip(labels, prediction)}

        st.write('Predictions:')
        for label, result in results.items():
            st.write(f'{label}: {result}')
    else:
        st.write('Please enter a comment to classify.')
