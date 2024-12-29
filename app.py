import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


## Load imdb dataset
word_index = imdb.get_word_index()
index_word = {i: w for w, i in word_index.items()}

## Load model
model = load_model('rnn_model.keras')

def decode_review(review):
    return ' '.join([index_word.get(i - 3, '?') for i in review])

## preprocess function
def preprocess(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=400)
    return padded_review

## predict function
def predict(text):
    preprocessed_text = preprocess(text)
    prediction = model.predict(preprocessed_text)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

import streamlit as st

st.title('Simple RNN')
st.write('This is a simple RNN model for sentiment analysis')

text = st.text_area('Enter your review')

if st.button('Predict'):
    sentiment, confidence = predict(text)
    
    # Highlight prediction in green if positive sentiment
    if sentiment == 'Positive':
        sentiment_color = 'green'
    else:
        sentiment_color = 'red'
    
    st.markdown(f'<h3 style="color:{sentiment_color}">Sentiment: {sentiment}</h3>', unsafe_allow_html=True)
    
    if sentiment == 'Positive':
        st.write(f'Confidence: {confidence:.4f}')
    else:
        st.write(f'Confidence: {1 - confidence:.4f}')
else:
    st.write('Enter a review and click on the predict button')
