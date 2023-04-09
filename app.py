import streamlit as st  #Web App
from transformers import pipeline

#title
st.title("Sentiment Analysis - Classify Sentiment of text")

data = []
text = st.text_input("Enter text here:")
data.append(text)

sentiment_pipeline = pipeline("sentiment-analysis")

result = sentiment_pipeline(data)

st.write(result)


