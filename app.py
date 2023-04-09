import streamlit as st  #Web App
from transformers import pipeline

#title
st.title("Sentiment Analysis - Classify Sentiment of text")

data = []
text = st.text_input("Enter text here:","Artificial Intelligence is useful")
if st.button("Run Sentiment Analysis of Text"): 
    data.append(text)
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(data)
    label = result[0]["label"]
    score = result[0]["score"]
    st.write("The classification of the given text is " + label + " with a score of " + str(score))



