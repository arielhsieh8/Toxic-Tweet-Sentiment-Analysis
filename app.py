import streamlit as st  #Web App
from transformers import pipeline
from pysentimiento import create_analyzer



model = st.selectbox("Which pretrained model would you like to use?",("DistilBERT","twitter-XLM-roBERTa-base","bertweet-sentiment-analysis"))

#title
st.title("Sentiment Analysis - Classify Sentiment of text")

data = []
text = st.text_input("Enter text here:","Artificial Intelligence is useful")
data.append(text)
if model == "DistilBERT":
    #1
    if st.button("Run Sentiment Analysis of Text"): 
        model_path = "distilbert-base-uncased-finetuned-sst-2-english"
        sentiment_pipeline = pipeline("sentiment-analysis",model=model_path, tokenizer=model_path)
        result = sentiment_pipeline(data)
        label = result[0]["label"]
        score = result[0]["score"]
        st.write("The classification of the given text is " + label + " with a score of " + str(score))
elif model == "Twitter-roBERTa-base":
    #2
    if st.button("Run Sentiment Analysis of Text"): 
        model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
        result = sentiment_task(text)
        st.write(result)

elif model == "bertweet-sentiment-analysis": 
    #3 
    if st.button("Run Sentiment Analysis of Text"): 
        analyzer = create_analyzer(task="sentiment", lang="en")
        result = analyzer.predict(text)
        st.write(result)




