import streamlit as st  #Web App
from transformers import pipeline
import numpy as np 
import pandas as pd

#title
st.title("Toxic Tweets")

model = st.selectbox("Which pretrained model would you like to use?",("roberta-large-mnli","twitter-XLM-roBERTa-base","bertweet-sentiment-analysis"))

#d = {'col1':[1,2],'col2':[3,4]}
#data = pd.DataFrame(data=d)
#st.table(data)

data = []
text = st.text_input("Enter text here:","Artificial Intelligence is useful")
data.append(text)

if model == "roberta-large-mnli":
    #1
    if st.button("Run Sentiment Analysis of Text"): 
        model_path = "roberta-large-mnli"
        sentiment_pipeline = pipeline(model=model_path)
        result = sentiment_pipeline(data)
        label = result[0]["label"]
        score = result[0]["score"]
        d = {'tweet':[model_path],'classification':[label],'score':[score]}
        dataframe = pd.DataFrame(data=d)
        st.table(dataframe)
        #st.write("The classification of the given text is " + label + " with a score of " + str(score))


# data = []
# text = st.text_input("Enter text here:","Artificial Intelligence is useful")
# data.append(text)
# if model == "roberta-large-mnli":
#     #1
#     if st.button("Run Sentiment Analysis of Text"): 
#         model_path = "roberta-large-mnli"
#         sentiment_pipeline = pipeline(model=model_path)
#         result = sentiment_pipeline(data)
#         label = result[0]["label"]
#         score = result[0]["score"]
#         st.write("The classification of the given text is " + label + " with a score of " + str(score))
# elif model == "twitter-XLM-roBERTa-base":
#     #2
#     if st.button("Run Sentiment Analysis of Text"): 
#         model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
#         sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
#         result = sentiment_task(text)
#         label = result[0]["label"].capitalize()
#         score = result[0]["score"]
#         st.write("The classification of the given text is " + label + " with a score of " + str(score))

# elif model == "bertweet-sentiment-analysis": 
#     #3 
#     if st.button("Run Sentiment Analysis of Text"): 
#         analyzer = create_analyzer(task="sentiment", lang="en")
#         result = analyzer.predict(text)
#         if result.output == "POS": 
#             label = "POSITIVE"
#         elif result.output == "NEU": 
#             label = "NEUTRAL"
#         else: 
#             label = "NEGATIVE"
        
#         neg = result.probas["NEG"]
#         pos = result.probas["POS"]
#         neu = result.probas["NEU"]
#         st.write("The classification of the given text is " + label + " with the scores broken down as: Positive - " + str(pos) + ", Neutral - " + str(neu) + ", Negative - " + str(neg))




