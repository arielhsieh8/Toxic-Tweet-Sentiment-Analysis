import streamlit as st  #Web App
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np 
import pandas as pd

#title
st.title("Toxic Tweets")

# model = st.selectbox("Which pretrained model would you like to use?",("roberta-large-mnli","twitter-XLM-roBERTa-base","bertweet-sentiment-analysis"))

#d = {'col1':[1,2],'col2':[3,4]}
#data = pd.DataFrame(data=d)
#st.table(data)

# data = []
# text = st.text_input("Enter text here:","Artificial Intelligence is useful")
# data.append(text)

tokenizer = AutoTokenizer.from_pretrained("Ariel8/toxic-tweets-classification")
model = AutoModelForSequenceClassification.from_pretrained("Ariel8/toxic-tweets-classification")

X_train = ["Why is Owen's retirement from football not mentioned? He hasn't played a game since 2005."]
batch = tokenizer(X_train, truncation=True, padding='max_length', return_tensors="pt")
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

with torch.no_grad():
  outputs = model(**batch)
  predictions = torch.sigmoid(outputs.logits)*100
  probs = predictions[0].tolist()
  for i in range(len(probs)):
    st.write(f"{labels[i]}: {round(probs[i], 3)}%")

# if model == "roberta-large-mnli":
#     #1
#     if st.button("Run Sentiment Analysis of Text"): 
#         model_path = "roberta-large-mnli"
#         sentiment_pipeline = pipeline(model=model_path)
#         result = sentiment_pipeline(data)
#         label = result[0]["label"]
#         score = result[0]["score"]
#         d = {'tweet':[model_path],'classification':[label],'score':[score]}
#         dataframe = pd.DataFrame(data=d)
#         st.table(dataframe)
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




