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

tweets = ["BlackKite being a faggot",
"Yo bitch Ja Rule is more succesful then you'll ever be whats up with you and hating you sad mofuckas...i should bitch slap ur pethedic white faces and get you to kiss my ass you guys sicken me. Ja rule is about pride in da music man. dont diss that shit on him. and nothin is wrong bein like tupac he was a brother too...fuckin white boys get things right next time.,",
":Dear god this site is horrible.",
"I think its crap that the link to roggenbier is to this article. Somebody that knows how to do things should change it.",
":::::I have added more wikilinks to my sections and included more secondary sources as you suggested. For all the citations under human disease section, I went through them and exchanged review articles for the primary sources. Thanks again for all your input!",
"DJ Robinson is gay as hell! he sucks his dick so much!!!!!",
"I WILL BURN YOU TO HELL IF YOU REVOKE MY TALK PAGE ACCESS!!!!!!!!!!!!!",
"um, taking a shot here... Um lets say Three separate people whos wills are to each other.",
"How dare you vandalize that page about the HMS Beagle! Don't vandalize again, demon!",
":Thanks for the comment about Wiki-defenderness. I like that one. I usually wikiling Wiki-defender. I agree that at first he was somewhat innocent but now have my doubts as he is being really agressive about the whole matter."]

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
main_class = []
toxic_type = []

for i in range(len(tweets)):
    batch = tokenizer(tweets[i], truncation=True, padding='max_length', return_tensors="pt") 
    with torch.no_grad():
        outputs = model(**batch)
        predictions = torch.sigmoid(outputs.logits)*100
        probs = predictions[0].tolist()
        # for i in range(len(probs)):
        #     st.write(f"{labels[i]}: {round(probs[i], 3)}%")
        # results.append(probs)
        first_max = max(probs)
        fm_index = probs.index(first_max)
        main_class.append((first_max,fm_index))
        second_max = max(probs[2:])
        sm_index = probs.index(second_max)
        second_class.append((second_max,sm_index))
        

# main_class = []
# toxic_type = []
d = {'tweet':[tweets],'Main Classification':[labels[main_class[i][1]] for i in range(len(main_class))],'Score':[round(main_class[i][1],3) for i in range(len(main_class))],
        'Toxicity Type':[labels[second_class[i][1]] for i in range(len(second_class))],'Toxicity Score':[round(second_class[i][1],3) for i in range(len(second_class))]}
dataframe = pd.DataFrame(data=d)
st.table(dataframe)
   

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




