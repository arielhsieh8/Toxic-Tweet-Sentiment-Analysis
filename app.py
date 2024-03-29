import streamlit as st  #Web App
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np 
import pandas as pd

#title
st.title("Toxic Tweet Classification / Sentiment Analysis")

selection = st.selectbox("Select fine-tuned model",("Ariel8/toxic-tweets-classification","roberta-large-mnli","twitter-XLM-roBERTa-base"))

if selection == "Ariel8/toxic-tweets-classification":
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

    text = st.text_input("Enter Text here for Toxicity Classification:","Artificial Intelligence is useful")

    if st.button("Run Toxicity Classification of Text (and prepopulated Tweets)"): 
        tweets.append(text)

        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        main_class = []
        toxic_types = []

        for i in range(len(tweets)):
            batch = tokenizer(tweets[i], truncation=True, padding='max_length', return_tensors="pt") 
            with torch.no_grad():
                outputs = model(**batch)
                predictions = torch.sigmoid(outputs.logits)*100
                probs = predictions[0].tolist()

            first_max = max(probs)
            fm_index = probs.index(first_max)
            main_class.append((first_max,fm_index))
            second_max = max(probs[2:])
            sm_index = probs.index(second_max)
            toxic_types.append((second_max,sm_index))


        d = {'Tweet':tweets,'Highest Class':[labels[main_class[i][1]] for i in range(len(main_class))],'Classification Score':[round(main_class[i][0],3) for i in range(len(main_class))],
                'Toxicity Type':[labels[toxic_types[i][1]] for i in range(len(toxic_types))],'Toxicity Type Score':[round(toxic_types[i][0],3) for i in range(len(toxic_types))]}
        dataframe = pd.DataFrame(data=d)
        st.table(dataframe)
else: 
    data = []
    text = st.text_input("Enter text here for Sentiment Analysis:","Artificial Intelligence is useful")
    data.append(text)
    if selection == "roberta-large-mnli":
        #1
        if st.button("Run Sentiment Analysis of Text"): 
            model_path = "roberta-large-mnli"
            sentiment_pipeline = pipeline(model=model_path)
            result = sentiment_pipeline(data)
            label = result[0]["label"]
            score = result[0]["score"]
            st.write("The classification of the given text is " + label + " with a score of " + str(score))
    elif selection == "twitter-XLM-roBERTa-base":
        #2
        if st.button("Run Sentiment Analysis of Text"): 
            model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
            result = sentiment_task(text)
            label = result[0]["label"].capitalize()
            score = result[0]["score"]
            st.write("The classification of the given text is " + label + " with a score of " + str(score))
   




