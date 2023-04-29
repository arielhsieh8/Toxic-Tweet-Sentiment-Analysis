# cs-uy-4613-project milestone 4

Landing Page: https://sites.google.com/nyu.edu/cs-uy-4613-intro-to-ai/home 

HF Space: https://huggingface.co/spaces/Ariel8/Toxic-Tweets

--------------

Project Documentation: 

Model_FineTune.py 

The script trains a model by using the trainer function in the transformers library. It takes a model, in this case, the Distilbert based uncased model, and finetunes the model based on the dataset of toxic tweets that is given in the train.csv file. The training vs validation split was 80% to 20%. The training took approximately 2 hours to complete, based upon the batch size and epochs that the trainer was given. The loss of each step of the training had a generally decreasing trend, and the loss started from 0.522000 and went down to 0.022900 at the end. The test accuracy of the model, which is then calculated on a test set of 1000 values that the trainer did not see, ended up being 0.936, which presents as a high accuracy. Therefore, the finetuning of the model was successful in the multi-class classification task that it was given. The finetuned model was saved and uploaded to the HF repository, where it is accessible to be used through importing, as shown in the App.py file. 

--

App.py

The script utilizes the streamlit API to connect to display the trained model and its functionalities in the huggingface space. Once the user navigates to the HF space, there is a drop down box that allows the user to select which pretrained model they would like to use. 

The first model under "Ariel8-toxic-tweets-classification" is the model that was finetuned in the Model_FineTune.py that detect the level of toxicity in a tweet or text, and then identifies the category of toxicity that the text falls under and its score as well. The table is prepopulated with 10 tweets to demonstrate its functionality. However, the user can also choose to input text of their own to analyze with the model. Once they have written the text they'd like to analyze, users can push the button to run the model which will tell the model to classify both the prepopulated tweets and the user inputted text. 

The second and third models are pretrained models from the HF site that utilize pipeline to easily perform sentiment analysis on text. The user can input any text they like, and after pushing the button to analyze the text, the model will perform sentiment analysis and output whether the text is Positive, Neutral, or Negative, and the percentage score of how sure it is of that conclusion. 



