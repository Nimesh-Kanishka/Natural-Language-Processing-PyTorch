In this project, the data from the ebay_reviews.csv file (downloaded from Kaggle; link below) was first preprocessed with the preprocess_text.py file (removed null rows, cleaned the text, tokenized, removed stopwords, lemmatized and padded sequences) and 2 Numpy arrays were created including the encoded tokens and the corresponding labels. Then a PyTorch neural network module (in network.py file) was trained to perform sentiment analysis on the preprocessed data. The network is trained to predict whether a certain user's review is positive, neutral or negative.

Dataset Used: Ebay Reviews Dataset (https://www.kaggle.com/datasets/wojtekbonicki/ebay-reviews)

Test Accuracy of the Trained Model: 94.3%
