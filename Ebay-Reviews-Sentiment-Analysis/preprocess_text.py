import re
from collections import Counter
import pickle
import numpy as np
import pandas as pd
import spacy
# spacy.cli.download("en_core_web_sm")
# import nltk
# nltk.download("stopwords")
from nltk.corpus import stopwords
from config import DATASET_PATH, VOCAB_SIZE, MAX_SEQ_LENGTH

# Load the dataset as a pandas dataframe
df = pd.read_csv(DATASET_PATH)

"""
Remove Null Rows
"""
# Retain only the rows that contain a review title or review content AND a rating
df = df[((df["review title"].notna()) | (df["review content"].notna())) & (df["rating"].notna())]
df = df[(df["review title"].str.strip() != "") | (df["review content"].str.strip() != "")]

"""
Clean Text
"""
def clean_text(text):
    # Remove URLs from the text by replacing any string that starts with "http"
    # followed by non-whitespace characters (\S+) with an empty string ("")
    text = re.sub(r"http\S+", "", text)

    # Remove punctuation, emojis, symbols and other special characters by replacing
    # all characters except letters (a-z, A-Z), digits (0-9), whitespaces (\s) and
    # apostrophes (') with an empty string ("")
    text = re.sub(r"[^a-zA-Z0-9\s']", "", text)

    # Replace multiple spaces/tabs/newlines (\s+) with a single space (" ")
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing whitespaces and convert all characters in the
    # string to lowercase
    return text.strip().lower()

df["review title"] = df["review title"].astype(str).apply(clean_text)
df["review content"] = df["review content"].astype(str).apply(clean_text)

print("----- Finished cleaning text -----")

"""
Tokenize, Remove Stopwords and Lemmatize
"""
# Load the English stopwords into a Python set
stop_words = set(stopwords.words("english"))
# Load SpaCy’s English language model
nlp = spacy.load("en_core_web_sm")

def tokenize_and_lemmatize_text(text):
    # Process the input text using SpaCy’s English language model and return a Doc
    # object containing tokens and linguistic features
    doc = nlp(text)

    # Remove all non-alphabetical tokens and stopwords and return the base forms
    # (lemmas) of the remaining tokens
    return [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]

df["review title"] = df["review title"].apply(tokenize_and_lemmatize_text)
df["review content"] = df["review content"].apply(tokenize_and_lemmatize_text)

print("----- Finished tokenizing and lemmatizing text -----")

"""
Build Vocabulary and Encode Tokens and Ratings to Numpy Arrays
"""
# Add all tokens from all review titles and review content into a single list
all_tokens = [token for row in zip(df["review title"], df["review content"]) for token_list in row for token in token_list]
# Count the frequency of each word in the list
word_counts = Counter(all_tokens)
# Build a vocabulary from the top VOCAB_SIZE most common words
# +3 to reserve space for special tokens <PAD>, <TTL>, and <UNK>
vocab = {word: idx + 3 for idx, (word, _) in enumerate(word_counts.most_common()[:VOCAB_SIZE])}
vocab["<PAD>"] = 0
vocab["<TTL>"] = 1
vocab["<UNK>"] = 2

# Save the vocabulary (as we need it later to encode tokens during inference)
with open("Ebay-Reviews-Sentiment-Analysis/Data/Vocabulary.pkl", "wb") as f:
    pickle.dump(vocab, f)

# Mapping of the ratings to the corresponding sentiments
rating_to_sentiment = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

# Initialize X and y as numpy arrays filled with zeros (to make sure there is enough memory to store the arrays)
# Use unsigned 16-bit integers to represent tokens (can represent upto 65535)
X = np.zeros((len(df.values), MAX_SEQ_LENGTH), dtype=np.uint16)
# Use unsigned 8-bit integers to represent sentiments (labels)
y = np.zeros(len(df.values), dtype=np.uint8)

for row_index, row_data in enumerate(df.values):
    # Add special token <TTL> to mark the beginning of the title
    X[row_index, 0] = vocab["<TTL>"]
    token_index = 1

    # Encode title tokens into X (Truncate if necessary to fit within max length)
    # The tokens not included in vocabulary will be replaced by the special token <UNK> (unknown)
    for token in row_data[1][:len(row_data[1]) if len(row_data[1]) < (MAX_SEQ_LENGTH - token_index) else MAX_SEQ_LENGTH - token_index]:
        X[row_index, token_index] = vocab.get(token, vocab["<UNK>"])
        token_index += 1

    # Add the special token to mark the end of the title and content tokens only if the number of tokens already
    # added (from the title) does not exceed MAX_SEQ_LENGTH
    if token_index < MAX_SEQ_LENGTH:
        # Add special token <TTL> to mark the end of the title
        X[row_index, token_index] = vocab["<TTL>"]
        token_index += 1

        # Encode content tokens into X (Truncate if necessary to fit within max length)
        # The tokens not included in vocabulary will be replaced by the special token <UNK> (unknown)
        for token in row_data[2][:len(row_data[2]) if len(row_data[2]) < (MAX_SEQ_LENGTH - token_index) else MAX_SEQ_LENGTH - token_index]:
            X[row_index, token_index] = vocab.get(token, vocab["<UNK>"])
            token_index += 1

    # Assign sentiment label
    y[row_index] = rating_to_sentiment.get(row_data[3], 0)

print("----- Finished encoding tokens and ratings -----")

"""
Save Preprocessed Data as Numpy Arrays
"""
np.savez(f"Ebay-Reviews-Sentiment-Analysis/Data/Preprocessed-Dataset.npz", X=X, y=y)

print(f"----- Preprocessed dataset saved to 'Ebay-Reviews-Sentiment-Analysis/Data/Preprocessed-Dataset.npz' -----")