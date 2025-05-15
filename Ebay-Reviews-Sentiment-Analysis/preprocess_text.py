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
    return [token.lemma_ for token in doc if (token.is_alpha) and (token.text not in stop_words)]

df["review title"] = df["review title"].apply(tokenize_and_lemmatize_text)
df["review content"] = df["review content"].apply(tokenize_and_lemmatize_text)

print("----- Finished tokenizing and lemmatizing text -----")


"""
Build Vocabulary
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

print("----- Finished building vocabulary -----")


"""
Encode Tokens and Ratings
"""
X, y = [], []

for row_data in df.values:
    title_tokens = row_data[1]
    content_tokens = row_data[2]

    # Check if the review (title or content) contains any tokens
    if len(title_tokens) + len(content_tokens) > 0:
        # Initialize a list of length MAX_SEQ_LENGTH filled with 0s (special token <PAD>)
        encoded_tokens = [vocab["<PAD>"]] * MAX_SEQ_LENGTH
        token_index = 0

        # Check if the review title contains any tokens
        if len(title_tokens) > 0:
            # Add special token <TTL> to mark the beginning of the title
            encoded_tokens[token_index] = vocab["<TTL>"]
            token_index += 1

            # Encode title tokens
            # Truncate if necessary to fit within max length (Always leave room for the special token <TTL> marking end of the title)
            # The tokens not included in vocabulary will be replaced by the special token <UNK> (unknown)
            tokens_available = MAX_SEQ_LENGTH - token_index - 1
            for token in title_tokens[:tokens_available]:
                encoded_tokens[token_index] = vocab.get(token, vocab["<UNK>"])
                token_index += 1

            # Add special token <TTL> to mark the end of the title
            encoded_tokens[token_index] = vocab["<TTL>"]
            token_index += 1

        # Encode content tokens
        # Truncate if necessary to fit within max length
        # The tokens not included in vocabulary will be replaced by the special token <UNK> (unknown)
        tokens_available = MAX_SEQ_LENGTH - token_index
        for token in content_tokens[:tokens_available]:
            encoded_tokens[token_index] = vocab.get(token, vocab["<UNK>"])
            token_index += 1

        X.append(encoded_tokens)
        y.append(rating_to_sentiment[row_data[3]])

X = np.array(X, dtype=np.uint16)
y = np.array(y, dtype=np.uint8)

print("----- Finished encoding tokens and ratings -----")


"""
Save Preprocessed Data as Numpy Arrays
"""
np.savez(f"Ebay-Reviews-Sentiment-Analysis/Data/Preprocessed-Dataset.npz", X=X, y=y)

print(f"----- Preprocessed dataset saved to 'Ebay-Reviews-Sentiment-Analysis/Data/Preprocessed-Dataset.npz' -----")
