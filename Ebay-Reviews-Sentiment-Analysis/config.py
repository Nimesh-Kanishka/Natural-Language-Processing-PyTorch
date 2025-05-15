# Path to the dataset (.csv file)
DATASET_PATH = r"D:\Datasets\Ebay-Reviews-Dataset-For-Sentiment-Analysis\ebay_reviews.csv"

# The maximum number of unique words to include in the vocabulary
VOCAB_SIZE = 5000
# Total number of unique tokens available: 37850

# The maximum number of input tokens for the model per review
MAX_SEQ_LENGTH = 100
# Reviews containing 1 - 100 tokens: 43984
# Reviews containing 100 - 200 tokens: 595
# Reviews containing > 200 tokens: 133

# Percentage of the dataset to be used for training
TRAINING_SPLIT = 0.8

# Percentage of the dataset to be used for validation
VALIDATION_SPLIT = 0.1

# Training batch size
BATCH_SIZE = 64

# Number of training epochs
NUM_EPOCHS = 20
