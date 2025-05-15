import torch
import torch.nn as nn

class SentimentAnalyzerNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, padding_idx=0):
        super().__init__()

        # Embedding layer: Map each integer token ID to a real-valued vector that is trainable,
        # capturing semantic relationships.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

        # Bidirectional LSTM layer: Capture sequential dependencies. Use both past and future
        # context (bidirectional).
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Global max pooling layer: Apply 1D max pooling across the sequence dimension.
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        # Fully connected layer 1 with ReLU activation
        self.fc1 = nn.Linear(hidden_dim * 2, 64)

        # Dropout layer: Randomly disable 50% of neurons during training to minimize overfitting.
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer 2 with no activation (as we are using cross entropy loss function
        # we should output raw logits from the network)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # [batch_size, max_seq_length] -> [batch_size, max_seq_len, embedding_dim]
        x = self.embedding(x)

        # [batch_size, max_seq_len, embedding_dim] -> [batch_size, max_seq_len, hidden_dim * 2]
        x, _ = self.lstm(x)

        # [batch_size, max_seq_len, hidden_dim * 2] -> [batch_size, hidden_dim * 2, max_seq_len] ->
        # [batch_size, hidden_dim * 2, 1] -> [batch_size, hidden_dim * 2]
        x = self.pool(x.permute(0, 2, 1)).squeeze(dim=-1)

        # [batch_size, hidden_dim * 2] -> [batch_size, 64]
        x = torch.relu(self.fc1(x))

        # [batch_size, 64]
        x = self.dropout(x)

        # [batch_size, 64] -> [batch_size, 3]
        return self.fc2(x)
