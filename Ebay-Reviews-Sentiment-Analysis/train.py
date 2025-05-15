import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from network import SentimentAnalyzerNetwork
from config import VOCAB_SIZE, TRAINING_SPLIT, VALIDATION_SPLIT, BATCH_SIZE, NUM_EPOCHS

# Load the preprocessed dataset
preprocessed_dataset = np.load("Ebay-Reviews-Sentiment-Analysis/Data/Preprocessed-Dataset.npz")
X = preprocessed_dataset["X"] # (num_reviews, max_seq_length)
y = preprocessed_dataset["y"] # (num_reviews,)

# Split the dataset into training, validation and test sets
train_end = int(X.shape[0] * TRAINING_SPLIT)
X_train = X[:train_end].astype(np.int64)
y_train = y[:train_end].astype(np.int64)

val_end = train_end + int(X.shape[0] * VALIDATION_SPLIT)
X_val = X[train_end:val_end].astype(np.int64)
y_val = y[train_end:val_end].astype(np.int64)

X_test = X[val_end:].astype(np.int64)
y_test = y[val_end:].astype(np.int64)

# Training, validation and test datasets
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

# Training, validation and test dataloaders
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vocab_size = VOCAB_SIZE + 3 to accomodate the 3 spacial tokens <PAD>, <TTL> and <UNK>
model = SentimentAnalyzerNetwork(vocab_size=VOCAB_SIZE + 3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

best_val_loss = 1000000
best_val_acc = 0

for epoch in range(NUM_EPOCHS):
    print(f"----- Epoch {epoch + 1}/{NUM_EPOCHS} -----")

    """
    Training
    """
    # Set the model to training mode
    model.train()

    total_train_loss = 0
    correct_preds = 0

    for X_batch, y_batch in tqdm(train_loader):
        # Move data batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Get predictions from the model
        y_pred = model(X_batch) # [batch_size, 3]

        # Calculate the loss
        loss = F.cross_entropy(y_pred, y_batch)

        total_train_loss += loss.item()
        correct_preds += (torch.argmax(y_pred, dim=1) == y_batch).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate training loss and accuracy
    train_loss = total_train_loss / len(train_loader)
    train_loss_history.append(train_loss)

    train_acc = correct_preds / len(train_loader.dataset)
    train_acc_history.append(train_acc)

    print(f"Train loss: {train_loss:.4f} - Train accuracy: {(train_acc * 100):.2f}%")

    """
    Validation
    """
    # Set the model to evaluation mode
    model.eval()

    total_val_loss = 0
    correct_preds = 0
    
    for X_batch, y_batch in val_loader:
        # Move data batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Get predictions from the model
        with torch.no_grad():
            y_pred = model(X_batch) # [batch_size, 3]

        # Calculate the loss and number of correct predictions
        total_val_loss += F.cross_entropy(y_pred, y_batch).item()
        correct_preds += (torch.argmax(y_pred, dim=1) == y_batch).sum().item()

    # Calculate validation loss and accuracy
    val_loss = total_val_loss / len(val_loader)
    val_loss_history.append(val_loss)

    val_acc = correct_preds / len(val_loader.dataset)
    val_acc_history.append(val_acc)

    print(f"Validation loss: {val_loss:.4f} - Validation accuracy: {(val_acc * 100):.2f}%")

    # Update the optimizer's learning rate after every epoch
    scheduler.step()

    # Save model checkpoint if we have completed more than 3 epochs AND validation accuracy has increased or
    # validation loss has decreased with validation accuracy remaining the same
    if (epoch > 2) and ((val_acc > best_val_acc) or ((val_acc == best_val_acc) and (val_loss <= best_val_loss))):
        best_val_loss = val_loss
        best_val_acc = val_acc
        
        # Save model checkpoint
        torch.save(model.state_dict(),
                   "Ebay-Reviews-Sentiment-Analysis/Training/Model_Checkpoint.pth")
        
        print(f"----- Model checkpoint saved to 'Ebay-Reviews-Sentiment-Analysis/Training/Model_Checkpoint.pth' -----")

# Plot training and validation loss and accuracy
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
epochs = [i for i in range(1, NUM_EPOCHS + 1)]

# Training/Validation Loss vs Epoch graph
axes[0].plot(epochs, train_loss_history, label="Train Loss")
axes[0].plot(epochs, val_loss_history, label="Validation Loss")
axes[0].set_title("Loss vs Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# Training/Validation Accuracy vs Epoch graph
axes[1].plot(epochs, train_acc_history, label="Train Accuracy")
axes[1].plot(epochs, val_acc_history, label="Validation Accuracy")
axes[1].set_title("Accuracy vs Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.savefig("Ebay-Reviews-Sentiment-Analysis/Training/Training_Graph.png")
plt.show()

"""
Testing
"""
# Load the saved weights from the best epoch and set the model to evaluation mode
model.load_state_dict(torch.load("Ebay-Reviews-Sentiment-Analysis/Training/Model_Checkpoint.pth"))
model.eval()

total_test_loss = 0
correct_preds = 0

for X_batch, y_batch in test_loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        y_pred = model(X_batch) # [batch_size, 3]

    total_test_loss += F.cross_entropy(y_pred, y_batch).item()
    correct_preds += (torch.argmax(y_pred, dim=1) == y_batch).sum().item()

test_loss = total_test_loss / len(test_loader)
test_acc = correct_preds / len(test_loader.dataset)

print(f"Test loss: {test_loss:.4f} - Test accuracy: {(test_acc * 100):.2f}%")