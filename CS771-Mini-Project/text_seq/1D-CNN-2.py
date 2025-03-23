import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim
import random
import os 
import sys
sys.path.append(os.path.abspath('../common')) # add path to common functions
from preprocess import getdfs

# Set seed for random number generation
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    # For deterministic behavior on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set a specific seed value (e.g., 42)
set_seed(42)


# Preprocessing function
# def preprocess_data(data, maxlen=None):
#     sequences = [[int(char) for char in list(seq)] for seq in data["input_str"].values]
#     padded_sequences = np.array(
#         [seq + [0] * (maxlen - len(seq)) for seq in sequences]
#     )  # Padding with 0s
#     labels = data["label"].values
#     return padded_sequences, labels


train_data = pd.read_csv("../datasets/train/train_text_seq.csv")
valid_data = pd.read_csv("../datasets/valid/valid_text_seq.csv")
test_data = pd.read_csv("../datasets/test/test_text_seq.csv")
# Get the maximum sequence length from training data
max_seq_len = train_data["input_str"].apply(len).max()
# Preprocess training and validation data
X_train, X_valid = getdfs('text_seq', 1)
y_train = X_train['label'].values
y_valid = X_valid['label'].values

X_train = X_train.values
X_valid = X_valid.values
X_train = [[int(x) for x in list(seq)] for seq in X_train]
X_valid = [[int(x) for x in list(seq)] for seq in X_valid]


print(type(X_train[0][0]))
# X_train, y_train = preprocess_data(train_data, maxlen=max_seq_len)
# X_valid, y_valid = preprocess_data(valid_data, maxlen=max_seq_len)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)


class TextSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# Create dataset objects
train_dataset = TextSequenceDataset(X_train_tensor, y_train_tensor)
valid_dataset = TextSequenceDataset(X_valid_tensor, y_valid_tensor)


# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

import torch.nn as nn


# class CNN1D(nn.Module):
#     def __init__(self, max_seq_len):
#         super(CNN1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         self.pool = nn.AvgPool1d(kernel_size=2)

#         self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
#         # fc1 output size will be calculated dynamically
#         self.fc2 = nn.Linear(64, 1)
#         self.dropout = nn.Dropout(0.6)
#         self.relu = nn.ReLU()

#         # Store max_seq_len to calculate the output size
#         self.max_seq_len = max_seq_len

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool(x)
        
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.pool(x)
        
#         # Dynamically calculate the flattened size for fc1
#         x = x.view(x.size(0), -1)  # Flatten
#         if not hasattr(self, 'fc1'):
#             self.fc1 = nn.Linear(x.size(1), 64)  # Dynamically set input size for fc1
        
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
        
#         return torch.sigmoid(x)

import torch

class CNN1D(torch.nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        
        # First Convolutional Layer: input 1 channel, output 20 channels
        self.layer1 = torch.nn.Conv1d(in_channels=1, out_channels=20, kernel_size=5, stride=2)
        self.act1 = torch.nn.ReLU()
        
        # Second Convolutional Layer: input 20 channels, output 1 channel
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3, stride=1)
        self.act2 = torch.nn.ReLU()

        # Fully connected layer for binary classification
        self.fc = torch.nn.Linear(10 * 23, 1)  # Adjust input size based on output of Conv layers

    def forward(self, x):
        # Forward pass through the layers
        x = self.layer1(x)  # First convolution
        x = self.act1(x)    # Activation
        
        x = self.layer2(x)  # Second convolution
        x = self.act2(x)    # Activation

        # Flatten the output before the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layer
        x = self.fc(x)

        # Applying sigmoid activation for binary classification
        log_probs = torch.sigmoid(x)

        return log_probs


# Initialize the model, loss function, and optimizer
model = CNN1D()
criterion = nn.BCELoss()  # Binary cross-entropy loss
# SGD optimizer with weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for sequences, labels in train_loader:
        sequences = sequences.unsqueeze(1).to(device)  # Add a channel dimension
        labels = labels.unsqueeze(1).to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for sequences, labels in valid_loader:
        sequences = sequences.unsqueeze(1).to(device)
        labels = labels.unsqueeze(1).to(device)

        outputs = model(sequences)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Validation Accuracy: {accuracy * 100:.2f}%")


