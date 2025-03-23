import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import random


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility in GPU computations
    torch.backends.cudnn.benchmark = False

# Call the function with your desired seed value
set_seed(42)  # You can change 42 to any integer seed


train_seq = pd.read_csv('../datasets/train/train_text_seq.csv')
valid_seq = pd.read_csv('../datasets/valid/valid_text_seq.csv')
test_seq = pd.read_csv('../datasets/test/test_text_seq.csv')

# Remove zeros from the input strings and convert to list of integers
train_seq['input_str'] = train_seq['input_str'].str.replace('0', '')
valid_seq['input_str'] = valid_seq['input_str'].str.replace('0', '')

# Convert input_str to list of integers
def str_to_int_list(s):
    return [int(char) for char in s]

train_seq['input_int_list'] = train_seq['input_str'].apply(str_to_int_list)
valid_seq['input_int_list'] = valid_seq['input_str'].apply(str_to_int_list)

# Get the maximum sequence length to pad all sequences to the same length
max_seq_len = max(train_seq['input_int_list'].apply(len).max(), valid_seq['input_int_list'].apply(len).max())

# Padding the sequences to the maximum length with zeros
def pad_sequence(seq, max_len):
    return seq + [0] * (max_len - len(seq))

train_seq['padded_input'] = train_seq['input_int_list'].apply(lambda x: pad_sequence(x, max_seq_len))
valid_seq['padded_input'] = valid_seq['input_int_list'].apply(lambda x: pad_sequence(x, max_seq_len))

# Convert the labels into a format suitable for classification
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_seq['label'])
valid_labels = label_encoder.transform(valid_seq['label'])

# Prepare data for PyTorch
X_train = torch.tensor(train_seq['padded_input'].tolist(), dtype=torch.float32)
X_valid = torch.tensor(valid_seq['padded_input'].tolist(), dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)
y_valid = torch.tensor(valid_labels, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = TensorDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)  # Initial cell state
        
        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h_0, c_0))
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Pass it through the fully connected layer
        out = self.fc(out)
        return out

# Hyperparameters
input_size = 1  # Each digit is treated as one input feature
hidden_size = 128
num_layers = 2
num_classes = 2  # Binary classification (0 or 1)
num_epochs = 10
learning_rate = 0.001

# Initialize the model, loss function, and optimizer
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.unsqueeze(-1)  # Adding an extra dimension for LSTM
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Validation loop
model.eval()
with torch.no_grad():
    y_pred_valid = []
    y_true_valid = []
    for inputs, labels in valid_loader:
        inputs = inputs.unsqueeze(-1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred_valid.extend(predicted.cpu().numpy())
        y_true_valid.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    print(f'Validation Set Accuracy: {accuracy * 100:.2f}%')
