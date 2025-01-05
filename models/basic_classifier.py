import torch.nn as nn

# Define the neural network
class BasicClassifier(nn.Module):
    def __init__(self, input_size):
        super(BasicClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)         # Second fully connected layer
        self.fc3 = nn.Linear(64, 1)           # Output layer
        self.relu = nn.ReLU()                 # Activation function
        self.dropout = nn.Dropout(0.5)        # Dropout for regularization

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here; we'll use BCEWithLogitsLoss
        return x