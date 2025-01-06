import torch.nn as nn

# Define the neural network
class DeepestWideClassifier(nn.Module):
    def __init__(self, input_size):
        super(DeepestWideClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 64)          # Second fully connected layer
        self.fc3 = nn.Linear(64, 64)           # Third fully connected layer
        self.fc4 = nn.Linear(64, 64)           # Fourth fully connected layer
        self.fc5 = nn.Linear(64, 64)           # Fifth fully connected layer
        self.fc6 = nn.Linear(64, 1)            # Output layer
        self.relu = nn.ReLU()                 # Activation function
        self.dropout = nn.Dropout(0.5)        # Dropout for regularization

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)  # No activation here; we'll use BCEWithLogitsLoss
        return x