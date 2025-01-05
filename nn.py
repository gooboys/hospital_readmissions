import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
random_state = 42

# Custom Dataset Class for Neural Networks
class ReadmissionDataset(Dataset):
    def __init__(self, features, target):
        """
        Initialize the dataset with preprocessed features and target.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# Load data
csv_file_path = "train.csv"
target_column = "readmitted"
data = pd.read_csv(csv_file_path)

# Handle categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Handle missing values
data = data.fillna(data.median())  # Replace missing values with median

# Balance the dataset (if imbalanced)
majority_class = data[data[target_column] == 0]
minority_class = data[data[target_column] == 1]
n_samples = min(len(majority_class), len(minority_class))

balanced_data = pd.concat([
    majority_class.sample(n=n_samples, random_state=random_state),
    minority_class.sample(n=n_samples, random_state=random_state)
]).sample(frac=1, random_state=random_state).reset_index(drop=True)

# Separate features and target
X = balanced_data.drop(columns=[target_column]).values
y = balanced_data[target_column].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

# Normalize the features (standard scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch datasets
train_dataset = ReadmissionDataset(X_train, y_train)
test_dataset = ReadmissionDataset(X_test, y_test)

# DataLoader initialization
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example: Checking shapes from train_loader
for batch_features, batch_targets in train_loader:
    print("Train Batch Features Shape:", batch_features.shape)
    print("Train Batch Targets Shape:", batch_targets.shape)
    break

# Define the neural network
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
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

# Set input size based on your dataset
input_size = 64  # Number of input features
model = BinaryClassifier(input_size).to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid activation + binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Training the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(features).squeeze()  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        # Print epoch loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluate the model
def evaluate_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    y_pred = []
    y_true = []
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features).squeeze()
            preds = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(targets.cpu().numpy())

    # Compute evaluation metrics
    from sklearn.metrics import classification_report, accuracy_score
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

# Train and evaluate
num_epochs = 10
train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
evaluate_model(model, test_loader)
