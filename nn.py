import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader 
from collections import defaultdict
import numpy as np
# Change which classifier is being imported to change which model is being tested
from models import BasicClassifier, DeepClassifier, DeeperClassifier, DeepWideClassifier, DeepestClassifier
from models import DeeperWideClassifier, DeepestWideClassifier, DeepestFunClassifier

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
csv_file_path = "zero_to_one.csv"
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
feature_count = 64

# Example: Checking shapes from train_loader
for batch_features, batch_targets in train_loader:
    print("Train Batch Features Shape:", batch_features.shape)
    print("Train Batch Targets Shape:", batch_targets.shape)
    feature_count = batch_features.shape[1]
    break

# Declares what the classifier is from the imported classes
BinaryClassifier = BasicClassifier

# Set input size based on your dataset
input_size = feature_count  # Number of input features
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

# Training the model with Early Stopping
def train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, patience=5):
    best_loss = float('inf')  # Initialize best loss as infinity
    patience_counter = 0      # Counter for early stopping
    best_model_state = None   # To store the best model's state

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(features).squeeze()  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        # Compute training loss
        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)  # Compute loss
                val_loss += loss.item()

        val_loss /= len(test_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check for improvement in validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  # Reset patience counter
            best_model_state = model.state_dict()  # Save the best model's state
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")

            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # Load the best model's state
    if best_model_state:
        model.load_state_dict(best_model_state)

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
    report = classification_report(y_true, y_pred, output_dict=True)
    return report

# # Train and evaluate
# num_epochs = 10
# train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
# evaluate_model(model, test_loader)

# Train with Early Stopping and evaluate
# num_epochs = 50
# patience = 5
# train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs, patience=patience)
# evaluate_model(model, test_loader)

def monteCarlo(runs, model, criterion, optimizer, num_epochs=50, patience=5):
    reports = []
    for i in range(runs):
        # Randomly split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state + i)

        # Normalize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create DataLoaders
        train_dataset = ReadmissionDataset(X_train, y_train)
        test_dataset = ReadmissionDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize a fresh model and optimizer for each run
        model = BinaryClassifier(X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters())

        train_model_with_early_stopping(model, train_loader, test_loader, criterion, optimizer, num_epochs=num_epochs, patience=patience)
        report = evaluate_model(model, test_loader)
        reports.append(report)
    # Aggregate metrics
    aggregated_metrics = defaultdict(lambda: defaultdict(list))  # For per-class and average metrics
    scalar_metrics = defaultdict(list)  # For scalar metrics like "accuracy"

    # Gather metrics for each class and overall averages
    for report in reports:
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):  # For dictionary-based metrics (e.g., precision, recall, f1-score)
                for metric_name, value in metrics.items():
                    aggregated_metrics[class_name][metric_name].append(value)
            else:  # For scalar metrics (e.g., "accuracy")
                scalar_metrics[class_name].append(metrics)

    # Compute means for each metric
    mean_metrics = {}

    # Compute mean for dictionary-based metrics
    for class_name, metrics in aggregated_metrics.items():
        mean_metrics[class_name] = {metric: float(np.mean(values)) for metric, values in metrics.items()}

    # Compute mean for scalar metrics
    for class_name, values in scalar_metrics.items():
        mean_metrics[class_name] = float(np.mean(values))

    # Print the aggregated report
    print("\nAverage Report:")
    for class_name, metrics in mean_metrics.items():
        if isinstance(metrics, dict):
            print(f"{class_name}: {metrics}")
        else:
            print(f"{class_name}: {metrics:.4f}")

        print("")
        print("")
        print("Average Report:")

        # Print the aggregated report
        for class_name, metrics in mean_metrics.items():
            print(f"{class_name}: {metrics}")

# CODE BELOW FOR SHAP ANALYSIS
# def model_predict(features):
#     """
#     Takes input features, runs them through the trained model, and returns predictions.
#     This wrapper is required because SHAP expects a callable function for the model.
#     """
#     model.eval()
#     with torch.no_grad():
#         features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
#         logits = model(features_tensor).squeeze().cpu().numpy()
#         return torch.sigmoid(torch.tensor(logits)).numpy()  # Return probabilities
    
# # Step 2: Select a background dataset for SHAP
# background = X_train[:100]  # Use a small subset of the training data for efficiency

# # Step 3: Initialize the SHAP Explainer
# explainer = shap.Explainer(model_predict, background)

# # Step 4: Generate SHAP values for the test set
# shap_values = explainer(X_test)

# # Step 5: Visualize SHAP results
# # Summary plot (overall feature importance)
# shap.summary_plot(shap_values, X_test)

# # Dependence plot for a specific feature
# shap.dependence_plot(0, shap_values.values, X_test)  # Replace 0 with the desired feature index



# BELOW CODE FOR MONTE CARLO VALIDATION

num_epochs = 50
patience = 5
runs = 10
monteCarlo(runs, model, criterion, optimizer)