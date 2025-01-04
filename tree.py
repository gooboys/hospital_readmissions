import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class
class ReadmissionDataset(Dataset):
    def __init__(self, csv_file, target_column):
        # Load data
        self.data = pd.read_csv(csv_file)
        self.features = self.data.drop(columns=[target_column])
        self.target = self.data[target_column]

        # Convert data to tensors
        self.features = torch.tensor(self.features.values, dtype=torch.float32)
        self.target = torch.tensor(self.target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]
    
# Dataset and DataLoader Initialization
csv_file_path = "pruned1.csv"
target_column = "readmitted"

# Load the CSV file
data = pd.read_csv(csv_file_path)

# Filter the rows with target value 0 and 1
data_zeros = data[data[target_column] == 0]
data_ones = data[data[target_column] == 1]

# Randomly select 2180 samples from rows where target is 0
data_zeros_sampled = data_zeros.sample(n=2180, random_state=42)

# Combine the sampled zeros with all ones
balanced_data = pd.concat([data_zeros_sampled, data_ones])

# Shuffle the balanced data
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced data to a new CSV file
balanced_csv_file_path = "balanced_pruned1.csv"
balanced_data.to_csv(balanced_csv_file_path, index=False)

# Use the new balanced CSV file for the dataset
csv_file_path = balanced_csv_file_path

# Dataset and DataLoader Initialization
dataset = ReadmissionDataset(csv_file=csv_file_path, target_column=target_column)

# Split indices into train and test sets 80-20 training split because tree model is simple
train_indices, test_indices = train_test_split(
    range(len(dataset)), test_size=0.2, random_state=42
)

# Create subsets for training and testing
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders for training and testing
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example: Checking shapes from train_loader
for batch_features, batch_targets in train_loader:
    print("Train Batch Features Shape:", batch_features.shape)
    print("Train Batch Targets Shape:", batch_targets.shape)
    break

# Example: Checking shapes from test_loader
for batch_features, batch_targets in test_loader:
    print("Test Batch Features Shape:", batch_features.shape)
    print("Test Batch Targets Shape:", batch_targets.shape)
    break

# Function to extract features and targets from DataLoader
def dataloader_to_numpy(dataloader):
    features, targets = [], []
    for batch_features, batch_targets in dataloader:
        features.append(batch_features.numpy())
        targets.append(batch_targets.numpy())
    return np.vstack(features), np.hstack(targets)

# Extract train and test data from DataLoaders
X_train, y_train = dataloader_to_numpy(train_loader)
X_test, y_test = dataloader_to_numpy(test_loader)

# Train the Random Forest classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=None,  # Maximum depth of trees
    random_state=42,  # For reproducibility
    n_jobs=-1,  # Use all available processors
    class_weight="balanced"  # Handle imbalanced datasets
)
rf_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
importances = rf_clf.feature_importances_
feature_names = dataset.data.drop(columns=[target_column]).columns
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop Features:")
print(feature_importance_df.head(10))