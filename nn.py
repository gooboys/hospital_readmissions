import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

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
