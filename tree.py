import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

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
csv_file_path = "train.csv"  # Replace with the actual path
target_column = "readmitted"  # Replace with the actual target column name

dataset = ReadmissionDataset(csv_file=csv_file_path, target_column=target_column)

# Split indices into train and test sets 70-30 training split because tree model is simple
train_indices, test_indices = train_test_split(
    range(len(dataset)), test_size=0.3, random_state=42
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


