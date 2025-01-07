import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
data = pd.read_csv("train.csv")

# The vector of thresholds
thresholds = [np.int64(12), np.int64(96), np.int64(5), np.int64(35), 
              np.int64(0), np.int64(0), np.int64(2), np.int64(13)]

# Step 1: Cap values larger than the thresholds
for i, col in enumerate(data.columns[:8]):  # Loop over the first 8 columns
    data[col] = np.where(data[col] > thresholds[i], thresholds[i] + 1, data[col])

# Step 2: Scale values in the first 8 columns to the range [0, 1]
scaler = MinMaxScaler()
data[data.columns[:8]] = scaler.fit_transform(data[data.columns[:8]])

# Step 3: Write the updated dataset to a new CSV file
output_file = "zero_to_one.csv"
data.to_csv(output_file, index=False)

print(f"Transformed dataset has been written to '{output_file}'.")
