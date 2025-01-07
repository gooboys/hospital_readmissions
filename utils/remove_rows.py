import pandas as pd

# This file prunes the data for certain features

# File paths
input_file_path = "zero_to_one.csv"
output_file_path = "zero2one_noshapremoved.csv"

# Columns to remove
columns_to_remove = [
    "tolbutamide_No",
    "miglitol_No",
    "citoglipton_No",
    "metformin-pioglitazone_No",
    "tolazamide_No",
    "chlorpropamide_No",
    "examide_No",
    "metformin-rosiglitazone_No",
    "acetohexamide_No"
]

# Load the dataset
data = pd.read_csv(input_file_path)

# Drop the specified columns
data_cleaned = data.drop(columns=columns_to_remove)

# Save the result to a new file
data_cleaned.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to {output_file_path}")