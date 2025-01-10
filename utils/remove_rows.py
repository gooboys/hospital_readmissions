import pandas as pd

# This file prunes the data for certain features

# File paths
input_file_path = "zero2one_noshapremoved.csv"
output_file_path = "treeparsed.csv"

# Columns to remove
columns_to_remove = [
    "glyburide_No",
    "nateglinide_No",
    "glipizide-metformin_No",
    "diag_3_276",
    "troglitazone_No",
    "glimepiride-pioglitazone_No"
]

# Load the dataset
data = pd.read_csv(input_file_path)

# Drop the specified columns
data_cleaned = data.drop(columns=columns_to_remove)

# Save the result to a new file
data_cleaned.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to {output_file_path}")