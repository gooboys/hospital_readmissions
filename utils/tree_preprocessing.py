import pandas as pd

# File paths
input_file_path = "train.csv"
output_file_path = "pruned1.csv"

# Columns to remove
columns_to_remove = [
    "tolbutamide_No",
    "nateglinide_No",
    "miglitol_No",
    "acarbose_No",
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

output_file_path = "pruned2.csv"

# Columns to remove
columns_to_drop = [
    "payer_code_HM",
    "payer_code_SP",
    "payer_code_BC",
    "medical_specialty_Emergency/Trauma",
    "medical_specialty_Family/GeneralPractice",
    "medical_specialty_Cardiology",
    "diag_1_414",
    "diag_1_786",
    "diag_2_250",
    "diag_2_427",
    "diag_3_276",
    "diag_3_428",
    "max_glu_serum_None",
    "repaglinide_No",
    "glimepiride_No",
    "troglitazone_No",
    "pioglitazone_No",
    "rosiglitazone_No",
    "glyburide-metformin_No",
    "glipizide-metformin_No",
    "glimepiride-pioglitazone_No"
]

# Load the dataset
data = pd.read_csv(input_file_path)

# Drop the specified columns
data_cleaned = data.drop(columns=columns_to_drop)

# Save the result to a new file
data_cleaned.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to {output_file_path}")