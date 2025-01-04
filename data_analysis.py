import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import csv

# Load the dataset
data = pd.read_csv("train.csv")

# PRINT STATEMENTS FOR INITIAL VIEW
print(data.info())
print(data.head())
print(data.isnull().sum().to_string())
print(data.describe())
# First 8 columns are ints, remaining are boolean values

# Prepare features and target
X = data.drop('readmitted', axis=1)
y = data['readmitted']

# Initialize a DataFrame to store results
feature_analysis = pd.DataFrame({'Feature': X.columns})

# Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)
feature_analysis['Mutual Information'] = mi_scores

# Point-Biserial Correlation for integer features
point_biserial_scores = []

# Ensure to use original integer columns from `data`
integer_columns = [col for col in data.columns if data[col].dtype in ['int64', 'int32'] and col != 'readmitted']  # Exclude target

for col in X.columns:
    if col in integer_columns:  # Check if the column is in the list of integer columns
        correlation, _ = pointbiserialr(data[col], y)  # Use original data for calculation
        point_biserial_scores.append(correlation)
    else:
        point_biserial_scores.append(None)  # Append None for non-integer columns

# Add the Point-Biserial Correlation scores to the feature_analysis DataFrame
feature_analysis['Point-Biserial Correlation'] = point_biserial_scores

# Chi-Square test
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
chi2_scores, p_values = chi2(X_scaled, y)
feature_analysis['Chi-Square Score'] = chi2_scores
feature_analysis['Chi-Square p-value'] = p_values

# Random Forest Feature Importance
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_analysis['Random Forest Importance'] = feature_importances.reindex(feature_analysis['Feature']).values

# Sort by Mutual Information (or any metric of choice)
feature_analysis.sort_values(by='Mutual Information', ascending=False, inplace=True)

# Print consolidated results
print(feature_analysis)

# Save consolidated results to an Excel file
feature_analysis.to_csv("feature_analysis_results.csv", index=False)

# Extract and print the top features for each metric

# Number of top features to display
top_n = 64

# Mutual Information
print("\nTop Features by Mutual Information:")
top_mi = feature_analysis[['Feature', 'Mutual Information']].sort_values(by='Mutual Information', ascending=False).head(top_n)
for index, row in top_mi.iterrows():
    print(f"{row['Feature']}: {row['Mutual Information']:.4f}")

# Point-Biserial Correlation
print("\nTop Features by Point-Biserial Correlation:")
top_pb = feature_analysis[['Feature', 'Point-Biserial Correlation']].dropna().sort_values(by='Point-Biserial Correlation', ascending=False).head(top_n)
for index, row in top_pb.iterrows():
    print(f"{row['Feature']}: {row['Point-Biserial Correlation']:.4f}")

# Chi-Square Scores
print("\nTop Features by Chi-Square Score (with p-value):")
top_chi2 = feature_analysis[['Feature', 'Chi-Square Score', 'Chi-Square p-value']].sort_values(by='Chi-Square Score', ascending=False).head(top_n)
for index, row in top_chi2.iterrows():
    print(f"{row['Feature']}: Score = {row['Chi-Square Score']:.4f}, p-value = {row['Chi-Square p-value']:.4e}")

# Random Forest Feature Importance
print("\nTop Features by Random Forest Importance:")
top_rf = feature_analysis[['Feature', 'Random Forest Importance']].sort_values(by='Random Forest Importance', ascending=False).head(top_n)
for index, row in top_rf.iterrows():
    print(f"{row['Feature']}: {row['Random Forest Importance']:.4f}")

print(feature_analysis.columns)

# Count rows where the last column ends with 1 or 0
num_rows_end_1 = data[data.iloc[:, -1] == 1].shape[0]
num_rows_end_0 = data[data.iloc[:, -1] == 0].shape[0]
print(f"Rows ending in 1: {num_rows_end_1}, Rows ending in 0: {num_rows_end_0}")

# Code below used to view rankings of features

# Load the CSV file into a DataFrame
df = pd.read_csv('feature_analysis_results.csv')

# Convert each metric column to ranks
# Replace NaN values with a large rank
df['Mutual Information Rank'] = df['Mutual Information'].rank(ascending=False, method='min').fillna(len(df) + 1)
df['Point-Biserial Rank'] = df['Point-Biserial Correlation'].rank(ascending=False, method='min').fillna(len(df) + 1)
df['Chi-Square Rank'] = df['Chi-Square Score'].rank(ascending=False, method='min').fillna(len(df) + 1)
df['Forest Importance Rank'] = df['Random Forest Importance'].rank(ascending=False, method='min').fillna(len(df) + 1)

# Calculate the sum of ranks
df['Sum of Ranks'] = (
    df['Mutual Information Rank'] +
    df['Point-Biserial Rank'] +
    df['Chi-Square Rank'] +
    df['Forest Importance Rank']
)

# Sort by the sum of ranks
df_sorted = df.sort_values(by='Sum of Ranks')
df_sorted_unique = df_sorted.drop_duplicates()

# Print the header
print("Feature Name, Mutual Information Rank, Point-Biserial Rank, Chi-Square Rank, Forest Importance Rank, Sum of Ranks")

# Print each row in the desired format
for _, row in df_sorted.iterrows():
    print(f"{row['Feature']}, {int(row['Mutual Information Rank'])}, {int(row['Point-Biserial Rank'])}, "
          f"{int(row['Chi-Square Rank'])}, {int(row['Forest Importance Rank'])}, {int(row['Sum of Ranks'])}")
    
# Resulting bottom 23 results, stopping where the cross significance is 9
# diag_2_427
# glimepiride_No
# glipizide-metformin_No
# repaglinide_No
# payer_code_HM
# troglitazone_No

# below have a Point-Biserial Correlation of below 0.05

# tolbutamide_No
# nateglinide_No
# miglitol_No
# acarbose_No
# citoglipton_No
# metformin-pioglitazone_No
# tolazamide_No
# chlorpropamide_No
# examide_No
# metformin-rosiglitazone_No
# acetohexamide_No