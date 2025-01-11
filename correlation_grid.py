import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, ttest_ind, pointbiserialr
import dcor
from scipy.stats import chi2_contingency

# Load the dataset (update the file path if necessary)
# If the dataset is already loaded as a DataFrame, skip this line
df = pd.read_csv('train.csv')

df = df.astype(float)

'''
Continuous feature analysis
'''

# # Select the first 8 features
# features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
#             'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
# data = df[features]

# # Initialize matrices for correlations
# n_features = len(features)
# spearman_matrix = np.zeros((n_features, n_features))
# pearson_matrix = np.zeros((n_features, n_features))
# distance_matrix = np.zeros((n_features, n_features))

# # Compute the correlation matrices
# for i in range(n_features):
#     for j in range(n_features):
#         # Spearman Correlation
#         spearman_corr, _ = spearmanr(data.iloc[:, i], data.iloc[:, j])
#         spearman_matrix[i, j] = spearman_corr

#         # Pearson Correlation
#         pearson_corr, _ = pearsonr(data.iloc[:, i], data.iloc[:, j])
#         pearson_matrix[i, j] = pearson_corr

#         # Distance Correlation
#         distance_corr = dcor.distance_correlation(data.iloc[:, i], data.iloc[:, j])
#         distance_matrix[i, j] = distance_corr

# # Convert to DataFrame for better readability
# spearman_df = pd.DataFrame(spearman_matrix, index=features, columns=features)
# pearson_df = pd.DataFrame(pearson_matrix, index=features, columns=features)
# distance_df = pd.DataFrame(distance_matrix, index=features, columns=features)

# # Function to plot heatmaps
# def plot_heatmap(corr_df, title):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()

# # Plot the heatmaps
# print("Spearman Correlation Matrix:")
# plot_heatmap(spearman_df, "Spearman Correlation Matrix")

# print("Pearson Correlation Matrix:")
# plot_heatmap(pearson_df, "Pearson Correlation Matrix")

# print("Distance Correlation Matrix:")
# plot_heatmap(distance_df, "Distance Correlation Matrix")


'''
Continuous vs. binary features
'''
# Select the first 8 features (continuous)
continuous_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                       'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']

# Select the remaining features (assumed binary for this example)
binary_features = df.columns.difference(continuous_features)

# Initialize matrices to store results
ttest_matrix = np.zeros((len(continuous_features), len(binary_features)))
pb_matrix = np.zeros((len(continuous_features), len(binary_features)))
biserial_matrix = np.zeros((len(continuous_features), len(binary_features)))

# Define a function for Biserial Correlation (assumes latent continuous variable)
def biserial_correlation(x, y):
    x = np.array(x)
    y = np.array(y)
    mean_continuous_0 = np.mean(x[y == 0])
    mean_continuous_1 = np.mean(x[y == 1])
    std_continuous = np.std(x)
    p = np.mean(y)
    q = 1 - p
    return (mean_continuous_1 - mean_continuous_0) * np.sqrt(p * q) / std_continuous

# Compute the correlations
for i, cont_feat in enumerate(continuous_features):
    for j, bin_feat in enumerate(binary_features):
        # Ensure the binary feature is binary (0/1)
        unique_values = df[bin_feat].unique()
        if len(unique_values) == 2 and set(unique_values) <= {0, 1}:
            
            # T-Test
            group0 = df[df[bin_feat] == 0][cont_feat]
            group1 = df[df[bin_feat] == 1][cont_feat]
            t_stat, _ = ttest_ind(group0, group1, equal_var=False)  # Unequal variance assumed
            ttest_matrix[i, j] = t_stat

            # Point-Biserial Correlation
            pb_corr, _ = pointbiserialr(df[cont_feat], df[bin_feat])
            pb_matrix[i, j] = pb_corr

            # Biserial Correlation
            bis_corr = biserial_correlation(df[cont_feat], df[bin_feat])
            biserial_matrix[i, j] = bis_corr

# Convert results to DataFrame for better readability
ttest_df = pd.DataFrame(ttest_matrix, index=continuous_features, columns=binary_features)
pb_df = pd.DataFrame(pb_matrix, index=continuous_features, columns=binary_features)
biserial_df = pd.DataFrame(biserial_matrix, index=continuous_features, columns=binary_features)

# Function to plot heatmaps
def plot_heatmap(corr_df, title):
    plt.figure(figsize=(48, 40))
    sns.heatmap(corr_df, annot=False, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# # Plot the heatmaps
# print("T-Test Matrix:")
# plot_heatmap(ttest_df, "T-Test Correlation Matrix")

# print("\nPoint-Biserial Correlation Matrix:")
# plot_heatmap(pb_df, "Point-Biserial Correlation Matrix")

# print("\nBiserial Correlation Matrix:")
# plot_heatmap(biserial_df, "Biserial Correlation Matrix")

'''
Binary v Binary features
'''

# Initialize matrices to store results
chi2_matrix = np.zeros((len(binary_features), len(binary_features)))
phi_matrix = np.zeros((len(binary_features), len(binary_features)))

# Compute Chi-Squared and Phi Coefficient for binary-binary pairs
for i, bin_feat_1 in enumerate(binary_features):
    for j, bin_feat_2 in enumerate(binary_features):
        contingency_table = pd.crosstab(df[bin_feat_1], df[bin_feat_2])
        chi2, _, _, _ = chi2_contingency(contingency_table)
        chi2_matrix[i, j] = chi2

        # Compute Phi Coefficient
        n = contingency_table.values.sum()
        phi = np.sqrt(chi2 / n)
        phi_matrix[i, j] = phi

# Convert results to DataFrame for better readability
chi2_df = pd.DataFrame(chi2_matrix, index=binary_features, columns=binary_features)
phi_df = pd.DataFrame(phi_matrix, index=binary_features, columns=binary_features)

# Function to plot heatmaps
def plot_heatmap(corr_df, title):
    plt.figure(figsize=(48, 40))
    sns.heatmap(corr_df, annot=False, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plot the heatmaps
print("\nChi-Squared Matrix:")
plot_heatmap(chi2_df, "Chi-Squared Correlation Matrix")

print("\nPhi Coefficient Matrix:")
plot_heatmap(phi_df, "Phi Coefficient Correlation Matrix")
