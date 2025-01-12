import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("train.csv")

target_column = "readmitted"

# Separate features and target
X = data.drop(columns=[target_column]).values  # Features as a NumPy array
y = data[target_column].values  # Target as a NumPy array

# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
''''''

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
feature_names = data.drop(columns=[target_column]).columns
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc)

print(feature_importance_df)

''''''

# # Train the Gradient Boosting classifier
# gb_clf = GradientBoostingClassifier(
#     n_estimators=100,      # Number of boosting stages
#     learning_rate=0.1,     # Step size shrinkage
#     max_depth=4,           # Maximum depth of each tree
#     random_state=42        # For reproducibility
# )
# gb_clf.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = gb_clf.predict(X_test)
# y_pred_proba = gb_clf.predict_proba(X_test)[:, 1]

# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Feature Importance
# importances = gb_clf.feature_importances_
# feature_names = data.drop(columns=[target_column]).columns
# feature_importance_df = pd.DataFrame({
#     "Feature": feature_names,
#     "Importance": importances
# }).sort_values(by="Importance", ascending=False)

# y_pred_proba = gb_clf.predict_proba(X_test)[:, 1]
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# print("ROC-AUC Score:", roc_auc)

# # Set maximum rows to display to None (displays all rows)
# pd.set_option('display.max_rows', None)
# # print(feature_importance_df)
# # Optionally, reset the display option after printing
# pd.reset_option('display.max_rows')

''''''
# Implementing grid search

# Define the parameter grid for RandomForestClassifier
param_grid_rf = {
    'n_estimators': [100, 200, 500],  # Number of trees
    'max_depth': [None, 10, 20],      # Maximum depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],    # Minimum samples at a leaf node
    'class_weight': ['balanced', None]  # Handle class imbalance
}

# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid_rf,
    scoring='roc_auc',  # Use ROC-AUC as the evaluation metric
    cv=5,               # 5-fold cross-validation
    n_jobs=-1,          # Use all available processors
    verbose=2           # Show progress
)


# Fit GridSearchCV to the training data
# grid_search_rf.fit(X_train, y_train)

# # Print the best parameters and best score
# print("Best Parameters:", grid_search_rf.best_params_)
# print("Best ROC-AUC Score:", grid_search_rf.best_score_)

# # Train the final model with the best parameters
# best_rf_clf = grid_search_rf.best_estimator_
# best_rf_clf.fit(X_train, y_train)

# # Evaluate on the test set
# y_pred = best_rf_clf.predict(X_test)
# y_pred_proba = best_rf_clf.predict_proba(X_test)[:, 1]

# print("Final Test Accuracy:", accuracy_score(y_test, y_pred))
# print("Final Test Classification Report:\n", classification_report(y_test, y_pred))
# print("Final Test ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))