importances = {}

with open("top20feat.csv", 'r') as file:
    for line in file:
        # Split the line by comma to get feature and its importance
        parts = line.strip().split(',')
        feature = parts[0]  # First value is the feature name
        importance = int(parts[1])  # Second value is the importance (convert to int)
        
        # Update the importances dictionary
        importances[feature] = importances.get(feature, 0) + importance

# Sort the dictionary by its values (from least to greatest)
sorted_importances = sorted(importances.items(), key=lambda x: x[1])

# Print the sorted result
for feature, importance in sorted_importances:
    print(feature + ": " + str(importance))