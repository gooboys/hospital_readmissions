importances = {}
occurences = {}

with open("top20feat.csv", 'r') as file:
    for line in file:
        # Split the line by comma to get feature and its importance
        parts = line.strip().split(',')
        feature = parts[0]  # First value is the feature name
        importance = int(parts[1])  # Second value is the importance (convert to int)
        
        # Update the importances dictionary
        importances[feature] = importances.get(feature, 0) + importance
        occurences[feature] = occurences.get(feature, 0) + 1

index = {}

with open("zero2one_noshapremoved.csv", 'r') as file:
    first_line = file.readline().strip()  # Remove any trailing newline or spaces
    # Split the line by commas to process its content
    parts = first_line.split(',')
    count = 0
    for part in parts:
        index[part] = str(count)
        count += 1

# Sort by most occurrences (descending), then by importance (ascending)
sorted_importances = sorted(importances.items(), key=lambda x: (-occurences[x[0]], x[1]))

# Print the sorted result
for feature, importance in sorted_importances:
    print(feature + ": " + str(importance) + ", occurences: " + str(occurences[feature]))

print("")
print("")
# Print the index of features for later analysis
for feature, importance in sorted_importances:
    print(feature + ": " + index[feature])